"""
Weather tool — standalone module.

Provides weather forecasts for the device's current location
with hourly and daily forecast support. Location is auto-detected via IP
geolocation and cached for future use.
"""

import json
import logging
import re
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from tools.models import WeatherArgs
from clients.weather_client import WeatherClient

logger = logging.getLogger(__name__)

LOCATION_CACHE_FILE = (
    Path(__file__).parent.parent.parent
    / "homeassist-ref"
    / "scripts"
    / "scheduled"
    / "weather_briefing"
    / "location_cache.json"
)

_weather_client = WeatherClient()
_cached_location: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------


def _get_location_from_ip() -> Optional[Dict[str, Any]]:
    """Get location from IP address using ipinfo.io with fallbacks."""
    # Primary: ipinfo.io
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc", "")
            if loc and "," in loc:
                lat_str, lon_str = loc.split(",")
                lat, lon = float(lat_str), float(lon_str)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return {
                        "lat": lat,
                        "lon": lon,
                        "city": data.get("city", "") or "",
                        "region": data.get("region", "") or "",
                        "zip_code": data.get("postal", "") or "",
                        "source": "ipinfo.io",
                    }
    except Exception:
        pass

    # Fallback: ip-api.com
    try:
        response = requests.get(
            "http://ip-api.com/json/?fields=lat,lon,city,regionName,zip,status",
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") != "fail":
                lat, lon = data.get("lat"), data.get("lon")
                if lat is not None and lon is not None:
                    return {
                        "lat": float(lat),
                        "lon": float(lon),
                        "city": data.get("city", "") or "",
                        "region": data.get("regionName", "") or "",
                        "zip_code": str(data.get("zip", "") or ""),
                        "source": "ip-api.com",
                    }
    except Exception:
        pass

    return None


def _load_cached_location() -> Optional[Dict[str, Any]]:
    """Load location from cache file if it exists."""
    try:
        if LOCATION_CACHE_FILE.exists():
            with open(LOCATION_CACHE_FILE) as f:
                data = json.load(f)
                if "lat" in data and "lon" in data:
                    return data
    except Exception:
        pass
    return None


def _save_location_to_cache(location: Dict[str, Any]) -> None:
    """Save location to cache file."""
    try:
        LOCATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCATION_CACHE_FILE, "w") as f:
            json.dump(location, f, indent=2)
    except Exception:
        pass


def _get_device_location() -> Optional[Dict[str, Any]]:
    """Get the device's current location, using cache or IP geolocation."""
    global _cached_location

    if _cached_location:
        return _cached_location

    cached = _load_cached_location()
    if cached:
        _cached_location = cached
        return cached

    location = _get_location_from_ip()
    if location:
        _cached_location = location
        _save_location_to_cache(location)
        return location

    return None


def _get_location_display_name(location: Dict[str, Any]) -> str:
    """Get a human-readable location name."""
    city = location.get("city", "")
    region = location.get("region", "")
    if city and region:
        return f"{city}, {region}"
    elif city:
        return city
    elif location.get("zip_code"):
        return f"ZIP {location['zip_code']}"
    else:
        return f"({location['lat']:.2f}, {location['lon']:.2f})"


# ---------------------------------------------------------------------------
# Timeframe / validation helpers
# ---------------------------------------------------------------------------


def _validate_parameters(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate parameters; return error dict or None if valid."""
    hours = params.get("hours")
    days = params.get("days")
    specific_date = params.get("specific_date")

    provided_params = sum(
        [hours is not None, days is not None, specific_date is not None]
    )
    if provided_params != 1:
        return {
            "success": False,
            "error": (
                "Provide exactly one timeframe: either 'hours' (1-168), "
                "'days' (1-7), or 'specific_date' (within next 7 days)"
            ),
        }

    if hours is not None:
        try:
            h = int(hours)
        except Exception:
            return {"success": False, "error": "'hours' must be an integer"}
        if h < 1 or h > 168:
            return {"success": False, "error": "'hours' must be between 1 and 168"}

    if days is not None:
        try:
            d = int(days)
        except Exception:
            return {"success": False, "error": "'days' must be an integer"}
        if d < 1 or d > 7:
            return {"success": False, "error": "'days' must be between 1 and 7"}

    return None


def _normalize_timeframe(
    hours: Optional[int], days: Optional[int]
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Apply the 36-hour rule and max day cap.
    Returns: (mode, normalized_hours, normalized_days)
    """
    if hours is not None and days is not None:
        raise ValueError("Provide either hours or days, not both")

    if hours is not None:
        h = max(1, min(168, int(hours)))
        if h <= 36:
            return "hourly", h, None
        d = ceil(h / 24)
        d = max(1, min(7, d))
        return "daily", None, d

    if days is not None:
        d = max(1, min(7, int(days)))
        if d * 24 <= 36:
            h = d * 24
            return "hourly", h, None
        return "daily", None, d

    raise ValueError("Missing timeframe: provide hours or days")


def _parse_specific_date(date_str: str) -> Optional[int]:
    """
    Parse a specific date string and return the number of days from today.
    Returns None if the date is invalid or not within 7 days.
    """
    today = datetime.now().date()
    date_str_lower = date_str.lower().strip()

    if date_str_lower == "today":
        return 1
    elif date_str_lower == "tomorrow":
        return 2

    weekdays = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]
    if date_str_lower in weekdays:
        target_weekday = weekdays.index(date_str_lower)
        current_weekday = today.weekday()
        days_ahead = (target_weekday - current_weekday) % 7
        if days_ahead == 0:
            days_ahead = 7
        if days_ahead > 7:
            return None
        return days_ahead + 1

    if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            delta = (target_date - today).days
            if 0 <= delta <= 7:
                return delta + 1
            else:
                return None
        except ValueError:
            return None

    return None


def _get_current_timestamp() -> str:
    """Get current timestamp as an ISO 8601 string."""
    try:
        return datetime.now().isoformat()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: WeatherArgs) -> dict:
    """Execute the weather tool with the provided parameters."""
    try:
        params = args.model_dump()
        logger.debug("Executing Tool: Weather -- %s", params)

        location = _get_device_location()
        if not location:
            return {
                "success": False,
                "error": "Could not detect device location. Check network connection.",
            }

        location_name = _get_location_display_name(location)
        lat, lon = location["lat"], location["lon"]

        validation_error = _validate_parameters(params)
        if validation_error:
            return validation_error

        hours = args.hours
        days = args.days
        specific_date = args.specific_date

        if specific_date:
            date_days = _parse_specific_date(specific_date)
            if date_days is None:
                return {
                    "success": False,
                    "error": f"Invalid or out-of-range date: '{specific_date}'. Must be within next 7 days.",
                    "location": location_name,
                }
            mode = "specific_date"
            normalized_hours, normalized_days = None, date_days
            daily = _weather_client.fetch_open_meteo_daily(
                lat, lon, days=date_days, units="imperial"
            )
            if daily and isinstance(daily, dict):
                day_idx = date_days - 1
                specific_daily = {}
                for key, values in daily.items():
                    if isinstance(values, list) and len(values) > day_idx:
                        specific_daily[key] = [values[day_idx]]
                    else:
                        specific_daily[key] = values
                forecast_data = {"daily": specific_daily}
            else:
                forecast_data = {"daily": daily}
        else:
            mode, normalized_hours, normalized_days = _normalize_timeframe(hours, days)

            if mode == "hourly":
                hourly = _weather_client.fetch_open_meteo_hourly(
                    lat, lon, hours=normalized_hours, units="imperial"
                )
                forecast_data = {"hourly": hourly}
            else:
                daily = _weather_client.fetch_open_meteo_daily(
                    lat, lon, days=normalized_days, units="imperial"
                )
                forecast_data = {"daily": daily}

        response: Dict[str, Any] = {
            "success": True,
            "location": location_name,
            "coordinates": {"lat": lat, "lon": lon},
            "zip_code": location.get("zip_code"),
            "mode": mode,
            "timeframe_hours": normalized_hours if mode == "hourly" else None,
            "timeframe_days": normalized_days if mode in ("daily", "specific_date") else None,
            "specific_date": specific_date if specific_date else None,
            "units": "imperial",
            "forecast": forecast_data,
            "timestamp": _get_current_timestamp(),
        }

        return response

    except Exception as e:
        logger.error("Error executing weather tool: %s", e)
        return {
            "success": False,
            "error": str(e),
        }
