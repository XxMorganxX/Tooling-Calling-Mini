import requests
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional


class WeatherClient:
    """
    Simple client to fetch raw daily weather forecasts for US ZIP codes.

    Implementation details:
    - ZIP -> lat/lon via Zippopotam.us (no API key required)
    - Daily forecast via Open-Meteo (no API key required)

    Returned data is intentionally "raw" (as provided by Open-Meteo's `daily` object),
    suitable for downstream formatting or modeling.
    """

    ZIPPOTAM_BASE_URL = "https://api.zippopotam.us/us"
    OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_TIMEOUT_SEC = 10

    def __init__(self, user_agent: Optional[str] = None, timeout_sec: Optional[float] = None) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or "RasPi-Smart-Home/1.0 (weather_client)"
        })
        self.timeout_sec = timeout_sec or self.DEFAULT_TIMEOUT_SEC

    @staticmethod
    def _validate_zip_code(zip_code: str) -> str:
        """
        Ensure the provided ZIP code is a 5-digit US ZIP code string.
        """
        zip_str = str(zip_code).strip()
        if len(zip_str) != 5 or not zip_str.isdigit():
            raise ValueError(f"Invalid US ZIP code: {zip_code}")
        return zip_str

    @lru_cache(maxsize=256)
    def zip_to_latlon(self, zip_code: str) -> Tuple[float, float]:
        """
        Convert a US ZIP code to (latitude, longitude) using Zippopotam.us.
        """
        zip_str = self._validate_zip_code(zip_code)
        url = f"{self.ZIPPOTAM_BASE_URL}/{zip_str}"
        response = self.session.get(url, timeout=self.timeout_sec)
        if response.status_code != 200:
            raise RuntimeError(
                f"Geocoding failed for ZIP {zip_str}: {response.status_code} {response.text[:200]}"
            )
        payload = response.json()
        places = payload.get("places") or []
        if not places:
            raise LookupError(f"No geocoding results for ZIP {zip_str}")
        latitude = float(places[0]["latitude"])  # strings in API, convert to float
        longitude = float(places[0]["longitude"])  # strings in API, convert to float
        return latitude, longitude

    def _clamp_days(self, days: int) -> int:
        """
        Clamp requested days to the supported range [1, 7].
        """
        try:
            value = int(days)
        except Exception:
            value = 1
        return max(1, min(7, value))

    def fetch_open_meteo_daily(
        self,
        latitude: float,
        longitude: float,
        days: int,
        units: str = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch raw daily forecast data from Open-Meteo for the given coordinates.
        Returns the `daily` object from the API response.
        """
        clamped_days = self._clamp_days(days)
        params: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "weathercode",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "sunrise",
                "sunset",
            ]),
            "timezone": "auto",
            "forecast_days": clamped_days,
        }
        if str(units).lower() == "imperial":
            params.update({
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
            })
        response = self.session.get(self.OPEN_METEO_BASE_URL, params=params, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily")
        if not daily:
            raise RuntimeError("Daily forecast not available in Open-Meteo response")
        return daily

    def get_daily_forecast(self, zip_code: str, days: int = 7, units: str = "auto") -> Dict[str, Any]:
        """
        Get raw daily forecast for a single US ZIP code.

        Returns a dictionary with keys: `zip_code`, `latitude`, `longitude`, `days`, `daily`.
        The `daily` value is the raw Open-Meteo `daily` object.
        """
        latitude, longitude = self.zip_to_latlon(zip_code)
        daily = self.fetch_open_meteo_daily(latitude, longitude, days, units=units)
        return {
            "zip_code": self._validate_zip_code(zip_code),
            "latitude": latitude,
            "longitude": longitude,
            "days": self._clamp_days(days),
            "daily": daily,
            "units": units,
        }

    def get_daily_forecast_for_zipcodes(self, zip_codes: List[str], days: int = 7, units: str = "auto") -> Dict[str, Dict[str, Any]]:
        """
        Get raw daily forecasts for multiple US ZIP codes.
        Returns a mapping: ZIP -> forecast dict or error dict.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for zip_code in zip_codes:
            try:
                results[zip_code] = self.get_daily_forecast(zip_code, days, units=units)
            except Exception as exc:
                results[zip_code] = {"error": str(exc)}
        return results
    
    def get_specific_date_forecast(self, zip_code: str, day_index: int, units: str = "auto") -> Dict[str, Any]:
        """
        Get weather forecast for a specific day (1-based index, where 1 is today).
        Fetches full forecast and extracts the specific day's data.
        """
        days_to_fetch = min(day_index, 7)
        forecast = self.get_daily_forecast(zip_code, days_to_fetch, units=units)
        
        daily = forecast.get("daily", {})
        if daily and isinstance(daily, dict):
            day_idx = day_index - 1  # Convert to 0-based index
            specific_daily = {}
            for key, values in daily.items():
                if isinstance(values, list) and len(values) > day_idx:
                    specific_daily[key] = [values[day_idx]]
                else:
                    specific_daily[key] = values
            
            forecast["daily"] = specific_daily
            forecast["days"] = 1
            forecast["specific_day_index"] = day_index
        
        return forecast
    
    def get_specific_date_forecast_for_zipcodes(self, zip_codes: List[str], day_index: int, units: str = "auto") -> Dict[str, Dict[str, Any]]:
        """
        Get weather forecast for a specific day for multiple ZIP codes.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for zip_code in zip_codes:
            try:
                results[zip_code] = self.get_specific_date_forecast(zip_code, day_index, units=units)
            except Exception as exc:
                results[zip_code] = {"error": str(exc)}
        return results

    # -----------------------
    # Hourly forecast support
    # -----------------------
    def _clamp_hours(self, hours: int) -> int:
        """
        Clamp requested hours to the supported range [1, 36].
        """
        try:
            value = int(hours)
        except Exception:
            value = 1
        return max(1, min(36, value))

    def fetch_open_meteo_hourly(
        self,
        latitude: float,
        longitude: float,
        hours: int,
        units: str = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch raw hourly forecast data from Open-Meteo for the given coordinates.
        Returns the `hourly` object (potentially truncated to requested hours).
        """
        clamped_hours = self._clamp_hours(hours)
        params: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join([
                "temperature_2m",
                "precipitation",
                "precipitation_probability",
                "weathercode",
                "wind_speed_10m",
                "wind_gusts_10m",
                "relative_humidity_2m",
            ]),
            "timezone": "auto",
        }
        if str(units).lower() == "imperial":
            params.update({
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
            })
        response = self.session.get(self.OPEN_METEO_BASE_URL, params=params, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly")
        if not hourly:
            raise RuntimeError("Hourly forecast not available in Open-Meteo response")
        try:
            times: List[str] = hourly.get("time", [])
            cutoff = min(len(times), clamped_hours)
            sliced: Dict[str, Any] = {}
            for key, series in hourly.items():
                if isinstance(series, list):
                    sliced[key] = series[:cutoff]
                else:
                    sliced[key] = series
            return sliced
        except Exception:
            return hourly

    def get_hourly_forecast(self, zip_code: str, hours: int = 24, units: str = "auto") -> Dict[str, Any]:
        """
        Get raw hourly forecast for a single US ZIP code.

        Returns a dictionary with keys: `zip_code`, `latitude`, `longitude`, `hours`, `hourly`.
        """
        latitude, longitude = self.zip_to_latlon(zip_code)
        hourly = self.fetch_open_meteo_hourly(latitude, longitude, hours, units=units)
        return {
            "zip_code": self._validate_zip_code(zip_code),
            "latitude": latitude,
            "longitude": longitude,
            "hours": self._clamp_hours(hours),
            "hourly": hourly,
            "units": units,
        }

    def get_hourly_forecast_for_zipcodes(self, zip_codes: List[str], hours: int = 24, units: str = "auto") -> Dict[str, Dict[str, Any]]:
        """
        Get raw hourly forecasts for multiple US ZIP codes.
        Returns a mapping: ZIP -> forecast dict or error dict.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for zip_code in zip_codes:
            try:
                results[zip_code] = self.get_hourly_forecast(zip_code, hours, units=units)
            except Exception as exc:
                results[zip_code] = {"error": str(exc)}
        return results
