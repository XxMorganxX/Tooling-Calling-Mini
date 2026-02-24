"""
Spotify Playback Tool – standalone module.

Ported from MCP BaseTool class to module-level functions.
Controls Spotify playback via spotipy with per-user OAuth caching.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

from tools.models import SpotifyPlaybackArgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sp_clients: Dict[str, spotipy.Spotify] = {}
_device_cache: Dict[str, Any] = {}
_devices_cache_time: Dict[str, float] = {}
_CACHE_TIMEOUT = 30  # seconds

_available_actions = [
    "play", "pause", "next", "previous", "volume",
    "search_track", "search_artist", "status", "devices",
    "shuffle", "repeat",
]

# User / credential config from env
_DEFAULT_USER = os.getenv("DEFAULT_SPOTIFY_USER", "morgan").lower()

_SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"
_TARGET_DEVICE_ID = os.getenv("SPOTIFY_DEVICE_ID")

# Credentials dicts keyed by lowercase username
_SPOTIPY_CLIENT_ID: Dict[str, Optional[str]] = {}
_SPOTIPY_CLIENT_SECRET: Dict[str, Optional[str]] = {}
_SPOTIPY_REDIRECT_URI: Dict[str, str] = {}
_configured_users: List[str] = []


def _init_credentials() -> None:
    """Populate credential dicts from environment variables."""
    global _configured_users

    users_env = os.getenv("SPOTIFY_USERS", _DEFAULT_USER)
    _configured_users = [u.strip().lower() for u in users_env.split(",") if u.strip()]
    if _DEFAULT_USER not in _configured_users:
        _configured_users.insert(0, _DEFAULT_USER)

    default_redirect = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    for user in _configured_users:
        if user == _DEFAULT_USER:
            _SPOTIPY_CLIENT_ID[user] = os.getenv("SPOTIFY_CLIENT_ID")
            _SPOTIPY_CLIENT_SECRET[user] = os.getenv("SPOTIFY_CLIENT_SECRET")
        else:
            prefix = user.upper()
            _SPOTIPY_CLIENT_ID[user] = os.getenv(f"{prefix}_SPOTIFY_CLIENT_ID")
            _SPOTIPY_CLIENT_SECRET[user] = os.getenv(f"{prefix}_SPOTIFY_CLIENT_SECRET")
        _SPOTIPY_REDIRECT_URI[user] = os.getenv(
            f"{user.upper()}_SPOTIFY_REDIRECT_URI", default_redirect
        )


_init_credentials()


def _available_users() -> List[str]:
    return list(_configured_users)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(args: SpotifyPlaybackArgs) -> dict:
    """Execute Spotify playback control."""
    try:
        params = args.model_dump()
        action = params.get("action")
        user = (params.get("user") or _DEFAULT_USER).lower()

        if not action:
            return {
                "success": False,
                "error": "Missing required parameter: action",
                "available_actions": _available_actions,
                "available_users": _available_users(),
            }

        if action not in _available_actions:
            return {
                "success": False,
                "error": (
                    f"Invalid Spotify action '{action}'. "
                    f"Available actions: {', '.join(_available_actions)}"
                ),
            }

        if user not in _available_users():
            return {
                "success": False,
                "error": (
                    f"Unknown Spotify user '{user}'. "
                    f"Available users: {', '.join(_available_users())}"
                ),
            }

        validation_error = _validate_action_parameters(action, params)
        if validation_error:
            return validation_error

        sp_client = _get_spotify_client(user)
        if not sp_client:
            project_root = Path(__file__).parent.parent.parent
            cache_file = project_root / "creds" / ".spotify_cache"
            cache_exists = cache_file.exists()
            has_credentials = bool(
                _SPOTIPY_CLIENT_ID.get(user) and _SPOTIPY_CLIENT_SECRET.get(user)
            )

            if not cache_exists:
                error_msg = (
                    "Spotify not authenticated. Please run 'python scripts/authenticate_spotify.py' "
                    "first to authorize Spotify access."
                )
            elif not has_credentials:
                error_msg = (
                    "Spotify credentials missing. Please check your .env file has "
                    "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (without spaces around '=')."
                )
            else:
                error_msg = (
                    "Spotify authentication failed. Please re-run the authentication script "
                    "to refresh your access tokens."
                )

            return {
                "success": False,
                "error": error_msg,
                "user": user,
                "authentication_required": not cache_exists,
                "credentials_missing": not has_credentials,
            }

        result = _execute_spotify_action(sp_client, action, params, user)
        result.update({
            "user": user,
            "action": action,
            "timestamp": _get_current_timestamp(),
        })
        return result

    except Exception as e:
        logger.error("Error executing Spotify control: %s", e)
        return {
            "success": False,
            "error": f"Spotify control failed: {str(e)}",
            "action": args.action if args else "unknown",
            "user": (args.user if args else None) or "unknown",
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_action_parameters(action: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if action in ["search_track", "search_artist"]:
        if not params.get("query"):
            return {
                "success": False,
                "error": f"Action '{action}' requires 'query' parameter",
                "example": "For search_track: 'Bohemian Rhapsody', for search_artist: 'Queen'",
            }
    elif action == "volume":
        volume = params.get("volume_level")
        if volume is None:
            return {
                "success": False,
                "error": "Action 'volume' requires 'volume_level' parameter (0-100)",
            }
        if not isinstance(volume, int) or not (0 <= volume <= 100):
            return {
                "success": False,
                "error": "volume_level must be an integer between 0 and 100",
            }
    return None


# ---------------------------------------------------------------------------
# Client creation
# ---------------------------------------------------------------------------


def _get_spotify_client(user: str) -> Optional[spotipy.Spotify]:
    if user in _sp_clients:
        return _sp_clients[user]

    try:
        client_id = _SPOTIPY_CLIENT_ID.get(user)
        client_secret = _SPOTIPY_CLIENT_SECRET.get(user)
        redirect_uri = _SPOTIPY_REDIRECT_URI.get(user)

        if not all([client_id, client_secret, redirect_uri]):
            missing = []
            if not client_id:
                missing.append("CLIENT_ID")
            if not client_secret:
                missing.append("CLIENT_SECRET")
            if not redirect_uri:
                missing.append("REDIRECT_URI")
            logger.error("Missing Spotify credentials for %s: %s", user, ", ".join(missing))
            return None

        project_root = Path(__file__).parent.parent.parent
        cache_path = str(project_root / "creds" / ".spotify_cache")

        if not os.path.exists(cache_path):
            logger.error("Spotify authentication cache not found at: %s", cache_path)
            return None

        cache_handler = CacheFileHandler(cache_path=cache_path)

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=_SCOPE,
            cache_handler=cache_handler,
            open_browser=False,
        )

        token_info = auth_manager.get_cached_token()
        if not token_info:
            logger.error("No valid token in cache file")
            return None

        if auth_manager.is_token_expired(token_info):
            logger.info("Token expired, attempting refresh...")
            try:
                token_info = auth_manager.refresh_access_token(token_info["refresh_token"])
                logger.info("Token refreshed successfully")
            except Exception as refresh_error:
                logger.error("Failed to refresh token: %s", refresh_error)
                return None

        sp = spotipy.Spotify(auth_manager=auth_manager)
        _sp_clients[user] = sp

        try:
            info = sp.current_user()
            display_name = (info or {}).get("display_name")
            user_id = (info or {}).get("id")
            logger.info("Spotify client ready for %s: %s (%s)", user, display_name, user_id)
        except Exception as test_error:
            logger.error("Failed to verify Spotify connection: %s", test_error)
            return None

        return sp

    except Exception as e:
        logger.error("Failed to create Spotify client for %s: %s", user, e)
        return None


# ---------------------------------------------------------------------------
# Action dispatcher
# ---------------------------------------------------------------------------


def _execute_spotify_action(
    sp_client: spotipy.Spotify,
    action: str,
    params: Dict[str, Any],
    user: str,
) -> Dict[str, Any]:
    try:
        if action == "play":
            if params.get("query"):
                search_type = params.get("search_type", "track")
                if search_type == "artist":
                    return _handle_search_artist(sp_client, params["query"], params.get("search_limit", 5))
                return _handle_search_track(sp_client, params["query"], params.get("search_limit", 5))
            return _handle_play(sp_client, params)
        elif action == "pause":
            return _handle_pause(sp_client)
        elif action == "next":
            return _handle_next(sp_client)
        elif action == "previous":
            return _handle_previous(sp_client)
        elif action == "volume":
            return _handle_volume(sp_client, params["volume_level"])
        elif action == "search_track":
            return _handle_search_track(sp_client, params["query"], params.get("search_limit", 5))
        elif action == "search_artist":
            return _handle_search_artist(sp_client, params["query"], params.get("search_limit", 5))
        elif action == "status":
            return _handle_status(sp_client)
        elif action == "devices":
            return _handle_devices(sp_client)
        elif action == "shuffle":
            return _handle_shuffle(sp_client, params.get("shuffle_state"))
        elif action == "repeat":
            return _handle_repeat(sp_client, params.get("repeat_mode"))
        else:
            return {"success": False, "error": f"Action '{action}' not implemented"}
    except Exception as e:
        return {"success": False, "error": f"Failed to execute {action}: {str(e)}"}


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_play(sp_client: spotipy.Spotify, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        device_id = params.get("device_id") or _ensure_active_device(sp_client)

        if not device_id:
            return {
                "success": False,
                "error": "No Spotify device available. Please open Spotify on your Mac or another device first.",
                "hint": "Run this tool with action 'devices' after opening Spotify to see available devices.",
            }

        retried = False
        try:
            sp_client.start_playback(device_id=device_id)
        except SpotifyException as e:
            if e.http_status == 404 and not retried:
                retried = True
                try:
                    sp_client.transfer_playback([device_id], force_play=False)
                    time.sleep(0.3)
                    sp_client.start_playback(device_id=device_id)
                except SpotifyException:
                    raise e
            else:
                raise

        current = sp_client.current_playback()
        track_info = _extract_track_info(current) if current else {}

        return {
            "success": True,
            "message": "Playback started",
            "current_track": track_info,
            "device_id": device_id,
        }
    except SpotifyException as e:
        error_str = str(e).lower()
        if "403" in error_str and "restriction" in error_str:
            return {
                "success": False,
                "error": "Cannot resume playback: Spotify restrictions apply. Try using Spotify Premium or a different device.",
                "details": str(e),
            }
        elif e.http_status == 404 or "404" in error_str:
            return {
                "success": False,
                "error": "No active Spotify device found. Please open Spotify on a device first.",
                "hint": "Make sure Spotify is open and has played something recently.",
                "details": str(e),
            }
        else:
            return {"success": False, "error": f"Failed to start playback: {str(e)}", "details": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to start playback: {str(e)}", "details": str(e)}


def _handle_pause(sp_client: spotipy.Spotify) -> Dict[str, Any]:
    try:
        sp_client.pause_playback()
        return {"success": True, "message": "Playback paused"}
    except Exception as e:
        error_str = str(e).lower()
        if "403" in error_str and "restriction" in error_str:
            return {
                "success": False,
                "error": "Cannot pause: Spotify playback restrictions apply to this device or content. Try using a Spotify Premium account or a different playback device.",
                "details": str(e),
            }
        elif "404" in error_str:
            return {
                "success": False,
                "error": "No active Spotify device found. Please start playing music on a Spotify device first.",
                "details": str(e),
            }
        else:
            return {"success": False, "error": f"Failed to pause playback: {str(e)}", "details": str(e)}


def _handle_next(sp_client: spotipy.Spotify) -> Dict[str, Any]:
    sp_client.next_track()
    time.sleep(0.5)
    current = sp_client.current_playback()
    track_info = _extract_track_info(current) if current else {}
    return {"success": True, "message": "Skipped to next track", "current_track": track_info}


def _handle_previous(sp_client: spotipy.Spotify) -> Dict[str, Any]:
    sp_client.previous_track()
    time.sleep(0.5)
    current = sp_client.current_playback()
    track_info = _extract_track_info(current) if current else {}
    return {"success": True, "message": "Skipped to previous track", "current_track": track_info}


def _handle_volume(sp_client: spotipy.Spotify, volume_level: int) -> Dict[str, Any]:
    sp_client.volume(volume_level)
    return {"success": True, "message": f"Volume set to {volume_level}%", "volume_level": volume_level}


def _handle_search_track(sp_client: spotipy.Spotify, query: str, limit: int) -> Dict[str, Any]:
    device_id = _ensure_active_device(sp_client)

    if not device_id:
        return {
            "success": False,
            "error": "No Spotify device available. Please open Spotify on your Mac or another device first.",
            "hint": "Run this tool with action 'devices' after opening Spotify to see available devices.",
            "query": query,
        }

    results = sp_client.search(q=query, type="track", limit=limit)
    tracks = results["tracks"]["items"]

    if not tracks:
        return {"success": False, "message": f"No tracks found for query: '{query}'", "query": query}

    track = tracks[0]
    try:
        sp_client.start_playback(uris=[track["uri"]], device_id=device_id)
    except SpotifyException as e:
        if e.http_status == 404:
            try:
                sp_client.transfer_playback([device_id], force_play=False)
                time.sleep(0.3)
                sp_client.start_playback(uris=[track["uri"]], device_id=device_id)
            except SpotifyException:
                return {
                    "success": False,
                    "error": f"Found track but could not start playback: {str(e)}",
                    "track_found": _extract_track_info_from_item(track),
                    "query": query,
                }
        else:
            raise

    return {
        "success": True,
        "message": f"Playing track: {track['name']} by {', '.join(a['name'] for a in track['artists'])}",
        "track_played": _extract_track_info_from_item(track),
        "total_results": len(tracks),
        "query": query,
        "device_id": device_id,
    }


def _handle_search_artist(sp_client: spotipy.Spotify, query: str, limit: int) -> Dict[str, Any]:
    device_id = _ensure_active_device(sp_client)

    if not device_id:
        return {
            "success": False,
            "error": "No Spotify device available. Please open Spotify on your Mac or another device first.",
            "hint": "Run this tool with action 'devices' after opening Spotify to see available devices.",
            "query": query,
        }

    results = sp_client.search(q=query, type="artist", limit=limit)
    artists = results["artists"]["items"]

    if not artists:
        return {"success": False, "message": f"No artists found for query: '{query}'", "query": query}

    artist = artists[0]
    top_tracks = sp_client.artist_top_tracks(artist["id"])["tracks"]

    if top_tracks:
        track = top_tracks[0]
        try:
            sp_client.start_playback(uris=[track["uri"]], device_id=device_id)
        except SpotifyException as e:
            if e.http_status == 404:
                try:
                    sp_client.transfer_playback([device_id], force_play=False)
                    time.sleep(0.3)
                    sp_client.start_playback(uris=[track["uri"]], device_id=device_id)
                except SpotifyException:
                    return {
                        "success": False,
                        "error": f"Found artist but could not start playback: {str(e)}",
                        "artist": artist["name"],
                        "track_found": _extract_track_info_from_item(track),
                        "query": query,
                    }
            else:
                raise

        return {
            "success": True,
            "message": f"Playing top track by {artist['name']}: {track['name']}",
            "artist": artist["name"],
            "track_played": _extract_track_info_from_item(track),
            "query": query,
            "device_id": device_id,
        }
    else:
        return {
            "success": False,
            "message": f"No playable tracks found for artist: {artist['name']}",
            "artist": artist["name"],
            "query": query,
        }


def _handle_status(sp_client: spotipy.Spotify) -> Dict[str, Any]:
    current = sp_client.current_playback()

    if not current:
        return {"success": True, "message": "No active playback", "is_playing": False}

    track_info = _extract_track_info(current)
    return {
        "success": True,
        "is_playing": current["is_playing"],
        "current_track": track_info,
        "device": current["device"]["name"] if current.get("device") else "Unknown",
        "volume": current["device"]["volume_percent"] if current.get("device") else None,
        "shuffle": current.get("shuffle_state", False),
        "repeat": current.get("repeat_state", "off"),
    }


def _handle_devices(sp_client: spotipy.Spotify) -> Dict[str, Any]:
    devices = sp_client.devices()["devices"]
    device_list = [
        {
            "id": d["id"],
            "name": d["name"],
            "type": d["type"],
            "is_active": d["is_active"],
            "volume": d["volume_percent"],
        }
        for d in devices
    ]
    return {"success": True, "devices": device_list, "total_devices": len(device_list)}


def _handle_shuffle(sp_client: spotipy.Spotify, shuffle_state: Optional[bool]) -> Dict[str, Any]:
    if shuffle_state is None:
        current = sp_client.current_playback()
        current_shuffle = current.get("shuffle_state", False) if current else False
        new_shuffle = not current_shuffle
    else:
        new_shuffle = shuffle_state

    sp_client.shuffle(new_shuffle)
    return {
        "success": True,
        "message": f"Shuffle {'enabled' if new_shuffle else 'disabled'}",
        "shuffle_state": new_shuffle,
    }


def _handle_repeat(sp_client: spotipy.Spotify, repeat_mode: Optional[str]) -> Dict[str, Any]:
    if repeat_mode is None:
        current = sp_client.current_playback()
        current_repeat = current.get("repeat_state", "off") if current else "off"
        cycle = {"off": "context", "context": "track", "track": "off"}
        new_repeat = cycle.get(current_repeat, "off")
    else:
        new_repeat = repeat_mode

    sp_client.repeat(new_repeat)
    return {
        "success": True,
        "message": f"Repeat mode set to {new_repeat}",
        "repeat_mode": new_repeat,
    }


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def _ensure_active_device(sp_client: spotipy.Spotify) -> Optional[str]:
    """Ensure a device is active for playback. Returns device_id or None."""
    try:
        devices_response = sp_client.devices()
        devices = devices_response.get("devices", [])
        logger.info("Found %d Spotify device(s)", len(devices))

        for device in devices:
            if device.get("is_active"):
                logger.info("Found active device: %s", device["name"])
                return device["id"]

        target_device_id = _TARGET_DEVICE_ID

        if target_device_id:
            target_in_list = any(d["id"] == target_device_id for d in devices)
            if target_in_list:
                logger.info("Transferring playback to target device")
                try:
                    sp_client.transfer_playback([target_device_id], force_play=False)
                    time.sleep(0.3)
                    return target_device_id
                except SpotifyException as e:
                    logger.warning("Transfer playback failed: %s", e)
                    return target_device_id

        if devices and not target_device_id:
            first_device = devices[0]
            logger.info("No target device configured, activating first available: %s", first_device["name"])
            try:
                sp_client.transfer_playback([first_device["id"]], force_play=False)
                time.sleep(0.3)
                return first_device["id"]
            except SpotifyException as e:
                logger.warning("Transfer playback failed: %s", e)
                return first_device["id"]

        if not devices:
            logger.info("No devices found – attempting to open Spotify app...")
            if _open_spotify_app():
                time.sleep(2)
                devices_response = sp_client.devices()
                devices = devices_response.get("devices", [])
                if devices:
                    if target_device_id:
                        for device in devices:
                            if device["id"] == target_device_id:
                                logger.info("Target device now available: %s", device["name"])
                                return device["id"]
                    device = devices[0]
                    logger.info("Device now available: %s", device["name"])
                    try:
                        sp_client.transfer_playback([device["id"]], force_play=False)
                        time.sleep(0.3)
                    except SpotifyException:
                        pass
                    return device["id"]
                else:
                    logger.warning("Still no devices after opening Spotify")

        logger.warning("Could not find or activate any Spotify device")
        return None

    except Exception as e:
        logger.error("Error ensuring active device: %s", e)
        return None


def _open_spotify_app() -> bool:
    """Use AppleScript to open Spotify (macOS only)."""
    if platform.system() != "Darwin":
        return False

    script = '''
tell application "Spotify"
    activate
    delay 1
    play
    delay 0.5
    pause
end tell
'''
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["osascript", script_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                logger.info("Spotify app opened and activated via AppleScript")
            return True
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    except subprocess.TimeoutExpired:
        logger.warning("AppleScript timed out – Spotify may be slow to respond")
        return True
    except FileNotFoundError:
        logger.error("osascript not found – AppleScript not available")
        return False
    except Exception as e:
        logger.error("Failed to open Spotify via AppleScript: %s", e)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_track_info(playback_data: Dict) -> Dict[str, Any]:
    if not playback_data or not playback_data.get("item"):
        return {}
    return _extract_track_info_from_item(playback_data["item"])


def _extract_track_info_from_item(track_item: Dict) -> Dict[str, Any]:
    return {
        "name": track_item["name"],
        "artists": [artist["name"] for artist in track_item["artists"]],
        "album": track_item["album"]["name"],
        "duration_ms": track_item["duration_ms"],
        "uri": track_item["uri"],
    }


def _get_current_timestamp() -> str:
    try:
        return datetime.now().isoformat()
    except Exception:
        return "unknown"
