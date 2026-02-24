"""
Kasa Lighting tool — standalone module.

Supports two interaction styles:
1) Direct control of individual lights (on/off)
2) Scene application across a room or selected lights
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tools.models import KasaLightingArgs
from clients.kasa_lighting_client import KasaLightingClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_light_config: Dict[str, Any] = json.loads(
    os.environ.get("KASA_LIGHT_CONFIG", '{"lights": {}, "rooms": {}}')
)

_client: Optional[KasaLightingClient] = None
_last_light_name: Optional[str] = None


def _get_client() -> KasaLightingClient:
    """Lazy-init the Kasa client on first use."""
    global _client
    if _client is None:
        _client = KasaLightingClient()
    return _client


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: KasaLightingArgs) -> dict:
    """Execute the kasa lighting tool with the provided parameters."""
    global _last_light_name
    try:
        params = args.model_dump()
        logger.debug("Executing Tool: KasaLighting -- %s", params)

        interaction: str = (args.interaction or "").strip().lower()
        if interaction not in ("direct", "scene"):
            return {
                "success": False,
                "error": (
                    f"Invalid interaction type '{interaction}'. Must be 'direct' (control a single light on/off) "
                    f"or 'scene' (apply a scene to a room or set of lights)"
                ),
            }

        client = _get_client()

        if interaction == "direct":
            action = (args.action or "").strip().lower()
            light_name: Optional[str] = args.light_name

            if not light_name and _last_light_name:
                light_name = _last_light_name
            if not action and not light_name:
                return {
                    "success": False,
                    "error": (
                        "Direct control requires both 'action' (on/off) and 'light_name'. "
                        f"Available lights: {list(_light_config.get('lights', {}).keys()) or 'none configured'}"
                    ),
                }
            if not action:
                return {
                    "success": False,
                    "error": f"Missing 'action' for direct control of light '{light_name}'. Must be 'on' or 'off'",
                }
            if not light_name:
                return {
                    "success": False,
                    "error": (
                        f"Missing 'light_name' for direct '{action}' action. "
                        f"Available lights: {list(_light_config.get('lights', {}).keys()) or 'none configured'}"
                    ),
                }

            if action == "on":
                result = client.turn_on(light_name)
            elif action == "off":
                result = client.turn_off(light_name)
            else:
                return {"success": False, "error": f"Unknown action '{action}'. Allowed: on, off"}

            if result and result.get("success"):
                _last_light_name = light_name
            return {
                "success": bool(result.get("success")),
                "result": result,
                "last_light": _last_light_name,
            }

        # --- scene interaction ---
        scene_name = args.scene_name
        room = args.room
        light_names = args.light_names

        targets: List[str] = []
        if light_names:
            targets = [n for n in light_names if n in _light_config.get("lights", {})]
        elif room:
            targets = list(_light_config.get("rooms", {}).get(room, []))
        if not targets:
            available_rooms = list(_light_config.get("rooms", {}).keys()) or "none configured"
            available_lights = list(_light_config.get("lights", {}).keys()) or "none configured"
            return {
                "success": False,
                "error": (
                    f"No target lights resolved for scene '{scene_name or 'unnamed'}'. "
                    f"Provide 'room' (available: {available_rooms}) or "
                    f"'light_names' list (available: {available_lights})"
                ),
            }

        per_light = []
        for name in targets:
            per_light.append(client.turn_on(name))
        overall = any(r.get("success") for r in per_light)
        return {
            "success": overall,
            "result": {"scene": scene_name, "targets": targets, "results": per_light},
        }

    except Exception as e:
        return {"success": False, "error": f"Lighting control failed: {e}"}
