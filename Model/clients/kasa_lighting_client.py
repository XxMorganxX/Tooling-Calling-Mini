"""
Dedicated Kasa lighting client.

Provides typed, debuggable control for both individual lights and scenes.
Safe to call from synchronous code (runs async operations in a dedicated thread).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List


class KasaLightingClient:
    """High-level controller for Kasa lights with scene support and strong debugging."""

    def __init__(
        self,
        light_mapping: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        debug: bool = True,
    ) -> None:
        self.light_config: Dict[str, Any] = light_mapping or {"lights": {}, "rooms": {}}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.debug = debug

        self._discover = None
        self._import_error: Optional[str] = None

        self._clients: Dict[str, Any] = {}

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kasa-client")

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

    # ----------------------------- Public API (sync) -----------------------------
    def turn_on(self, light_name: str) -> Dict[str, Any]:
        return self._run_async_blocking(self._turn_on_async(light_name))

    def turn_off(self, light_name: str) -> Dict[str, Any]:
        return self._run_async_blocking(self._turn_off_async(light_name))

    def toggle(self, light_name: str) -> Dict[str, Any]:
        return self._run_async_blocking(self._toggle_async(light_name))

    def set_brightness(self, light_name: str, brightness: int) -> Dict[str, Any]:
        return self._run_async_blocking(self._set_brightness_async(light_name, brightness))

    def set_color_temperature(self, light_name: str, kelvin: int) -> Dict[str, Any]:
        return self._run_async_blocking(self._set_color_temp_async(light_name, kelvin))

    def apply_scene(self, scene_name: str, room: Optional[str] = None, light_names: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._run_async_blocking(self._apply_scene_async(scene_name, room, light_names))

    def get_status(self, light_name: str) -> Dict[str, Any]:
        return self._run_async_blocking(self._get_status_async(light_name))

    # ----------------------------- Async Internals ------------------------------
    async def _turn_on_async(self, light_name: str) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            await client.turn_on()
            return self._success(light_name, action="on")
        except Exception as e:
            return self._failure(light_name, str(e))

    async def _turn_off_async(self, light_name: str) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            await client.turn_off()
            return self._success(light_name, action="off")
        except Exception as e:
            return self._failure(light_name, str(e))

    async def _toggle_async(self, light_name: str) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            if getattr(client, 'is_on', None):
                await client.turn_off()
                return self._success(light_name, action="off")
            else:
                await client.turn_on()
                return self._success(light_name, action="on")
        except Exception as e:
            return self._failure(light_name, str(e))

    async def _set_brightness_async(self, light_name: str, brightness: int) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            brightness = max(1, min(int(brightness), 100))
            await client.turn_on()
            if hasattr(client, 'set_brightness'):
                await client.set_brightness(brightness)
                return self._success(light_name, action="set_brightness", brightness=brightness)
            return self._failure(light_name, "Device does not support brightness control")
        except Exception as e:
            return self._failure(light_name, str(e))

    async def _set_color_temp_async(self, light_name: str, kelvin: int) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            if hasattr(client, 'set_color_temp'):
                await client.set_color_temp(int(kelvin))
                return self._success(light_name, action="set_color_temp", kelvin=int(kelvin))
            return self._failure(light_name, "Device does not support color temperature control")
        except Exception as e:
            return self._failure(light_name, str(e))

    async def _apply_scene_async(self, scene_name: str, room: Optional[str], light_names: Optional[List[str]]) -> Dict[str, Any]:
        targets = self._resolve_targets(room=room, light_names=light_names)
        if not targets:
            return {"success": False, "error": "No target lights found", "scene": scene_name}

        scene = self._get_scene_definition(scene_name)
        if not scene:
            return {"success": False, "error": f"Unknown scene '{scene_name}'", "scene": scene_name}

        results = []
        for name in targets:
            client = await self._get_or_create_client(name)
            if not client:
                results.append(self._failure(name, "Client not initialized"))
                continue
            try:
                await client.turn_on()
                if 'brightness' in scene and hasattr(client, 'set_brightness'):
                    await client.set_brightness(scene['brightness'])
                if 'kelvin' in scene and hasattr(client, 'set_color_temp'):
                    await client.set_color_temp(scene['kelvin'])
                results.append(self._success(name, action="scene", scene=scene_name))
            except Exception as e:
                results.append(self._failure(name, str(e)))

        overall_success = any(r.get('success') for r in results)
        return {
            "success": overall_success,
            "scene": scene_name,
            "results": results,
            "targets": targets,
        }

    async def _get_status_async(self, light_name: str) -> Dict[str, Any]:
        client = await self._get_or_create_client(light_name)
        if not client:
            return self._failure(light_name, "Client not initialized")
        try:
            props = {
                'is_on': getattr(client, 'is_on', None),
                'alias': getattr(client, 'alias', None),
                'model': getattr(client, 'model', None),
                'mac': getattr(client, 'mac', None),
                'host': getattr(client, 'host', None),
            }
            return {"success": True, "light": light_name, "status": props}
        except Exception as e:
            return self._failure(light_name, str(e))

    # ----------------------------- Helpers --------------------------------------
    def _resolve_targets(self, room: Optional[str], light_names: Optional[List[str]]) -> List[str]:
        if light_names:
            return [n for n in light_names if n in self.light_config.get('lights', {})]
        if room:
            return list(self.light_config.get('rooms', {}).get(room, []))
        return []

    def _get_scene_definition(self, scene_name: str) -> Optional[Dict[str, Any]]:
        scenes = {
            'movie': {"brightness": 20},
            'work': {"brightness": 90},
            'mood': {"brightness": 40},
            'reading': {"brightness": 70},
            'party': {"brightness": 80},
            'relax': {"brightness": 30},
        }
        return scenes.get(scene_name.lower())

    async def _get_or_create_client(self, light_name: str):
        if light_name in self._clients:
            return self._clients[light_name]
        light_info = self.light_config.get('lights', {}).get(light_name)
        if not light_info or not light_info.get('ip'):
            self._log_debug(f"Missing IP for light '{light_name}' in LIGHT_ROOM_MAPPING.lights")
            return None
        await self._ensure_kasa_import()
        if not self._discover:
            self._log_debug(f"kasa import failed: {self._import_error}")
            return None
        try:
            credentials = light_info.get('credentials', {})
            if credentials.get('username') and credentials.get('password'):
                device = await self._discover.discover_single(
                    light_info['ip'],
                    username=credentials['username'],
                    password=credentials['password'],
                )
            else:
                device = await self._discover.discover_single(light_info['ip'])
            self._clients[light_name] = device
            self._log_debug(f"Initialized client for {light_name} at {light_info['ip']}")
            return device
        except Exception as e:
            self._log_debug(f"Discover failed for {light_name} at {light_info.get('ip')}: {e}")
            return None

    async def _ensure_kasa_import(self) -> None:
        if self._discover is not None:
            return
        try:
            from kasa import Discover as _D  # type: ignore
            self._discover = _D
            self._import_error = None
        except Exception as e1:
            try:
                from Kasa import Discover as _D  # type: ignore
                self._discover = _D
                self._import_error = None
            except Exception as e2:
                self._discover = None
                self._import_error = f"Import errors: kasa -> {e1}; Kasa -> {e2}"

    def _run_async_blocking(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return self._executor.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _success(self, light: str, **kwargs) -> Dict[str, Any]:
        payload = {"success": True, "light": light}
        payload.update(kwargs)
        self._log_debug(f"SUCCESS {payload}")
        return payload

    def _failure(self, light: str, error: str) -> Dict[str, Any]:
        payload = {"success": False, "light": light, "error": error}
        self._log_debug(f"FAIL {payload}")
        return payload

    def _log_debug(self, msg: str) -> None:
        if self.debug:
            try:
                self.logger.debug(msg)
            except Exception:
                print(f"[KasaLightingClient] {msg}")
