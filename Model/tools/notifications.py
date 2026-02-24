"""
Get Notifications Tool – standalone module.

Ported from MCP BaseTool class to module-level functions.
Retrieves pending notifications from Supabase (primary) and local state (fallback).
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.models import GetNotificationsArgs

try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_project_root = Path(__file__).parent.parent.parent
_state_file = _project_root / "state_management" / "app_state.json"

_supabase_client: Optional[Any] = None
_supabase_available = False

# User config from env
_users_env = os.getenv("NOTIFICATION_USERS", "Morgan")
_configured_users = [u.strip() for u in _users_env.split(",") if u.strip()]
_default_user = os.getenv("DEFAULT_NOTIFICATION_USER", _configured_users[0] if _configured_users else "Morgan")


def _init_supabase() -> None:
    global _supabase_client, _supabase_available

    if not SUPABASE_AVAILABLE:
        logger.debug("Supabase package not installed, using local state only")
        return

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.debug("SUPABASE_URL or SUPABASE_KEY not set, using local state only")
        return

    try:
        _supabase_client = create_client(url, key)
        _supabase_available = True
        logger.debug("Supabase client initialized for notifications")
    except Exception as e:
        logger.warning("Failed to initialize Supabase client: %s", e)


_init_supabase()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(args: GetNotificationsArgs) -> dict:
    """Execute the notifications tool."""
    try:
        params = args.model_dump()

        user = params.get("user", _default_user)
        try:
            if isinstance(user, str):
                lowered = user.strip().lower()
                configured_lower = [u.lower() for u in _configured_users]
                if lowered in configured_lower:
                    idx = configured_lower.index(lowered)
                    user = _configured_users[idx]
                elif lowered in {"me", "my", "default"}:
                    user = _default_user
                else:
                    user = user.title()
        except Exception:
            user = _default_user

        type_filter = params.get("type_filter", "all")
        limit = params.get("limit", 10)
        include_metadata = params.get("include_metadata", True)
        priority_filter = params.get("priority_filter", "all")

        # Load notifications
        notifications_data = _load_notifications(type_filter=type_filter, user_filter=user)
        user_notifications = notifications_data.get(user, [])

        # Apply priority filter
        if priority_filter != "all":
            user_notifications = [
                n for n in user_notifications
                if n.get("priority", "normal").lower() == priority_filter.lower()
            ]

        # Sort by priority then recency
        def _sort_key(x: Dict[str, Any]):
            pv = {"high": 0, "normal": 1, "low": 2}.get(str(x.get("priority", "normal")).lower(), 1)
            ts = _coerce_timestamp(x.get("timestamp"))
            return (pv, -ts)

        user_notifications.sort(key=_sort_key)

        total_available = len(user_notifications)
        limited = user_notifications[:limit]

        processed: List[Dict[str, Any]] = []
        unread_ids: List[str] = []

        for notif in limited:
            p = notif.copy()
            original_read = notif.get("read_status", "unread")
            previously_seen = original_read == "read"
            p["previously_seen"] = previously_seen

            if not previously_seen and notif.get("id"):
                unread_ids.append(notif["id"])

            if not include_metadata:
                for field in ("timestamp", "priority", "source", "read_status", "id"):
                    p.pop(field, None)
            else:
                if "timestamp" in p:
                    try:
                        ts = p["timestamp"]
                        if isinstance(ts, (int, float)):
                            p["human_time"] = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

            processed.append(p)

        # Mark as read
        marked_count = 0
        if unread_ids:
            marked_count = _mark_notifications_as_read(user, unread_ids)
            logger.debug("Auto-marked %d notifications as read after retrieval", marked_count)

        response: Dict[str, Any] = {
            "success": True,
            "user": user,
            "type_filter": type_filter,
            "priority_filter": priority_filter,
            "notifications": processed,
            "count": len(processed),
            "total_available": total_available,
            "has_more": total_available > limit,
            "newly_marked_as_read": marked_count,
            "metadata_included": include_metadata,
            "filters_applied": {"type": type_filter, "priority": priority_filter, "limit": limit},
        }

        if processed:
            type_counts: Dict[str, int] = {}
            priority_counts: Dict[str, int] = {}
            for notif in user_notifications:
                nt = notif.get("type", "other")
                np_ = notif.get("priority", "normal")
                type_counts[nt] = type_counts.get(nt, 0) + 1
                priority_counts[np_] = priority_counts.get(np_, 0) + 1

            response["summary"] = {
                "total_by_type": type_counts,
                "total_by_priority": priority_counts,
                "most_recent": processed[0].get("human_time") if processed else None,
            }

        return response

    except Exception as e:
        logger.error("Error executing notifications tool: %s", e)
        return {
            "success": False,
            "error": str(e),
            "user": (args.user if args else None) or "unknown",
            "count": 0,
            "notifications": [],
            "total_available": 0,
        }


# ---------------------------------------------------------------------------
# Loading notifications
# ---------------------------------------------------------------------------


def _load_notifications(
    type_filter: str = "all", user_filter: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """Load from Supabase (primary) and local state (fallback)."""
    notifications: Dict[str, List[Dict]] = {}
    supabase_loaded_types: set = set()

    if _supabase_available and _supabase_client:
        try:
            sb = _load_from_supabase(type_filter=type_filter, user_filter=user_filter)
            if sb:
                notifications = sb
                for user_notifs in sb.values():
                    for n in user_notifs:
                        supabase_loaded_types.add(n.get("type"))
                logger.debug(
                    "Loaded notifications from Supabase: %d total (types=%s)",
                    sum(len(v) for v in notifications.values()),
                    supabase_loaded_types,
                )
        except Exception as e:
            logger.warning("Failed to load from Supabase, falling back to local: %s", e)

    local = _load_from_local_state(type_filter=type_filter, user_filter=user_filter)

    for user, local_list in local.items():
        if user not in notifications:
            notifications[user] = []

        existing_ids: set = set()
        existing_email_ids: set = set()
        for n in notifications.get(user, []):
            nid = n.get("id", "")
            existing_ids.add(nid)
            existing_ids.add(_strip_id_prefix(nid))
            for eid in n.get("email_ids") or []:
                existing_email_ids.add(eid)

        for notif in local_list:
            notif_type = notif.get("type", "other")
            if notif_type in supabase_loaded_types:
                continue

            notif_id = notif.get("id", "")
            base_id = _strip_id_prefix(notif_id)
            if notif_id in existing_ids or base_id in existing_ids:
                continue

            notif_email_ids = notif.get("email_ids") or []
            if notif_email_ids and any(eid in existing_email_ids for eid in notif_email_ids):
                continue

            notifications[user].append(notif)
            existing_ids.add(notif_id)
            existing_ids.add(base_id)
            for eid in notif_email_ids:
                existing_email_ids.add(eid)

    return notifications


def _load_from_supabase(
    type_filter: str = "all", user_filter: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """Load from Supabase notification_sources table (most recent batch per type)."""
    if not _supabase_client:
        return {}

    try:
        notifications: Dict[str, List[Dict]] = {}

        types_to_fetch: List[str] = []
        if type_filter == "all":
            types_to_fetch = ["email", "news"]
        elif type_filter in ["email", "news"]:
            types_to_fetch = [type_filter]

        for source_type in types_to_fetch:
            batch_query = (
                _supabase_client.table("notification_sources")
                .select("batch_id")
                .eq("source_type", source_type)
            )
            if user_filter:
                batch_query = batch_query.eq("user_id", user_filter)

            batch_response = batch_query.order("created_at", desc=True).limit(1).execute()

            if not batch_response.data:
                continue

            most_recent_batch_id = batch_response.data[0].get("batch_id")
            if not most_recent_batch_id:
                continue

            query = (
                _supabase_client.table("notification_sources")
                .select("*")
                .eq("source_type", source_type)
                .eq("batch_id", most_recent_batch_id)
            )
            if user_filter:
                query = query.eq("user_id", user_filter)

            response = query.order("created_at", desc=True).execute()

            if not response.data:
                continue

            for row in response.data:
                user_id = row.get("user_id", "Unknown")
                created_at = row.get("created_at")
                source_generated_at = row.get("source_generated_at")
                timestamp = _parse_supabase_timestamp(source_generated_at or created_at)

                notif_type = source_type if source_type in ["email", "news"] else "other"
                metadata = row.get("metadata") or {}

                normalized = {
                    "id": row.get("id"),
                    "content": row.get("content", ""),
                    "type": notif_type,
                    "priority": row.get("priority", "normal"),
                    "timestamp": timestamp,
                    "source": f"supabase_{source_type}",
                    "read_status": row.get("read_status", "unread"),
                    "title": row.get("title", ""),
                    "topic": metadata.get("topic"),
                    "email_ids": metadata.get("email_ids"),
                    "source_articles_count": metadata.get("source_articles_count"),
                    "batch_id": row.get("batch_id"),
                }

                if user_id not in notifications:
                    notifications[user_id] = []
                notifications[user_id].append(normalized)

        return notifications

    except Exception as e:
        logger.error("Error loading from Supabase: %s", e)
        return {}


def _load_from_local_state(
    type_filter: str = "all", user_filter: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """Load from local app_state.json (fallback)."""
    try:
        if not _state_file.exists():
            logger.debug("State file not found: %s", _state_file)
            return {}

        with open(_state_file, "r") as f:
            state = json.load(f)

        # New format
        if isinstance(state.get("notifications"), dict):
            result = state["notifications"]
            if user_filter:
                result = {k: v for k, v in result.items() if k == user_filter}
            if type_filter and type_filter != "all":
                result = {
                    u: [n for n in ns if n.get("type") == type_filter]
                    for u, ns in result.items()
                }
            for uk in result:
                result[uk] = _filter_most_recent_per_type(result[uk])
            return result

        # Legacy format
        autonomous_state = state.get("autonomous_state", {})
        notification_queue = autonomous_state.get("notification_queue", {})
        if not isinstance(notification_queue, dict):
            return {}

        def should_include_type(nt: str) -> bool:
            return type_filter == "all" or nt == type_filter

        normalized: Dict[str, List[Dict]] = {}
        for user, user_data in notification_queue.items():
            if user_filter and user != user_filter:
                continue

            nlist: List[Dict[str, Any]] = []

            if type_filter in ["all", "other"]:
                for raw in (user_data or {}).get("notifications", []):
                    content = raw.get("notification_content") or raw.get("content") or ""
                    raw_type = raw.get("type")
                    notif_type = _normalize_type(raw_type, content)
                    if not should_include_type(notif_type):
                        continue
                    priority = _normalize_priority(raw.get("priority"))
                    timestamp = _coerce_timestamp(raw.get("timestamp"))
                    source = raw.get("source", "unknown")
                    notif_id = raw.get("id") or _generate_notification_id(content, notif_type, timestamp)
                    nlist.append({
                        "id": notif_id,
                        "content": content,
                        "type": notif_type,
                        "priority": priority,
                        "timestamp": timestamp,
                        "source": source,
                        "read_status": raw.get("read_status", "unread"),
                    })

            # Most recent email only
            if type_filter in ["all", "email"]:
                emails_list = (user_data or {}).get("emails", [])
                most_recent_email = None
                most_recent_ts = 0
                for email_item in emails_list:
                    try:
                        ts = _coerce_timestamp(email_item.get("timestamp"))
                        if ts >= most_recent_ts:
                            most_recent_ts = ts
                            most_recent_email = email_item
                    except Exception:
                        continue

                if most_recent_email:
                    try:
                        email_content = most_recent_email.get("content", "")
                        email_title = most_recent_email.get("title") or "Email Update"
                        priority = _normalize_priority(most_recent_email.get("priority"))
                        timestamp = _coerce_timestamp(most_recent_email.get("timestamp"))
                        source = most_recent_email.get("source", "email_summarizer")
                        notif_id = most_recent_email.get("id") or _generate_notification_id(
                            f"{email_title}|{email_content}", "email", timestamp
                        )
                        nlist.append({
                            "id": notif_id,
                            "content": email_content,
                            "type": "email",
                            "priority": priority,
                            "timestamp": timestamp,
                            "source": source,
                            "read_status": most_recent_email.get("read_status", "unread"),
                            "title": email_title,
                            "topic": most_recent_email.get("topic"),
                            "email_ids": most_recent_email.get("email_ids"),
                        })
                    except Exception:
                        pass

            # News
            if type_filter in ["all", "news"]:
                news_data = (user_data or {}).get("news")
                if news_data and isinstance(news_data, dict):
                    try:
                        news_summary = news_data.get("summary", "")
                        news_title = "Tech News Summary"
                        generated_at = news_data.get("generated_at")
                        if generated_at:
                            try:
                                timestamp = int(datetime.fromisoformat(generated_at).timestamp())
                            except Exception:
                                timestamp = _coerce_timestamp(None)
                        else:
                            timestamp = _coerce_timestamp(None)
                        notif_id = _generate_notification_id(f"{news_title}|{news_summary}", "news", timestamp)
                        nlist.append({
                            "id": notif_id,
                            "content": news_summary,
                            "type": "news",
                            "priority": "normal",
                            "timestamp": timestamp,
                            "source": "news_summarizer",
                            "read_status": "unread",
                            "title": news_title,
                            "generated_at": generated_at,
                            "source_articles_count": news_data.get("source_articles_count"),
                            "source_file": news_data.get("source_file"),
                        })
                    except Exception:
                        pass

            normalized[user] = nlist

        return normalized

    except Exception as e:
        logger.error("Error loading notifications from local state: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Mark as read
# ---------------------------------------------------------------------------


def _mark_notifications_as_read(user: str, notification_ids: List[str]) -> int:
    if not notification_ids:
        return 0

    marked_count = 0

    if _supabase_available and _supabase_client:
        try:
            sb_marked = _mark_as_read_in_supabase(notification_ids)
            marked_count += sb_marked
            logger.debug("Marked %d notifications as read in Supabase", sb_marked)
        except Exception as e:
            logger.warning("Failed to mark as read in Supabase: %s", e)

    try:
        local_marked = _mark_as_read_in_local_state(user, notification_ids)
        if not _supabase_available:
            marked_count += local_marked
        logger.debug("Marked %d notifications as read in local state", local_marked)
    except Exception as e:
        logger.warning("Failed to mark as read in local state: %s", e)

    logger.info("Marked %d notifications as read for %s", marked_count, user)
    return marked_count


def _mark_as_read_in_supabase(notification_ids: List[str]) -> int:
    if not _supabase_client or not notification_ids:
        return 0

    try:
        marked = 0
        for notif_id in notification_ids:
            try:
                _supabase_client.table("notification_sources").update(
                    {"read_status": "read"}
                ).eq("id", notif_id).execute()
                marked += 1
            except Exception as e:
                logger.debug("Failed to mark %s as read: %s", notif_id, e)
        return marked
    except Exception as e:
        logger.error("Error marking as read in Supabase: %s", e)
        return 0


def _mark_as_read_in_local_state(user: str, notification_ids: List[str]) -> int:
    try:
        if not _state_file.exists():
            return 0

        with open(_state_file, "r") as f:
            state = json.load(f)

        marked_count = 0

        if isinstance(state.get("notifications"), dict):
            user_notifications = state.get("notifications", {}).get(user, [])
            initial_count = len(user_notifications)
            filtered = [n for n in user_notifications if n.get("id") not in notification_ids]
            if "notifications" not in state:
                state["notifications"] = {}
            state["notifications"][user] = filtered
            marked_count = initial_count - len(filtered)
        else:
            autonomous_state = state.get("autonomous_state", {})
            queue = autonomous_state.get("notification_queue", {})
            user_block = queue.get(user, {})
            raw_list = user_block.get("notifications", [])

            remaining = []
            for raw in raw_list:
                content = raw.get("notification_content") or raw.get("content") or ""
                notif_type = raw.get("type") or _determine_type_from_content(content)
                timestamp = raw.get("timestamp")
                raw_id = raw.get("id") or _generate_notification_id(content, notif_type, timestamp)
                if raw_id not in notification_ids:
                    remaining.append(raw)

            initial_count = len(raw_list)
            user_block["notifications"] = remaining

            emails_list = user_block.get("emails", [])
            remaining_emails = []
            for email_item in emails_list:
                email_id = email_item.get("id")
                if email_id and email_id not in notification_ids:
                    remaining_emails.append(email_item)
            user_block["emails"] = remaining_emails

            news_data = user_block.get("news")
            if news_data and isinstance(news_data, dict):
                news_summary = news_data.get("summary", "")
                news_title = "Tech News Summary"
                generated_at = news_data.get("generated_at")
                if generated_at:
                    try:
                        timestamp = int(datetime.fromisoformat(generated_at).timestamp())
                    except Exception:
                        timestamp = 0
                else:
                    timestamp = 0
                news_id = _generate_notification_id(f"{news_title}|{news_summary}", "news", timestamp)
                if news_id in notification_ids:
                    user_block.pop("news", None)
                    marked_count += 1

            queue[user] = user_block
            autonomous_state["notification_queue"] = queue
            state["autonomous_state"] = autonomous_state
            marked_count += initial_count - len(remaining)

        with open(_state_file, "w") as f:
            json.dump(state, f, indent=2)

        return marked_count

    except Exception as e:
        logger.error("Error marking notifications as read in local state: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_most_recent_per_type(notifications: List[Dict]) -> List[Dict]:
    if not notifications:
        return []

    most_recent_by_type: Dict[str, Any] = {}

    for notif in notifications:
        notif_type = notif.get("type", "other")
        timestamp = _coerce_timestamp(notif.get("timestamp"))

        if notif_type in ["email", "news"]:
            existing = most_recent_by_type.get(notif_type)
            if existing is None or timestamp > _coerce_timestamp(existing.get("timestamp")):
                most_recent_by_type[notif_type] = notif
        else:
            if notif_type not in most_recent_by_type:
                most_recent_by_type[notif_type] = []
            if isinstance(most_recent_by_type[notif_type], list):
                most_recent_by_type[notif_type].append(notif)
            else:
                most_recent_by_type[notif_type] = [most_recent_by_type[notif_type], notif]

    result = []
    for value in most_recent_by_type.values():
        if isinstance(value, list):
            result.extend(value)
        else:
            result.append(value)

    return result


def _generate_notification_id(content: str, notif_type: str, timestamp: Any) -> str:
    try:
        base = f"{notif_type}|{timestamp or ''}|{content}".encode("utf-8", errors="ignore")
        return hashlib.sha1(base).hexdigest()[:16]
    except Exception:
        return f"{notif_type}-{str(timestamp)}-{abs(hash(content)) % (10**8)}"


def _determine_type_from_content(content: str) -> str:
    txt = (content or "").lower()
    if "email" in txt:
        return "email"
    if "news" in txt:
        return "news"
    return "other"


def _normalize_type(raw_type: Optional[str], content: str) -> str:
    if not raw_type:
        return _determine_type_from_content(content)
    t = str(raw_type).lower()
    return t if t in {"email", "news", "other"} else "other"


def _normalize_priority(raw_priority: Optional[str]) -> str:
    if not raw_priority:
        return "normal"
    p = str(raw_priority).lower()
    return p if p in {"high", "normal", "low"} else "normal"


def _coerce_timestamp(raw_ts: Any) -> int:
    try:
        if raw_ts is None:
            return 0
        if isinstance(raw_ts, (int, float)):
            return int(raw_ts)
        try:
            return int(datetime.fromisoformat(str(raw_ts)).timestamp())
        except Exception:
            return 0
    except Exception:
        return 0


def _strip_id_prefix(notif_id: str) -> str:
    if not notif_id:
        return ""
    for prefix in ("email_", "news_"):
        if notif_id.startswith(prefix):
            return notif_id[len(prefix):]
    return notif_id


def _parse_supabase_timestamp(ts_str: Optional[str]) -> int:
    if not ts_str:
        return 0
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_str)
        return int(dt.timestamp())
    except Exception:
        return 0
