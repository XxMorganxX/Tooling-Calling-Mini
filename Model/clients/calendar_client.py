from __future__ import print_function
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timezone, timedelta

try:
    from inference import config
except Exception:
    class config:  # type: ignore
        DEBUG_MODE = False
        CALENDAR_USERS = {}
        CALENDAR_SCOPES = [
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
        ]
        DEFAULT_TIME_ZONE = None
        GOOGLE_SERVICE_ACCOUNT_FILE = None


class CalendarComponent:
    """
    Google Calendar component with proper credentials management.
    
    Each instance is bound to a specific user that must be explicitly provided.
    To work with a different user, create a new CalendarComponent instance.
    """
    
    def __init__(self, user: str):
        """
        Initialize calendar component for a specific user.
        
        Args:
            user: User name ('morgan_personal' or 'morgan_school' etc).
        """
        self._user = user
        self.creds: Optional[Credentials] = None
        self.service = None
        self.error_message: Optional[str] = None
        
        if not self._validate_user():
            users = list(getattr(config, 'CALENDAR_USERS', {}).keys())
            raise ValueError(f"Invalid user '{self.user}'. Must be one of: {users}")
        
        self._initialize_credentials()
    
    @property
    def user(self) -> str:
        return self._user

    def get_calendar_mappings(self):
        """Get calendar ID->name and name->ID mappings from Google Calendar API."""
        try:
            self._maybe_refresh_credentials()
            if not self.service:
                return {}, {}
            
            calendar_list = self.service.calendarList().list().execute()
            
            id_to_name = {}
            name_to_id = {}
            
            for cal in calendar_list.get('items', []):
                cal_id = cal['id']
                cal_name = cal.get('summary', cal_id)
                
                id_to_name[cal_id] = cal_name
                name_to_id[cal_name] = cal_id
            
            return id_to_name, name_to_id
            
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"Error getting calendar mappings: {e}")
            return {}, {}

    def calendar_id_to_name(self, cal_id) -> str:
        id_to_name, _ = self.get_calendar_mappings()
        return id_to_name.get(cal_id, cal_id)

    def calendar_name_to_id(self, name) -> str:
        _, name_to_id = self.get_calendar_mappings()
        if name in name_to_id:
            return name_to_id[name]
        name_lower = name.lower()
        for cal_name, cal_id in name_to_id.items():
            if cal_name.lower() == name_lower:
                return cal_id
        return name

    def get_morning_datetime(self) -> datetime:
        return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'

    def _validate_user(self) -> bool:
        return self._user in getattr(config, 'CALENDAR_USERS', {})
    
    def _get_user_config(self) -> Dict[str, str]:
        users_cfg = getattr(config, 'CALENDAR_USERS', {})
        return users_cfg[self._user]
    
    def _is_headless(self) -> bool:
        return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    
    def _get_env_refresh_token(self) -> Optional[str]:
        """Get the refresh token for this user from environment variables."""
        user_upper = self._user.upper().replace("_PERSONAL", "").replace("_SCHOOL", "")
        user_token = os.getenv(f"GCAL_REFRESH_TOKEN_{user_upper}")
        if user_token:
            return user_token
        
        return (
            os.getenv("GCAL_REFRESH_TOKEN")
            or os.getenv("GMAIL_REFRESH_TOKEN")
            or os.getenv("GOOGLE_REFRESH_TOKEN")
        )
    
    def _try_service_account(self) -> bool:
        """Try to authenticate using a Google Service Account.
        
        Checks in order:
        1. GOOGLE_SERVICE_ACCOUNT_JSON env var (raw JSON or base64)
        2. GOOGLE_SERVICE_ACCOUNT_FILE env var (file path)
        3. config.GOOGLE_SERVICE_ACCOUNT_FILE (default: creds/homeassist-google-service.json)
        """
        import json as json_module
        import base64
        
        scopes = getattr(
            config,
            "CALENDAR_SCOPES",
            [
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/calendar.events",
            ],
        )
        
        # Method 1: Service account JSON from environment variable
        sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        if sa_json:
            try:
                if sa_json.startswith("{"):
                    sa_data = json_module.loads(sa_json)
                else:
                    try:
                        decoded = base64.b64decode(sa_json).decode("utf-8")
                        sa_data = json_module.loads(decoded)
                    except Exception:
                        sa_data = json_module.loads(sa_json)
                
                self.creds = service_account.Credentials.from_service_account_info(
                    sa_data, scopes=scopes
                )
                
                if getattr(config, "DEBUG_MODE", False):
                    print(f"Calendar credentials loaded via Service Account for {self.user}")
                return True
                
            except Exception as e:
                print(f"Failed to load service account from env: {e}")
        
        # Method 2: Service account from file path (env var)
        sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
        if sa_file and os.path.exists(sa_file):
            try:
                self.creds = service_account.Credentials.from_service_account_file(
                    sa_file, scopes=scopes
                )
                
                if getattr(config, "DEBUG_MODE", False):
                    print(f"Calendar credentials loaded via Service Account file for {self.user}")
                return True
                
            except Exception as e:
                print(f"Failed to load service account from file: {e}")
        
        # Method 3: Service account from config file path
        config_sa_file = getattr(config, "GOOGLE_SERVICE_ACCOUNT_FILE", None)
        if config_sa_file and config_sa_file != sa_file:
            if not os.path.isabs(config_sa_file):
                project_root = Path(__file__).parent.parent
                config_sa_file = str(project_root / config_sa_file)
            
            if os.path.exists(config_sa_file):
                try:
                    self.creds = service_account.Credentials.from_service_account_file(
                        config_sa_file, scopes=scopes
                    )
                    
                    print(f"Calendar credentials loaded via Service Account for {self.user}")
                    return True
                    
                except Exception as e:
                    print(f"Failed to load service account from config: {e}")
            else:
                if getattr(config, "DEBUG_MODE", False):
                    print(f"Service account file not found: {config_sa_file}")
        
        return False
    
    def _try_env_credentials(self) -> bool:
        """Try to create credentials from environment variables."""
        import json as json_module
        import base64
        
        def decode_json_secret(value: str) -> dict:
            value = value.strip()
            if not value:
                return {}
            if value.startswith("{"):
                return json_module.loads(value)
            try:
                decoded = base64.b64decode(value).decode("utf-8")
                return json_module.loads(decoded)
            except Exception:
                return json_module.loads(value)
        
        scopes = getattr(
            config,
            "CALENDAR_SCOPES",
            [
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/calendar.events",
            ],
        )
        
        # Method 1: Full JSON blobs
        token_json = os.getenv("GOOGLE_CALENDAR_TOKEN_JSON", "").strip() or os.getenv("GOOGLE_TOKEN_JSON", "").strip()
        creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
        
        if token_json and creds_json:
            try:
                token_data = decode_json_secret(token_json)
                creds_data = decode_json_secret(creds_json)
                
                if "installed" in creds_data:
                    client_info = creds_data["installed"]
                elif "web" in creds_data:
                    client_info = creds_data["web"]
                else:
                    client_info = creds_data
                
                client_id = client_info.get("client_id")
                client_secret = client_info.get("client_secret")
                refresh_token = token_data.get("refresh_token")
                
                if client_id and client_secret and refresh_token:
                    self.creds = Credentials(
                        token_data.get("token"),
                        refresh_token=refresh_token,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=client_id,
                        client_secret=client_secret,
                        scopes=scopes,
                    )
                    
                    if not self.creds.valid:
                        self.creds.refresh(Request())
                    
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"Calendar credentials loaded from GOOGLE_*_JSON for {self.user}")
                    return True
                    
            except Exception as e:
                print(f"Failed to parse GOOGLE_*_JSON: {e}")
        
        # Method 2: Individual secrets
        client_id = (
            os.getenv("GCAL_CLIENT_ID") or 
            os.getenv("GMAIL_CLIENT_ID") or 
            os.getenv("GOOGLE_CLIENT_ID")
        )
        client_secret = (
            os.getenv("GCAL_CLIENT_SECRET") or 
            os.getenv("GMAIL_CLIENT_SECRET") or 
            os.getenv("GOOGLE_CLIENT_SECRET")
        )
        refresh_token = self._get_env_refresh_token()
        
        if not (client_id and client_secret and refresh_token):
            return False
        
        try:
            self.creds = Credentials(
                None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
            )
            
            self.creds.refresh(Request())
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"Calendar credentials loaded from environment for {self.user}")
            return True
            
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"Env-based calendar OAuth failed for {self.user}: {e}")
            return False
    
    def _initialize_credentials(self) -> bool:
        """Initialize Google credentials for the user.
        
        Order of precedence:
        1. Service Account (best for CI - no user interaction, never expires)
        2. OAuth environment variables (for CI/GitHub Actions)
        3. Token file (for local development)
        4. OAuth flow (interactive, local only)
        """
        try:
            scopes = getattr(
                config,
                "CALENDAR_SCOPES",
                [
                    "https://www.googleapis.com/auth/calendar.readonly",
                    "https://www.googleapis.com/auth/calendar.events",
                ],
            )
            
            if self._try_service_account():
                self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
                return True
            
            if self._try_env_credentials():
                self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
                return True
            
            # Fall back to file-based credentials
            user_config = self._get_user_config()
            token_path = user_config.get("token")
            client_secret_path = user_config.get("client_secret")

            if not token_path or not client_secret_path:
                raise RuntimeError(
                    f"No credentials available for '{self._user}'. "
                    "Set GOOGLE_SERVICE_ACCOUNT_FILE / GOOGLE_SERVICE_ACCOUNT_JSON, "
                    "or add 'token' and 'client_secret' paths to the calendar user config."
                )
            
            os.makedirs(str(Path(token_path).parent), exist_ok=True)
            
            if os.path.exists(token_path):
                try:
                    self.creds = Credentials.from_authorized_user_file(token_path, scopes)
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"Loaded existing calendar credentials for {self.user}")

                    if self.creds and not getattr(self.creds, "refresh_token", None):
                        if self._is_headless():
                            raise RuntimeError("Headless mode: missing refresh token. Run reauth locally.")
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"Upgrading {self.user} credentials to offline access...")
                        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
                        self.creds = flow.run_local_server(
                            port=0,
                            access_type='offline',
                            prompt='consent'
                        )
                        with open(token_path, 'w') as token_file:
                            token_file.write(self.creds.to_json())
                except Exception as e:
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"Failed to load existing token: {e}")
                    self.creds = None
            
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    try:
                        self.creds.refresh(Request())
                        with open(token_path, 'w') as token_file:
                            token_file.write(self.creds.to_json())
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"Refreshed and saved calendar credentials for {self.user}")
                    except Exception as e:
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"Failed to refresh credentials: {e}")
                        self.creds = None
                
                if not self.creds or not self.creds.valid:
                    if self._is_headless():
                        raise RuntimeError(
                            f"Headless mode: No valid credentials for {self.user}. "
                            f"Set GCAL_REFRESH_TOKEN_{self._user.upper().replace('_PERSONAL', '').replace('_SCHOOL', '')} "
                            "or run OAuth flow locally first."
                        )
                    
                    if not os.path.exists(client_secret_path):
                        raise FileNotFoundError(f"Client secret file not found: {client_secret_path}")
                    
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"Starting OAuth flow for {self.user}...")
                    
                    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
                    self.creds = flow.run_local_server(
                        port=0,
                        access_type='offline',
                        prompt='consent'
                    )
                
                with open(token_path, 'w') as token_file:
                    token_file.write(self.creds.to_json())

            self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"Calendar service initialized for {self.user}")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            print(f"Failed to initialize calendar credentials for {self.user}: {e}")
            return False

    def _maybe_refresh_credentials(self) -> None:
        """Refresh credentials if expired and persist the updated token file."""
        try:
            if not self.creds:
                return
            if self.creds.expired and getattr(self.creds, "refresh_token", None):
                if getattr(config, "DEBUG_MODE", False):
                    print(f"Refreshing expired credentials for {self.user}...")
                self.creds.refresh(Request())
                user_cfg = self._get_user_config()
                token_path = user_cfg.get("token")
                if token_path:
                    with open(token_path, 'w') as token_file:
                        token_file.write(self.creds.to_json())
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"Saved refreshed credentials for {self.user} to {token_path}")
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"Failed to refresh credentials for {self.user}: {e}")
    
    def get_events(self, num_events: int = 10, time_min: str = None, time_max: str = None, calendar_id: str = 'primary') -> List[Dict]:
        """Get calendar events."""
        try:
            self.error_message = None
            self._maybe_refresh_credentials()
            if not self.service:
                self._initialize_credentials()
                if not self.service:
                    print(f"Calendar service not initialized for {self.user}")
                    return []
            
            if calendar_id != 'primary':
                calendars_to_query = [{'id': calendar_id}]
            else:
                calendar_list = self.service.calendarList().list().execute()
                calendars_to_query = calendar_list.get('items', [])
            
            all_events = []
            
            for cal in calendars_to_query:
                cal_id = cal['id']
                
                now = datetime.now(timezone.utc).isoformat() if time_min is None else time_min
                
                events_result = self.service.events().list(
                    calendarId=cal_id,
                    timeMin=now,
                    timeMax=time_max,
                    maxResults=num_events,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                events = events_result.get('items', [])
                for event in events:
                    event['calendar_id'] = cal_id
                
                all_events.extend(events)
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"Retrieved {len(all_events)} events from {len(calendars_to_query)} calendars for {self.user}")
            
            def get_event_start_time(event):
                start = event.get('start', {})
                dt_str = start.get('dateTime') or start.get('date')
                if not dt_str:
                    return datetime.max.replace(tzinfo=timezone.utc)
                try:
                    if 'T' in dt_str:
                        dt_str = dt_str.replace('Z', '+00:00')
                        return datetime.fromisoformat(dt_str)
                    else:
                        return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    return datetime.max.replace(tzinfo=timezone.utc)
            
            all_events.sort(key=get_event_start_time)
            limited_events = all_events[:num_events]
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"Returning {len(limited_events)} events (limited from {len(all_events)} total)")
            
            self.error_message = None
            return limited_events
            
        except HttpError as e:
            error_msg = f"Google Calendar API error: {e}"
            self.error_message = error_msg
            print(f"{error_msg}")
            return []
        except Exception as e:
            error_msg = f"Failed to get calendar events: {e}"
            self.error_message = error_msg
            print(f"{error_msg}")
            return []
    
    def get_formatted_events(self, num_events: int = 10, time_min: str = None) -> List[Dict]:
        raw_events = self.get_events(num_events, time_min)
        return [self.format_event(event) for event in raw_events]
    
    def format_event(self, event: Dict) -> Dict[str, str]:
        try:
            start_raw = event['start'].get('dateTime', event['start'].get('date'))
            start_parts = start_raw.split("T")
            start_date = start_parts[0]
            start_time = start_parts[1].split("-")[0] if len(start_parts) > 1 else "All Day"
            
            end_raw = event['end'].get('dateTime', event['end'].get('date'))
            end_parts = end_raw.split("T")
            end_date = end_parts[0]
            end_time = end_parts[1].split("-")[0] if len(end_parts) > 1 else "All Day"
            
            return {
                'id': event.get('id', ''),
                'calendar_name': self.calendar_id_to_name(event.get('calendar_id', '')),
                'summary': event.get('summary', 'No Title'),
                'description': event.get('description', ''),
                'start_date': start_date,
                'start_time': start_time,
                'end_date': end_date,
                'end_time': end_time,
                'location': event.get('location', ''),
                'status': event.get('status', ''),
                'all_day': 'dateTime' not in event['start']
            }
            
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"Error formatting event: {e}")
            return {
                'id': event.get('id', ''),
                'calendar_name': 'Unknown',
                'summary': event.get('summary', 'Error formatting event'),
                'description': '',
                'start_date': 'Unknown',
                'start_time': 'Unknown',
                'end_date': 'Unknown', 
                'end_time': 'Unknown',
                'location': '',
                'status': '',
                'all_day': False
            }
    
    def display_events_details(self, events_list: List[Dict]) -> None:
        if not events_list:
            print("No events to display")
            return
        
        print(f"Displaying {len(events_list)} events for {self.user}:")
        print("-" * 80)
        
        for event in events_list:
            if 'start_date' in event:
                formatted = event
            else:
                formatted = self.format_event(event)
            
            print(f"  {formatted['summary']}")
            print(f"   Calendar: {formatted['calendar_name']}")
            print(f"   {formatted['start_date']} {formatted['start_time']} - {formatted['end_time']}")
            if formatted['location']:
                print(f"   Location: {formatted['location']}")
            if formatted['description']:
                print(f"   {formatted['description'][:100]}{'...' if len(formatted['description']) > 100 else ''}")
            print()
    
    def get_todays_events(self) -> List[Dict]:
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            time_min = today_start.isoformat() + 'Z'
            
            all_events = self.get_events(num_events=50, time_min=time_min)
            
            today_str = datetime.now().strftime('%Y-%m-%d')
            todays_events = []
            
            for event in all_events:
                formatted = self.format_event(event)
                if formatted['start_date'] == today_str:
                    todays_events.append(formatted)
                elif formatted['start_date'] > today_str:
                    break
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"Found {len(todays_events)} events for today")
            
            return todays_events
            
        except Exception as e:
            error_msg = f"Failed to get today's events: {e}"
            self.error_message = error_msg
            print(f"{error_msg}")
            return []
    
    def get_next_event(self) -> Optional[Dict]:
        try:
            events = self.get_events(num_events=1)
            if events:
                return self.format_event(events[0])
            else:
                print(f"No upcoming events found for {self.user}")
                return None
            
        except Exception as e:
            error_msg = f"Failed to get next event: {e}"
            self.error_message = error_msg
            print(f"{error_msg}")
            return None
    
    def get_last_event(self) -> Optional[Dict]:
        try:
            current_time = datetime.now(timezone.utc)
            
            past_start_time = (current_time - timedelta(days=7)).isoformat().replace('+00:00', '') + 'Z'
            current_time_str = current_time.isoformat().replace('+00:00', '') + 'Z'
            
            events = self.get_events(
                num_events=100, 
                time_min=past_start_time, 
                time_max=current_time_str
            )
            
            if events:
                most_recent_past_event = None
                for event in reversed(events):
                    event_start = event['start'].get('dateTime', event['start'].get('date'))
                    if 'T' in event_start:
                        event_start_dt = datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                    else:
                        event_start_dt = datetime.fromisoformat(event_start + 'T00:00:00+00:00')
                    
                    if event_start_dt < current_time:
                        most_recent_past_event = event
                        break
                
                if most_recent_past_event:
                    return self.format_event(most_recent_past_event)
                else:
                    print(f"No past events found for {self.user}")
                    return None
            else:
                print(f"No events found for {self.user}")
                return None
            
        except Exception as e:
            error_msg = f"Failed to get last event: {e}"
            self.error_message = error_msg
            print(f"{error_msg}")
            return None
    
    def get_events_summary(self, num_events: int = 5) -> str:
        try:
            events = self.get_formatted_events(num_events)
            
            if not events:
                return "You have no upcoming events in your calendar."
            
            if len(events) == 1:
                event = events[0]
                return f"Your next event is '{event['summary']}' on {event['start_date']} at {event['start_time']}."
            
            summary_parts = [f"You have {len(events)} upcoming events:"]
            
            for i, event in enumerate(events[:3], 1):
                time_desc = event['start_time'] if event['start_time'] != 'All Day' else 'all day'
                summary_parts.append(f"{i}. '{event['summary']}' on {event['start_date']} at {time_desc}")
            
            if len(events) > 3:
                summary_parts.append(f"...and {len(events) - 3} more events.")
            
            return " ".join(summary_parts)
            
        except Exception:
            return "I'm having trouble accessing your calendar right now."

    def display_all_calendar_types(self):
        self._maybe_refresh_credentials()
        calendar_list = self.service.calendarList().list().execute()
        for cal in calendar_list['items']:
            cal_id = cal['id']
            print(self.calendar_id_to_name(cal_id))
    
    def get_upcoming_events(self, calendar_name: str = "primary", max_results: int = 10, include_past: bool = False) -> List[Dict]:
        try:
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            time_min = None if include_past else datetime.now(timezone.utc).isoformat()
            events = self.get_events(num_events=max_results, time_min=time_min, calendar_id=calendar_id)
            return events
        except Exception:
            return []
    
    def get_day_events(self, date: str, calendar_name: str = "primary", include_past: bool = False) -> List[Dict]:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            end_time = start_time.replace(hour=23, minute=59, second=59)
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            events = self.get_events(
                num_events=50,
                time_min=start_time.isoformat(),
                time_max=end_time.isoformat(),
                calendar_id=calendar_id
            )
            return events
        except Exception:
            return []
    
    def get_week_events(self, calendar_name: str = "primary", include_past: bool = False) -> List[Dict]:
        try:
            now = datetime.now(timezone.utc)
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            time_min = start_of_week.isoformat() if include_past else datetime.now(timezone.utc).isoformat()
            events = self.get_events(
                num_events=100,
                time_min=time_min,
                time_max=end_of_week.isoformat(),
                calendar_id=calendar_id
            )
            return events
        except Exception:
            return []
    
    def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._maybe_refresh_credentials()
            if not self.service:
                raise Exception("Calendar service not initialized")
            
            try:
                self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
            except Exception as rebuild_err:
                if getattr(config, "DEBUG_MODE", False):
                    print(f"Failed to rebuild service: {rebuild_err}")
            
            start_datetime = f"{event_data['date']}T{event_data['start_time']}:00"
            end_datetime = f"{event_data['date']}T{event_data['end_time']}:00"
            
            time_zone = event_data.get('time_zone') or getattr(config, 'DEFAULT_TIME_ZONE', None)
            if time_zone and isinstance(time_zone, str) and '/' in time_zone:
                event_body = {
                    'summary': event_data['title'],
                    'description': event_data.get('description', ''),
                    'start': {
                        'dateTime': start_datetime,
                        'timeZone': time_zone,
                    },
                    'end': {
                        'dateTime': end_datetime,
                        'timeZone': time_zone,
                    },
                }
            else:
                start_datetime += "Z"
                end_datetime += "Z"
                event_body = {
                    'summary': event_data['title'],
                    'description': event_data.get('description', ''),
                    'start': {
                        'dateTime': start_datetime,
                    },
                    'end': {
                        'dateTime': end_datetime,
                    },
                }
            
            if event_data.get('location'):
                event_body['location'] = event_data['location']
            
            if event_data.get('attendees'):
                event_body['attendees'] = [{'email': email} for email in event_data['attendees']]
            
            requested_calendar = event_data.get('calendar_name', 'primary')
            
            if '@' in requested_calendar:
                calendar_id = requested_calendar
            elif requested_calendar == 'primary':
                users_cfg = getattr(config, 'CALENDAR_USERS', {})
                user_config = users_cfg.get(self._user, {})
                calendar_id = user_config.get("calendar_id", "primary")
            else:
                calendar_id = requested_calendar
                try:
                    resolved = self.calendar_name_to_id(requested_calendar)
                    id_to_name, _ = self.get_calendar_mappings()
                    if resolved in id_to_name:
                        calendar_id = resolved
                except Exception:
                    pass
            
            max_retries = 3
            retry_delay = 0.5
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    created_event = self.service.events().insert(
                        calendarId=calendar_id,
                        body=event_body
                    ).execute()
                    
                    return {
                        'id': created_event.get('id'),
                        'htmlLink': created_event.get('htmlLink'),
                        'summary': created_event.get('summary'),
                        'start': created_event.get('start'),
                        'end': created_event.get('end'),
                    }
                except OSError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        
                        try:
                            self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
                        except Exception as rebuild_err:
                            if getattr(config, "DEBUG_MODE", False):
                                print(f"Failed to rebuild service on retry: {rebuild_err}")
                    else:
                        raise
                except Exception:
                    raise
            
            if last_error:
                raise last_error
            
            raise Exception("Failed to create event after retries")
            
        except Exception as e:
            raise Exception(f"Failed to create event: {str(e)}")
