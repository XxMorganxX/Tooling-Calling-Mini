"""
Generate novel test examples for the inference test suite.

Produces test_examples.jsonl in the same format as training_data.jsonl,
covering all 12 tools with diverse phrasings, multi-tool combos,
edge cases, and no-tool conversational inputs.

Usage:
    python generate_test_data.py
    python run.py --data test_examples.jsonl
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DATA = SCRIPT_DIR / "training_data.jsonl"
OUTPUT = SCRIPT_DIR / "test_examples.jsonl"


def load_system_prompt() -> str:
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    return next(m["content"] for m in first["messages"] if m["role"] == "system")


def make_example(system: str, user: str, thinking: str, calls: list[dict]) -> dict:
    calls_json = json.dumps(calls)
    assistant = f"<think>\n{thinking}\n</think>\n{calls_json}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def generate_examples(system: str) -> list[dict]:
    examples = []

    # ── Weather (single tool) ───────────────────────────────────────────
    weather_cases = [
        ("Is it going to rain today?", "The user wants to know about rain today.", [{"name": "weather", "arguments": {"specific_date": "today"}}]),
        ("What's the temperature right now?", "The user wants current temperature. I'll request a short hourly forecast.", [{"name": "weather", "arguments": {"hours": 1}}]),
        ("Will it be nice this weekend?", "The user wants a weekend forecast. Saturday is a specific date.", [{"name": "weather", "arguments": {"specific_date": "saturday"}}]),
        ("Do I need an umbrella tomorrow?", "The user is asking about tomorrow's weather to decide about rain gear.", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
        ("How's the weather looking for the next few hours?", "The user wants a short-term forecast covering the next few hours.", [{"name": "weather", "arguments": {"hours": 6}}]),
        ("Full week weather", "The user wants a complete 7-day forecast.", [{"name": "weather", "arguments": {"days": 7}}]),
        ("What's it like outside?", "The user wants current weather conditions.", [{"name": "weather", "arguments": {"hours": 1}}]),
        ("Weather on Friday", "The user wants weather for a specific day of the week.", [{"name": "weather", "arguments": {"specific_date": "friday"}}]),
    ]

    # ── Spotify (single tool) ──────────────────────────────────────────
    spotify_cases = [
        ("Put on some Coldplay", "The user wants to listen to the artist Coldplay.", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Coldplay", "search_type": "artist"}}]),
        ("Play Hotel California", "The user wants to play a specific track.", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Hotel California", "search_type": "track"}}]),
        ("Turn the volume down to 30", "The user wants to set volume to 30%.", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
        ("What song is this?", "The user wants to know what's currently playing.", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
        ("Next song please", "The user wants to skip to the next track.", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
        ("Go back to the last song", "The user wants to go to the previous track.", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
        ("Stop the music", "The user wants to pause playback.", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
        ("Put on some lo-fi beats", "The user wants to listen to lo-fi beats.", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi beats", "search_type": "track"}}]),
        ("Shuffle my music", "The user wants to toggle shuffle mode.", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
        ("Play Lose Yourself by Eminem", "The user wants a specific track by a specific artist.", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Lose Yourself Eminem", "search_type": "track"}}]),
        ("Crank it up to max volume", "The user wants maximum volume.", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 100}}]),
        ("Loop this song", "The user wants to repeat the current track.", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "track"}}]),
    ]

    # ── Kasa Lighting (single tool) ────────────────────────────────────
    kasa_cases = [
        ("Lights on", "The user wants to turn on a light. Direct control.", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
        ("Kill the lights", "The user wants to turn off a light.", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
        ("Movie time in the living room", "The user wants the movie scene in the living room.", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
        ("Set up the bedroom for sleeping", "The user wants a relaxing scene in the bedroom.", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}}]),
        ("Turn off Light 2", "The user wants to turn off a specific light.", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "off"}}]),
        ("Party lights in the living room", "The user wants the party scene applied.", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
        ("I need reading light", "The user wants the reading scene.", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
        ("Set the mood", "The user wants mood lighting.", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ]

    # ── Calendar (single tool) ─────────────────────────────────────────
    calendar_cases = [
        ("What's on my schedule today?", "The user wants today's calendar events.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
        ("Any meetings tomorrow?", "The user wants tomorrow's events.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
        ("Show me this week's calendar", "The user wants a week overview.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
        ("What are my next 3 events?", "The user wants upcoming events with a limit.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events", "limit": 3}]}}]),
        ("Schedule lunch with Sarah tomorrow at noon", "The user wants to create a calendar event.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch with Sarah", "date": "tomorrow", "start_time": "12:00", "end_time": "13:00"}]}}]),
        ("Am I free on Monday?", "The user wants to check Monday's schedule.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "monday"}]}}]),
        ("What do I have coming up?", "The user wants to see upcoming events.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
        ("Add gym session Wednesday at 7am to 8am", "The user wants to create a calendar event.", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Gym session", "date": "wednesday", "start_time": "07:00", "end_time": "08:00"}]}}]),
    ]

    # ── Stickies (single tool) ─────────────────────────────────────────
    stickies_cases = [
        ("What's on my sticky notes?", "The user wants to read their sticky notes.", [{"name": "stickies", "arguments": {"action": "read"}}]),
        ("Show me my to-do list", "The user wants to see their to-do items.", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
        ("Add buy groceries to my to-do list", "The user wants to add a task.", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "Buy groceries"}]}}]),
        ("Remove the gym task from my list", "The user wants to remove a to-do item.", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "gym"}]}}]),
        ("Jot down: server password is abc123", "The user wants to add a note.", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "Server", "content": "password is abc123"}]}}]),
        ("Check off laundry from my tasks", "The user wants to remove a completed task.", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "laundry"}]}}]),
        ("What notes do I have?", "The user wants to read the notes section.", [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]),
        ("Add pick up dry cleaning to my list, due Friday", "The user wants to add a task with a due date.", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "Pick up dry cleaning", "due": "Friday"}]}}]),
    ]

    # ── Google Search ──────────────────────────────────────────────────
    search_cases = [
        ("What are the best restaurants nearby?", "The user wants restaurant recommendations, which requires a web search.", [{"name": "google_search", "arguments": {"query": "best restaurants nearby", "query_type": "general"}}]),
        ("How tall is the Eiffel Tower?", "The user is asking a factual question that requires a web search.", [{"name": "google_search", "arguments": {"query": "how tall is the Eiffel Tower", "query_type": "general"}}]),
        ("Get me directions to the airport", "The user wants navigation directions.", [{"name": "google_search", "arguments": {"query": "directions to the airport", "query_type": "directions"}}]),
        ("Find the link for Python documentation", "The user wants a specific URL.", [{"name": "google_search", "arguments": {"query": "Python documentation official", "query_type": "link"}}]),
        ("What movies are playing tonight?", "The user wants current movie listings.", [{"name": "google_search", "arguments": {"query": "movies playing tonight near me", "query_type": "general"}}]),
        ("Who won the Super Bowl?", "The user wants recent sports results.", [{"name": "google_search", "arguments": {"query": "who won the Super Bowl", "query_type": "general"}}]),
        ("How do you make sourdough bread?", "The user wants a recipe or instructions.", [{"name": "google_search", "arguments": {"query": "how to make sourdough bread recipe", "query_type": "general"}}]),
        ("Directions to the nearest gas station", "The user wants navigation.", [{"name": "google_search", "arguments": {"query": "nearest gas station", "query_type": "directions"}}]),
    ]

    # ── Briefing ───────────────────────────────────────────────────────
    briefing_cases = [
        ("Remind me to call the dentist at 9am", "The user wants a timed reminder.", [{"name": "briefing", "arguments": {"action": "create", "message": "Call the dentist", "remind_at": "9am"}}]),
        ("Do I have any reminders?", "The user wants to see pending briefings.", [{"name": "briefing", "arguments": {"action": "list"}}]),
        ("Set a high priority reminder: submit tax forms by Friday", "The user wants a high priority briefing.", [{"name": "briefing", "arguments": {"action": "create", "message": "Submit tax forms by Friday", "priority": "high"}}]),
        ("What reminders are pending?", "The user wants to list their briefings.", [{"name": "briefing", "arguments": {"action": "list"}}]),
        ("Remind me tomorrow morning to water the plants", "The user wants a morning reminder.", [{"name": "briefing", "arguments": {"action": "create", "message": "Water the plants", "remind_at": "tomorrow 8am"}}]),
    ]

    # ── Notifications ──────────────────────────────────────────────────
    notification_cases = [
        ("Any new emails?", "The user wants to check email notifications.", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
        ("What notifications do I have?", "The user wants all notifications.", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
        ("Show me the latest news", "The user wants news notifications.", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
        ("Do I have any important notifications?", "The user wants high-priority notifications.", [{"name": "get_notifications", "arguments": {"priority_filter": "high"}}]),
        ("Check for messages", "The user wants to see if there are new notifications.", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
    ]

    # ── Send SMS ───────────────────────────────────────────────────────
    send_sms_cases = [
        ("Text me the grocery list", "The user wants to send themselves a text.", [{"name": "send_sms", "arguments": {"message": "Grocery list reminder"}}]),
        ("Send me a text saying meeting moved to 3pm", "The user wants an SMS notification.", [{"name": "send_sms", "arguments": {"message": "Meeting moved to 3pm"}}]),
        ("Message me: don't forget your keys", "The user wants a text reminder.", [{"name": "send_sms", "arguments": {"message": "Don't forget your keys"}}]),
    ]

    # ── Clipboard ──────────────────────────────────────────────────────
    clipboard_cases = [
        ("What did I copy?", "The user wants to read clipboard contents.", [{"name": "read_clipboard", "arguments": {}}]),
        ("Paste what's in my clipboard", "The user wants to see clipboard contents.", [{"name": "read_clipboard", "arguments": {}}]),
        ("Read my clipboard", "The user wants clipboard contents.", [{"name": "read_clipboard", "arguments": {}}]),
    ]

    # ── System Info ────────────────────────────────────────────────────
    system_info_cases = [
        ("How do you work?", "The user wants to understand the assistant's architecture.", [{"name": "system_info", "arguments": {"section": "overview"}}]),
        ("What tools do you have?", "The user wants to know about available tools.", [{"name": "system_info", "arguments": {"section": "tools"}}]),
        ("Tell me about your audio system", "The user wants audio-specific documentation.", [{"name": "system_info", "arguments": {"section": "audio"}}]),
    ]

    # ── Cursor Composer ────────────────────────────────────────────────
    cursor_cases = [
        ("Refactor the database module to use async", "A coding task for Cursor.", [{"name": "cursor_composer", "arguments": {"prompt": "Refactor the database module to use async"}}]),
        ("Write unit tests for the auth service", "A coding task.", [{"name": "cursor_composer", "arguments": {"prompt": "Write unit tests for the auth service"}}]),
        ("Add error handling to the API routes", "A coding task.", [{"name": "cursor_composer", "arguments": {"prompt": "Add error handling to the API routes"}}]),
        ("Create a dark mode toggle component", "A coding task.", [{"name": "cursor_composer", "arguments": {"prompt": "Create a dark mode toggle component"}}]),
    ]

    # ── Multi-tool combos ──────────────────────────────────────────────
    multi_cases = [
        (
            "Turn on the lights and put on some jazz",
            "Two actions: lights on and play jazz music.",
            [
                {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}},
                {"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}},
            ],
        ),
        (
            "What's the weather and do I have anything on my calendar?",
            "Two requests: weather check and calendar read.",
            [
                {"name": "weather", "arguments": {"hours": 24}},
                {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
            ],
        ),
        (
            "Set movie lights and play the Interstellar soundtrack",
            "Scene lighting and music for movie night.",
            [
                {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}},
                {"name": "spotify_playback", "arguments": {"action": "play", "query": "Interstellar soundtrack", "search_type": "track"}},
            ],
        ),
        (
            "Check my emails and read my notes",
            "Two read operations: notifications and stickies.",
            [
                {"name": "get_notifications", "arguments": {"type_filter": "email"}},
                {"name": "stickies", "arguments": {"action": "read"}},
            ],
        ),
        (
            "Add milk to my shopping list and remind me at 5pm",
            "Add to stickies and create a timed briefing.",
            [
                {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "Buy milk"}]}},
                {"name": "briefing", "arguments": {"action": "create", "message": "Buy milk", "remind_at": "5pm"}},
            ],
        ),
        (
            "Dim the lights and play sleep music in the bedroom",
            "Bedroom relax scene plus calming music.",
            [
                {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}},
                {"name": "spotify_playback", "arguments": {"action": "play", "query": "sleep music", "search_type": "track"}},
            ],
        ),
        (
            "Search for Italian restaurants and add dinner reservation to my calendar tomorrow at 7pm",
            "Web search plus calendar event creation.",
            [
                {"name": "google_search", "arguments": {"query": "Italian restaurants nearby", "query_type": "general"}},
                {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dinner reservation", "date": "tomorrow", "start_time": "19:00", "end_time": "20:30"}]}},
            ],
        ),
        (
            "Turn off all the lights, stop the music, and check tomorrow's schedule",
            "Three actions: lights off, pause music, read calendar.",
            [
                {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}},
                {"name": "spotify_playback", "arguments": {"action": "pause"}},
                {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}},
            ],
        ),
        (
            "Morning routine: weather, calendar, and notifications",
            "Three read operations for a morning briefing.",
            [
                {"name": "weather", "arguments": {"hours": 12}},
                {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
                {"name": "get_notifications", "arguments": {"type_filter": "all"}},
            ],
        ),
        (
            "Text me the wifi password and add it to my notes",
            "Send an SMS and write to stickies.",
            [
                {"name": "send_sms", "arguments": {"message": "WiFi password reminder"}},
                {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "WiFi", "content": "WiFi password"}]}},
            ],
        ),
    ]

    # ── No-tool (conversational) ───────────────────────────────────────
    no_tool_cases = [
        ("Hello!", "A greeting with no actionable request.", []),
        ("That sounds great", "A conversational acknowledgment. No tool needed.", []),
        ("Okay", "Simple acknowledgment. No tool call required.", []),
        ("Never mind", "The user is canceling or dismissing. No action needed.", []),
        ("I'm just thinking out loud", "Casual remark with no request.", []),
        ("Good morning", "A greeting. No tool call needed.", []),
        ("You're the best", "A compliment. No tool action required.", []),
        ("What can you do?", "The user is asking about capabilities. This maps to system_info.", [{"name": "system_info", "arguments": {"section": "tools"}}]),
    ]

    all_cases = (
        weather_cases + spotify_cases + kasa_cases + calendar_cases
        + stickies_cases + search_cases + briefing_cases + notification_cases
        + send_sms_cases + clipboard_cases + system_info_cases + cursor_cases
        + multi_cases + no_tool_cases
    )

    for user_msg, thinking, calls in all_cases:
        examples.append(make_example(system, user_msg, thinking, calls))

    return examples


def main():
    system = load_system_prompt()
    examples = generate_examples(system)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} test examples -> {OUTPUT.name}")

    # Distribution summary
    import re, collections
    tool_counts = collections.Counter()
    multi = 0
    for ex in examples:
        assistant = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        stripped = re.sub(r"<think>.*?</think>", "", assistant, flags=re.DOTALL).strip()
        calls = json.loads(stripped)
        if len(calls) > 1:
            multi += 1
        for c in calls:
            tool_counts[c["name"]] += 1
        if not calls:
            tool_counts["(no tool)"] += 1

    print(f"  Multi-tool: {multi}")
    for name, count in tool_counts.most_common():
        print(f"  {name:<25} {count:>3}")


if __name__ == "__main__":
    main()
