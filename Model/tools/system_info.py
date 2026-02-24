"""
System Information tool — standalone module.

Provides information about the assistant's internal structure,
capabilities, and configuration by reading the developer documentation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.models import SystemInfoArgs

logger = logging.getLogger(__name__)

# inference/tools/ -> inference/ -> Model/
_project_root = Path(__file__).parent.parent.parent
_dev_readme_path = _project_root / "homeassist-ref" / "DEV_README.md"

SECTION_HEADERS: Dict[str, List[str]] = {
    "overview": ["## Architecture Overview", "## Table of Contents"],
    "architecture": ["## Architecture Overview", "### Directory Structure", "### Execution Flow"],
    "providers": ["## Provider Pattern"],
    "orchestrator": ["## Orchestrator & State Machine"],
    "audio": ["## Audio Pipeline"],
    "tools": ["## MCP Tool System"],
    "memory": ["## Memory Systems"],
    "config": ["## Configuration System"],
    "models": ["## Data Models"],
    "scheduled": ["## Scheduled Jobs"],
    "errors": ["## Error Handling"],
    "development": ["## Common Development Tasks"],
    "performance": ["## Performance Considerations", "## Key Implementation Details"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_dev_readme() -> str:
    """Read the DEV_README file and return its contents."""
    try:
        if _dev_readme_path.exists():
            return _dev_readme_path.read_text(encoding="utf-8")
        else:
            return f"[DEV_README.md not found at {_dev_readme_path}]"
    except Exception as e:
        return f"[Error reading DEV_README.md: {str(e)}]"


def _extract_section(content: str, section_name: str) -> str:
    """Extract a specific section from markdown content."""
    lines = content.split("\n")
    result_lines: list[str] = []
    in_section = False
    current_level = 0

    target_headers = SECTION_HEADERS.get(section_name, [])

    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("#").strip()

            matched = False
            for header in target_headers:
                header_clean = header.lstrip("#").strip()
                if header_text == header_clean or header_text.startswith(header_clean):
                    matched = True
                    break

            if matched:
                in_section = True
                current_level = level
                result_lines.append(line)
            elif in_section:
                if level <= current_level:
                    in_section = False
                else:
                    result_lines.append(line)
        elif in_section:
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def _get_quick_summary() -> str:
    """Get a quick summary of the assistant."""
    return (
        "HomeAssist V3 is a fully local, always-listening voice assistant built in Python. "
        "V3 improvements include dramatically faster response times and proactive jump-in briefings.\n\n"
        "Key components:\n"
        "• Wake word detection (OpenWakeWord, process-isolated)\n"
        "• Real-time transcription (AssemblyAI or OpenAI Whisper)\n"
        "• GPT-4o Realtime for responses with MCP tool calling\n"
        "• Text-to-speech (Piper, Google, Chatterbox)\n"
        "• Three-tier memory: conversation context, persistent facts, vector search\n"
        "• Smart home tools: Spotify, Kasa lights, Google Calendar, weather, SMS, etc.\n"
        "• Scheduled briefings: email summaries, news digests, calendar reminders\n\n"
        "The codebase uses a provider pattern with swappable implementations for each component."
    )


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: SystemInfoArgs) -> dict:
    """Execute the system info tool."""
    try:
        section = args.section or "overview"

        logger.debug("Executing Tool: SystemInfo -- section=%s", section)

        dev_readme = _read_dev_readme()

        if "[DEV_README.md not found" in dev_readme or "[Error reading" in dev_readme:
            return {
                "success": False,
                "error": dev_readme,
                "fallback_summary": _get_quick_summary(),
            }

        if section == "all":
            return {
                "success": True,
                "message": "Complete developer documentation retrieved",
                "documentation": dev_readme,
                "summary": _get_quick_summary(),
            }

        elif section == "overview":
            extracted = _extract_section(dev_readme, "overview")
            architecture = _extract_section(dev_readme, "architecture")

            return {
                "success": True,
                "message": "Architecture overview retrieved",
                "summary": _get_quick_summary(),
                "architecture_overview": architecture if architecture else extracted,
                "note": "Use specific section names for detailed info: providers, orchestrator, audio, tools, memory, config, etc.",
            }

        else:
            extracted = _extract_section(dev_readme, section)

            if not extracted:
                return {
                    "success": True,
                    "message": f"Section '{section}' not found as distinct section",
                    "summary": _get_quick_summary(),
                    "available_sections": list(SECTION_HEADERS.keys()),
                    "note": "Try 'all' for complete documentation",
                }

            return {
                "success": True,
                "message": f"Documentation section '{section}' retrieved",
                "section": section,
                "content": extracted,
            }

    except Exception as e:
        logger.error("Failed to retrieve system info: %s", e)
        return {
            "success": False,
            "error": f"Failed to retrieve system documentation: {str(e)}",
            "fallback_summary": _get_quick_summary(),
        }
