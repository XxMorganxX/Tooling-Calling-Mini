"""Interactive terminal REPL for the Qwen3-4B tool-calling inference API."""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from src.client import APIError, InferenceClient
from src.config import load_config
from src.conversation import ConversationManager

# ── Theme / console ───────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "thinking": "dim italic",
        "tool": "bold magenta",
        "usage": "dim",
    }
)

console = Console(theme=_THEME)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_response(resp) -> None:
    """Render a ChatResponse to the terminal."""

    # Thinking trace
    if resp.thinking:
        console.print()
        console.print(
            Panel(
                Text(resp.thinking, style="thinking"),
                title="[thinking]Thinking[/thinking]",
                border_style="dim",
                expand=False,
            )
        )

    # Tool calls
    if resp.tool_calls:
        console.print()
        table = Table(
            title="Tool Calls",
            title_style="tool",
            show_header=True,
            header_style="bold",
            expand=False,
        )
        table.add_column("Tool", style="tool")
        table.add_column("Arguments")
        for tc in resp.tool_calls:
            table.add_row(tc.name, json.dumps(tc.arguments, indent=2))
        console.print(table)

    # Main content
    if resp.content:
        console.print()
        console.print(
            Panel(
                Markdown(resp.content),
                title="[bold]Assistant[/bold]",
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
        )

    # Usage stats
    if resp.usage:
        u = resp.usage
        console.print(
            f"  [usage]tokens: {u.prompt_tokens} prompt + "
            f"{u.completion_tokens} completion "
            f"| {u.tokens_per_second:.1f} tok/s[/usage]"
        )


def _print_config(manager: ConversationManager, config) -> None:
    """Display the current runtime configuration."""
    table = Table(title="Current Configuration", expand=False, show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Server URL", config.server_url)
    table.add_row("Timeout", f"{config.timeout}s")
    table.add_row("Thinking", str(manager.enable_thinking))
    table.add_row("Max history", str(config.max_history_messages))
    g = config.generation
    table.add_row("max_tokens", str(g.max_tokens))
    table.add_row("temperature", str(g.temperature))
    table.add_row("top_p", str(g.top_p))
    table.add_row("top_k", str(g.top_k))
    table.add_row("min_p", str(g.min_p))
    table.add_row("repeat_penalty", str(g.repeat_penalty))
    console.print()
    console.print(table)


def _print_help() -> None:
    """Print the list of available slash commands."""
    console.print()
    table = Table(title="Commands", expand=False, show_header=True, header_style="bold")
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/thinking", "Toggle thinking mode on/off")
    table.add_row("/config", "Show current configuration")
    table.add_row("/help", "Show this help message")
    table.add_row("/quit", "Exit the client")
    console.print(table)


# ── Startup ───────────────────────────────────────────────────────────────────


def _startup_health_check(client: InferenceClient) -> bool:
    """Run a health check and print status. Returns True if healthy."""
    try:
        health = client.health()
        if health.status == "ok":
            console.print("[success]Server is online.[/success] ", end="")
            console.print(f"[info]llama-server: {health.llama_server}[/info]")
            return True
        else:
            console.print(
                f"[warning]Server is degraded. "
                f"llama-server: {health.llama_server}[/warning]"
            )
            return True
    except Exception as exc:
        console.print(f"[error]Health check failed:[/error] {exc}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    config = load_config()

    if not config.refresh_token:
        console.print(
            "[error]INFERENCE_REFRESH_TOKEN is not set.[/error] "
            "Create a .env file or set the environment variable."
        )
        sys.exit(1)

    client = InferenceClient(config)
    manager = ConversationManager(client, config)

    console.print()
    console.print(
        Panel(
            "[bold]Tool Mini-Model Client[/bold]\n"
            "Interactive chat with a fine-tuned Qwen3-4B tool-calling model.\n"
            "Type [bold]/help[/bold] for commands.",
            border_style="blue",
            expand=False,
        )
    )

    _startup_health_check(client)

    # Authenticate before entering the chat loop
    try:
        token = client.authenticate()
        console.print(
            f"[success]Authenticated.[/success] "
            f"[info]Key expires at {token.expires_at:%H:%M:%S UTC}[/info]"
        )
    except APIError as exc:
        console.print(f"[error]Authentication failed ({exc.status_code}):[/error] {exc.detail}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[error]Authentication failed:[/error] {exc}")
        sys.exit(1)

    console.print()

    while True:
        try:
            user_input = console.input("[bold cyan]You>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Goodbye.[/info]")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[info]Goodbye.[/info]")
                break

            elif cmd == "/clear":
                manager.clear()
                console.print("[info]Conversation history cleared.[/info]")
                continue

            elif cmd == "/thinking":
                manager.enable_thinking = not manager.enable_thinking
                state = "on" if manager.enable_thinking else "off"
                console.print(f"[info]Thinking mode: {state}[/info]")
                continue

            elif cmd == "/config":
                _print_config(manager, config)
                continue

            elif cmd == "/help":
                _print_help()
                continue

            else:
                console.print(f"[warning]Unknown command: {cmd}[/warning]")
                continue

        # ── Send message ──────────────────────────────────────────────
        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = manager.send(user_input)
            _print_response(response)
        except APIError as exc:
            console.print(f"[error]API error ({exc.status_code}):[/error] {exc.detail}")
        except Exception as exc:
            console.print(f"[error]Error:[/error] {exc}")

        console.print()


if __name__ == "__main__":
    main()
