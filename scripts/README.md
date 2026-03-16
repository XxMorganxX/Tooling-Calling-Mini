# Scripts

Utility scripts for project maintenance and automation.

## Files

| File | Purpose | When to Run |
|------|---------|-------------|
| `generate_openclaw_spec.py` | Regenerates `INTEGRATION_API.md` from `tool_calling_config.json` | After adding/modifying tool schemas |
| `install-hooks.sh` | Installs git hooks into `.git/hooks/` | Once after cloning the repo |
| `pre-commit` | Git pre-commit hook — auto-regenerates `INTEGRATION_API.md` when `tool_calling_config.json` is staged | Runs automatically on `git commit` |

## OpenClaw Spec Generation

`INTEGRATION_API.md` is the external integration specification. It documents all API endpoints, authentication flows, request/response formats, and tool schemas.

```bash
python scripts/generate_openclaw_spec.py
```

The script reads tool schemas from `Model/model_qwen4_finetuning/tool_calling_config.json` and generates a complete API reference at the repo root.

The pre-commit hook handles this automatically: when `tool_calling_config.json` is staged, it regenerates the spec and adds it to the commit. Run `bash scripts/install-hooks.sh` to set up the hook.

## Git Hooks

```bash
bash scripts/install-hooks.sh
```

Installs the `pre-commit` hook that keeps `INTEGRATION_API.md` in sync with tool schema changes.
