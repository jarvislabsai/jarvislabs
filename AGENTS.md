## Code Style

- Python 3.11+, pythonic, simple, readable, maintainable
- **Don't overbloat.** Only write what's needed. No premature abstractions.
- Follow DRY. No duplicate code.
- Split files only when they are genuinely necessary.
- Use type hints on public API functions. Internal code: use them when they help clarity.
- Do not over-engineer. Keep it simple.
- Do not account for backward compatibility for major changes and refactors unless asked.

## Formatting

- Ruff is configured in `pyproject.toml`. Run `uv run ruff format . && uv run ruff check --fix .` after major code changes or before committing.
- Human output follows `DESIGN.md`: help layers, setup/connect/status rules, colour, space and voice.

## Preview harnesses

Claude Code and Codex need `/v1/messages` and `/v1/responses`, which aren't public yet. They stay
out of `jl connect` unless `JL_PREVIEW_HARNESSES=1` is set.

- `JL_ANTHROPIC_URL` and `JL_RESPONSES_URL` point the two writers at a local server.
- `uv run python -m tests.cli.mock_endpoints --port 8765 --log /tmp/requests.jsonl` serves both
  protocols and records every request.
- With the real clients installed, `JL_PREVIEW_E2E=1 uv run pytest tests/cli/test_preview_clients.py`
  lets `jl connect` write each config and completes one turn through the mock.

## Tooling

- Use `uv` for everything: `uv run`, `uv pip`, `uv venv`, `uv build`

## Security

- Never commit secrets, tokens, or .env files

## Git
- Do not commit changes unless you are asked to do so.
- Do not push changes to the main branch unless you are asked to do so.