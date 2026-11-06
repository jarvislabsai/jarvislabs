from __future__ import annotations

import pytest

from jarvislabs.config_edit import edit_config, replace_block

OPENCODE_JSONC = """\
{
  // OpenCode settings
  "$schema": "https://opencode.ai/config.json",
  "theme": "dark", // keep it dark
  /* providers
     go here */
  "provider": {
    "anthropic": {
      "options": {},
    },
  },

  "model": "anthropic/claude",
}
"""

PI_JSONC = """\
{
    "theme": "dark" // keep it dark
}
"""

CODEX_TOML = """\
# Codex config
model = "gpt-5"  # the default

[model_providers.openai]
name = "OpenAI"

# profiles below
[profiles.fast]
model = "o4-mini"
"""

AIDER_YAML = """\
# Aider settings
model: gpt-4o  # the default

lint-cmd:
    - "python: ruff"
    - 'go: golint'
env:
    TZ: UTC
"""


@pytest.mark.parametrize(
    ("text", "fmt", "set_values", "expected"),
    [
        pytest.param(
            OPENCODE_JSONC,
            "jsonc",
            {
                ("model",): "jarvislabs/deepseek",
                ("provider", "jarvislabs"): {"name": "Jarvislabs", "options": {"baseURL": "https://x"}},
            },
            """\
{
  // OpenCode settings
  "$schema": "https://opencode.ai/config.json",
  "theme": "dark", // keep it dark
  /* providers
     go here */
  "provider": {
    "anthropic": {
      "options": {},
    },
    "jarvislabs": {
      "name": "Jarvislabs",
      "options": {
        "baseURL": "https://x"
      }
    },
  },

  "model": "jarvislabs/deepseek",
}
""",
            id="jsonc-trailing-commas",
        ),
        pytest.param(
            # The new pair's comma goes after "dark", before the comment that sat on that line.
            PI_JSONC,
            "jsonc",
            {("defaultProvider",): "jarvislabs"},
            """\
{
    "theme": "dark", // keep it dark
    "defaultProvider": "jarvislabs"
}
""",
            id="jsonc-comment-on-last-pair",
        ),
        pytest.param(
            CODEX_TOML,
            "toml",
            {
                ("model",): "deepseek",
                ("model_provider",): "jarvislabs",
                ("model_providers", "jarvislabs"): {"name": "Jarvislabs", "base_url": "https://x"},
            },
            """\
# Codex config
model = "deepseek"  # the default
model_provider = "jarvislabs"

[model_providers.openai]
name = "OpenAI"

[model_providers.jarvislabs]
name = "Jarvislabs"
base_url = "https://x"

# profiles below
[profiles.fast]
model = "o4-mini"
""",
            id="toml",
        ),
        pytest.param(
            AIDER_YAML,
            "yaml",
            {("model",): "openai/deepseek", ("openai-api-base",): "https://x"},
            """\
# Aider settings
model: openai/deepseek  # the default

lint-cmd:
    - "python: ruff"
    - 'go: golint'
env:
    TZ: UTC
openai-api-base: https://x
""",
            id="yaml",
        ),
    ],
)
def test_an_edit_keeps_comments_and_formatting(text, fmt, set_values, expected):
    assert edit_config(text, fmt, set_values) == expected


GOOSE_YAML = """\
# Goose settings
GOOSE_MODEL: gpt-4o  # the default

extensions:
  developer:
    enabled: true
"""

CODEX_TOML_NO_PROVIDERS = """\
# Codex config
model = "gpt-5"  # the default

# profiles below
[profiles.fast]
model = "o4-mini"
"""


def test_dotted_keys_are_refused_rather_than_rewritten():
    # Adding a table would make tomlkit render the dotted keys under it; refusing keeps the file.
    with pytest.raises(ValueError, match="dotted keys"):
        edit_config('providers.jarvislabs.type = "x"\n', "toml", {("c",): 1})


def test_setting_the_same_toml_tables_again_leaves_the_text_unchanged():
    # jl re-runs the plan after connect (to refresh the catalog) and on every later connect; each
    # run must not add another blank line between the tables it rewrites.
    values = {
        ("providers", "jarvislabs"): {"type": "openai", "base_url": "https://x", "api_key": "k"},
        ("models", "jarvislabs/a"): {"provider": "jarvislabs", "model": "a", "max_context_size": 1},
        ("models", "jarvislabs/b"): {"provider": "jarvislabs", "model": "b", "max_context_size": 1},
        ("default_model",): "jarvislabs/a",
    }
    once = edit_config("", "toml", values)

    assert edit_config(once, "toml", values) == once


def test_deleting_the_last_toml_table_leaves_no_trailing_blank_line():
    # jl's tables start after a blank line; without one after them, deleting the last must take
    # that blank line with it, or the file ends with one more newline than before.
    original = '# Mine\ndefault_model = "kimi-code/k3"\ntelemetry = false\n'
    added = {
        ("providers", "jarvislabs"): {"type": "openai"},
        ("models", "jarvislabs/x"): {"provider": "jarvislabs"},
    }
    connected = edit_config(original, "toml", added)

    assert edit_config(connected, "toml", {}, delete=added) == original


@pytest.mark.parametrize(
    ("text", "fmt", "replaced", "added"),
    [
        pytest.param(
            OPENCODE_JSONC,
            "jsonc",
            {("model",): "anthropic/claude"},
            {("provider", "jarvislabs"): {"name": "Jarvislabs"}, ("agent", "build", "model"): "jarvislabs/deepseek"},
            id="jsonc-trailing-commas",
        ),
        pytest.param(
            PI_JSONC,
            "jsonc",
            {("theme",): "dark"},
            {("defaultProvider",): "jarvislabs"},
            id="jsonc-comment-on-last-pair",
        ),
        pytest.param(
            CODEX_TOML,
            "toml",
            {("model",): "gpt-5"},
            {("model_provider",): "jarvislabs", ("model_providers", "jarvislabs"): {"name": "Jarvislabs"}},
            id="toml-existing-parent",
        ),
        pytest.param(
            CODEX_TOML_NO_PROVIDERS,
            "toml",
            {("model",): "gpt-5"},
            {("model_provider",): "jarvislabs", ("model_providers", "jarvislabs"): {"name": "Jarvislabs"}},
            id="toml-new-parent",
        ),
        pytest.param(
            GOOSE_YAML,
            "yaml",
            {("GOOSE_MODEL",): "gpt-4o"},
            {("providers", "jarvislabs"): {"enabled": True, "model": "deepseek"}},
            id="yaml",
        ),
    ],
)
def test_deleting_jl_entries_restores_the_original_text(text, fmt, replaced, added):
    # Connect swaps in jl's values and adds its entries; disconnect puts the replaced values
    # back and deletes the added paths.
    connected = edit_config(text, fmt, dict.fromkeys(replaced, "jl") | added)

    assert edit_config(connected, fmt, replaced, delete=added) == text


CRUSHRC = """\
# my crush setup
provider add openai --api-key "$OPENAI_KEY"
"""

CRUSHRC_CONNECTED = """\
# my crush setup
# >>> jl >>>
provider add jarvislabs --type completions
model large jarvislabs/old
# <<< jl <<<
provider add openai --api-key "$OPENAI_KEY"
"""


@pytest.mark.parametrize(
    ("text", "block", "expected"),
    [
        pytest.param(
            CRUSHRC,
            "provider add jarvislabs --type completions\n",
            CRUSHRC + "# >>> jl >>>\nprovider add jarvislabs --type completions\n# <<< jl <<<\n",
            id="added-at-the-end",
        ),
        pytest.param(
            CRUSHRC_CONNECTED,
            "model large jarvislabs/new",
            """\
# my crush setup
# >>> jl >>>
model large jarvislabs/new
# <<< jl <<<
provider add openai --api-key "$OPENAI_KEY"
""",
            id="replaced-in-place",
        ),
        pytest.param(
            CRUSHRC_CONNECTED,
            None,
            """\
# my crush setup
provider add openai --api-key "$OPENAI_KEY"
""",
            id="removed",
        ),
    ],
)
def test_the_jl_block_is_replaced_added_or_removed(text, block, expected):
    assert replace_block(text, block) == expected


@pytest.mark.parametrize(
    ("text", "fmt", "set_values"),
    [
        pytest.param(OPENCODE_JSONC, "jsonc", {("provider", "jarvislabs"): {"name": "Jarvislabs"}}, id="jsonc"),
        # The comment above [profiles.fast] moves into jl's table when it is added, and must stay put.
        pytest.param(CODEX_TOML, "toml", {("model_providers", "jarvislabs"): {"name": "Jarvislabs"}}, id="toml"),
        pytest.param(AIDER_YAML, "yaml", {("model",): "openai/deepseek"}, id="yaml"),
    ],
)
def test_setting_the_same_values_again_changes_nothing(text, fmt, set_values):
    connected = edit_config(text, fmt, set_values)

    assert edit_config(connected, fmt, set_values) == connected


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("jsonc", '{\n  "provider": {\n    "name": "Jarvislabs"\n  }\n}\n'),
        ("toml", '[provider]\nname = "Jarvislabs"\n'),
        ("yaml", "provider:\n  name: Jarvislabs\n"),
    ],
)
def test_an_empty_file_gets_the_new_keys(fmt, expected):
    assert edit_config("", fmt, {("provider", "name"): "Jarvislabs"}) == expected
