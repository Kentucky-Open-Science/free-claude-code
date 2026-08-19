# Fork notes — Kentucky-Open-Science/free-claude-code

Fork of [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code)
maintained by Evan Damron. Upstream docs: [README.md](README.md). This file
documents everything the fork adds; it does not exist upstream.

## Fork additions

### `openai_compatible` provider

Any OpenAI-compatible Chat Completions deployment with a user-supplied
endpoint (e.g. the UK LLM Factory).

| Setting | Env var |
|---|---|
| Base URL | `OPENAI_COMPATIBLE_BASE_URL` (e.g. `https://api-llm-factory.ai.uky.edu/v1`) |
| API key | `OPENAI_COMPATIBLE_API_KEY` |
| Proxy (optional) | `OPENAI_COMPATIBLE_PROXY` |

Model refs: `openai_compatible/<model id>`, e.g. `openai_compatible/zai-org/GLM-5.2-FP8`.
Sends `max_completion_tokens`, passes `extra_body` through, consumes
`reasoning_content` deltas, and never sends reasoning-effort fields (arbitrary
backends may reject unknown request fields).

### `mlxvlm` provider (local)

[`mlx_vlm.server`](https://github.com/Blaizzy/mlx-vlm) on Apple Silicon —
OpenAI-compatible serving of MLX vision-language models. No API key.

| Setting | Env var | Default |
|---|---|---|
| Base URL | `MLXVLM_BASE_URL` | `http://localhost:8080/v1` (mlx_vlm.server default) |

Model refs: `mlxvlm/<model id>` where the id is what the server advertises at
`GET /v1/models` (for a locally loaded model this is its path or folder name).
Vision (image blocks in user messages) flows through the OpenAI chat
conversion; `<think>` reasoning is replayed as think tags like llama.cpp.
Appears in the Admin UI local-provider status panel like LM Studio/Ollama.

## Sync policy

- **Dependabot is disabled** (`.github/dependabot.yml` has an empty update
  list). Dependency bumps arrive via upstream.
- **`.github/workflows/upstream-sync.yml`** merges `upstream/main` daily:
  clean merge + green tests push straight to `main`; test failures open a PR
  from the `upstream-sync` branch; merge conflicts open an issue.
- Fork changes are kept additive and low-surface (profile entries, catalog
  descriptors, settings fields, tests) so automatic merges usually succeed.
  The known recurring conflict point is
  `tests/contracts/test_provider_catalog_order.py` (upstream freezes the
  provider order tuple); resolution is to re-append `openai_compatible`
  (before `lmstudio`) and `mlxvlm` (after `ollama`).
