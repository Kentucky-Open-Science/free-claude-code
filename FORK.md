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

- **Dependabot is disabled** by deleting `.github/dependabot.yml` entirely.
  An empty `updates:` list does *not* work — Dependabot keeps running the jobs
  registered from the last valid config, which is how this fork collected seven
  unwanted PRs after it was "disabled". Dependency bumps arrive via upstream.
- **`.github/workflows/upstream-sync.yml`** merges `upstream/main` daily:
  clean merge + green tests push straight to `main`; test failures open a PR
  from the `upstream-sync` branch; merge conflicts push the conflicted tree to
  that same branch, open a PR when the token permits, and fail the job. The
  workflow does not open issues — this repo's default workflow token is
  read-only and `createIssue` is denied, so the job-failure notification is the
  reliable channel.
- **The installers point at this fork's archive** (`REPO_ARCHIVE_URL` in
  `scripts/install.sh`, `$RepoArchiveUrl` in `scripts/install.ps1`). Upstream's
  installer installs upstream's package, which has neither fork provider.
- **No fork-specific version bumps.** `pyproject.toml`'s `version` is left
  exactly as upstream sets it: upstream bumps that line on nearly every commit,
  so diverging on it would turn every daily sync into a conflict. This is a
  deliberate exception to the repo's versioning rule for fork-only commits.
- Fork changes are kept additive and low-surface (profile entries, catalog
  descriptors, settings fields, tests) so automatic merges usually succeed.
  The known recurring conflict point is
  `tests/contracts/test_provider_catalog_order.py` (upstream freezes the
  provider order tuple); resolution is to re-append `openai_compatible`
  (before `lmstudio`) and `mlxvlm` (after `ollama`).
