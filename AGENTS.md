# AGENTS.md

Canonical instructions for AI coding agents (Claude Code, Codex, etc.) working in this repo. `CLAUDE.md` only imports this file.

## Project

AI Algorithms Demonstration Website: a FastAPI backend and a React/Vite frontend with interactive demos of classical AI, ML, deep learning, NLP, computer vision and reinforcement learning algorithms.

## Commands

### Backend (run from `backend/`)

```bash
uv venv && uv pip install --torch-backend cpu -r requirements.txt   # CPU torch avoids multi-GB CUDA wheels
uv run uvicorn main:app --reload --port 8000    # dev server
uv run pytest                                   # full suite (~15 min; trains real models)
uv run pytest tests/test_api.py::<test_name>    # single test
uv run pytest --cov=. --cov-report=term         # what CI runs
```

`pytest.ini` limits test discovery to `tests/`. The `backend/test_*.py`, `demo_*.py` and `verify_*.py` scripts at the backend root are manual scripts left over from feature work, and pytest does not collect them. Put real tests under `backend/tests/`.

The NLP tokenizer downloads its NLTK data (`punkt_tab`, `stopwords`) on first use, so no manual download is needed.

There is no Python linter or type checker configured yet.

### Frontend (run from `frontend/`)

```bash
npm ci
npm run dev     # port 3000; proxies /api and /ws to localhost:8000
npm run lint    # eslint, --max-warnings 0
npm run build   # tsc (noEmit typecheck), then vite build
```

There is no frontend test runner. `npm run build` is the typecheck gate.

### Docker

```bash
docker-compose up --build   # backend :8000, frontend :3000 (nginx proxies /api to backend)
```

### CI and hooks

- `.github/workflows/ci.yml` runs on pushes and PRs to `master`. It has three jobs: backend pytest with coverage on Python 3.11, frontend `npm ci`, `lint` and `build`, and a `docker build backend`. Before pushing, run the same commands locally (listed above).
- `lefthook.yml` defines pre-commit hooks for `gitleaks` on staged files, eslint on staged `ts`/`tsx` files, and `typos`. Enable them with `lefthook install`. Add known false positives to `_typos.toml`; don't skip the hook.

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `CORS_ORIGINS` | backend | Optional. Comma-separated extra CORS origins (set on Render). |
| `VITE_API_URL` | frontend | Optional. Backend origin when it is not same-origin. By default requests are relative and go through the Vite or nginx proxy. |

No secrets are required to run, test or build. Deployment is `render.yaml` for the backend (Docker, `/health` check) and `frontend/vercel.json` for the frontend.

## Architecture

### Backend

- `backend/main.py` mounts six routers under `/api`: `ml`, `deep_learning`, `nlp`, `computer_vision`, `reinforcement_learning` and `classical_ai`. It also serves `/health` and a `/ws` WebSocket that only echoes for now. The `lifespan` hook only logs; model preloading is a TODO.
- `backend/api/routes/<domain>.py` holds thin route handlers. Business logic belongs in `algorithms/`.
- Each algorithm has its own folder, `backend/algorithms/<domain>/<name>/`, containing `model.py` (implementation), `schema.py` (Pydantic request and response models) and `data.py` (sample or synthetic data).
  - Exception: `algorithms/reinforcement_learning/` is flat. It has `dqn.py`, `dqn_schema.py`, `ppo.py` and so on, with its docs (`Q_LEARNING_README.md`, `SARSA_README.md`, `PPO_IMPLEMENTATION.md`) at the domain level.
  - Algorithms share no base class. `utils/algorithm_metadata.py` (`AlgorithmMetadata`, `AlgorithmRegistry`) is a catalog, not a contract. `utils/response_schemas.py` holds optional generic templates.
- `utils/websocket_manager.py` defines a `ConnectionManager` for streaming training progress, but no route uses it yet.
- Many algorithm folders have their own `README.md` or `QUICK_START.md`. Read them before re-deriving a design.

### Frontend (`frontend/src/`)

- `App.tsx` defines the React Router route tree, for example `/ml/linear-regression`.
- `services/api.ts` exports the `apiService` Axios singleton (`trainAlgorithm`, `getAlgorithmInfo`, `listAlgorithms`, `healthCheck`, and others). Demos call it through TanStack Query.
- Each demo lives in `components/algorithm-demos/<domain>/<PascalName>/`, where `<domain>` is one of `classical-ai`, `computer-vision`, `deep-learning`, `ml`, `nlp` or `reinforcement-learning`. A demo usually has `Controls.tsx`, `Documentation.tsx`, `Visualization.tsx` and `index.tsx`.
- `components/common/` holds the shared demo scaffolding (ParameterControl, ResultsPanel, AlgorithmLayout, StepPlayer, and so on). `components/ui/` holds the shadcn/ui (Radix) primitives.
- The stack is React 18, TypeScript, Vite 5, React Router 6, TanStack Query 5, Tailwind 3, Recharts and Axios.

### Adding an algorithm

1. Add `backend/algorithms/<domain>/<name>/{model.py,schema.py,data.py}`, modeled on a sibling in the same domain.
2. Add an endpoint in `backend/api/routes/<domain>.py`, plus tests in `backend/tests/`.
3. Add `frontend/src/components/algorithm-demos/<domain>/<PascalName>/` and a route in `App.tsx`.

## Conventions

- Use Conventional Commits: `type(scope): description`. Keep each commit to one logical change. Review `git diff --cached` before committing, and split unrelated changes into separate commits.
- Never run a discarding command (`git restore`, `git checkout .`, `git reset --hard`) on a dirty tree. Stash unrelated work first with `git stash -u`.
- If a file that should obviously be tracked doesn't show up in `git status`, run `git check-ignore -v <path>` before assuming it doesn't exist.
- Never commit `.env*` (except `.env.example`), credentials or `*.local.*` files.

## Agent workflow

- **Parallel agents get their own git worktree** (under `.worktrees/`, which is gitignored). In a shared tree, agents race on the venv, `requirements.txt` and uncommitted edits, and one agent's cleanup can silently destroy another's unverified work. Skip isolation only when every agent is restricted to a disjoint, explicitly named set of files and none of them touch shared config or environments.
- **Specs and plans:** design specs from brainstorming go in `docs/agents/specs/`, and implementation plans go in `docs/agents/plans/`. Name them `YYYY-MM-DD-<topic>.md` and keep them after the work ships as a record of design decisions.
- **Shared Claude Code config** lives in `.claude/settings.json`. It enables the official `typescript-lsp` and `commit-commands` plugins. Personal permissions and preferences belong in `.claude/settings.local.json`, which is gitignored.
- **Verify before claiming done:** run the backend tests, the frontend lint and build, and report failures with their actual output.
