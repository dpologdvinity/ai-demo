# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit discipline (CRITICAL)

- ALWAYS use Conventional Commits: `<type>(<scope>): <description>` where type is `fix`, `feat`, `test`, `docs`, `refactor`, `chore`, etc.
- Think carefully about what belongs in each commit. Do not bundle unrelated changes.
- Review staged changes with `git status` and `git diff --cached` BEFORE committing to verify scope is correct.
- If a commit includes files that shouldn't be together, unstage the wrong ones with `git restore --staged <file>` and create separate commits for each logical unit.
- Do not take shortcuts or combine things that should be separate commits just to avoid extra work.
- Stash unrelated unstaged work before discarding it: `git stash -u` instead of `git restore` for bulk cleanup.

## Parallel agents (CRITICAL)

- When running multiple agents/subagents concurrently on this repo, give each its own git worktree. A shared working tree lets agents race on the same venv, requirements.txt, and uncommitted edits — one agent's cleanup step (e.g. `git restore`) can silently destroy another agent's unverified work.
- Only skip worktree isolation when every concurrent agent is restricted to a disjoint, explicitly-named set of files with no shared config/env touches.

## Project overview

AI Algorithms Demonstration Website: FastAPI backend + React/Vite frontend showcasing ML, Deep Learning, NLP, Computer Vision, and Reinforcement Learning algorithms with interactive demos.

## Commands

### Backend (from `backend/`)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

uvicorn main:app --reload --host 0.0.0.0 --port 8000   # dev server

pytest                                    # run test suite (backend/tests/)
pytest backend/tests/test_api.py::test_x  # single test
bash run_tests.sh                         # wrapper, forwards args to pytest
```

`pytest.ini` scopes discovery to `tests/`. The numerous `backend/test_*.py` files at the backend root (e.g. `test_gan.py`, `test_sift.py`) are ad-hoc/manual scripts left over from feature development, not part of the collected suite — don't assume `pytest` picks them up, and don't add new tests there; put real tests under `backend/tests/`.

### Frontend (from `frontend/`)

```bash
npm install
npm run dev     # port 3000, proxies /api to localhost:8000
npm run build   # tsc --noEmit gate, then vite build
```

Stack: React 18 + TypeScript + Vite + React Router + TanStack Query + Zustand + Tailwind + Shadcn/UI (Radix) + Recharts + Axios. `package.json`, `tsconfig.json`, and `tsconfig.node.json` were missing from the repo until this was fixed — root `.gitignore` had unscoped `*.json` and `lib/` rules that silently excluded them (and `src/lib/`) from every commit. Watch for the same trap if you see a file that should obviously be tracked but `git status` shows nothing: check `git check-ignore -v <path>` before assuming it doesn't exist.

### Docker

```bash
docker-compose up --build      # backend :8000, frontend :3000 (nginx)
docker-compose logs -f backend
docker-compose exec backend bash
```

## Architecture

### Backend layout

- `backend/main.py` — FastAPI app: routers mounted under `/api` prefix, CORS allows `localhost:3000`/`5173` and the Docker `frontend` host, a `lifespan` context manager (currently just logs — model preloading is a TODO), and a `/ws` WebSocket endpoint that currently just echoes (real streaming isn't wired up yet).
- `backend/api/routes/{ml,deep_learning,nlp,computer_vision,reinforcement_learning}.py` — thin route handlers per domain. Business logic belongs in `algorithms/`, not here.
- `backend/algorithms/<domain>/<algorithm_name>/` — one folder per algorithm with `model.py` (implementation), `schema.py` (Pydantic request/response models), `data.py` (sample/synthetic data generation). This is the dominant convention (e.g. `algorithms/ml/random_forest/`, `algorithms/computer_vision/yolo/`).
  - Exception: `backend/algorithms/reinforcement_learning/` breaks this pattern — it's flat files (`dqn.py`, `ppo.py`, `dqn_schema.py`, etc.) rather than per-algorithm subfolders.
  - There is no shared base class/interface across algorithms — each implements its own model class and its own Pydantic schemas independently. `backend/utils/algorithm_metadata.py` (`AlgorithmMetadata` / `AlgorithmRegistry`) is a catalog/metadata layer, not a contract algorithms are required to implement. `backend/utils/response_schemas.py` has generic `TrainingRequest`/`TrainingResponse` templates, but algorithms aren't required to use them.
- `backend/utils/websocket_manager.py` — `ConnectionManager` (connect/disconnect/send_personal_message/broadcast/stream_training_progress) for pushing real-time training progress to clients. Available but not yet consumed by any route.
- Most algorithm subfolders have their own `README.md` (and sometimes `QUICK_START.md`/`API_GUIDE.md`/`FRONTEND_GUIDE.md`) documenting that specific feature — check the algorithm's own folder for design notes before re-deriving them. `backend/algorithms/reinforcement_learning/` keeps its docs (`Q_LEARNING_README.md`, `SARSA_README.md`, `PPO_IMPLEMENTATION.md`) at the domain-folder level since those algorithms are flat files, not subfolders.

### Frontend layout (`frontend/src/`)

- `App.tsx` — React Router route tree.
- `services/` — Axios-based API client (`apiService.ts`-style singleton: `trainAlgorithm()`, `getAlgorithmInfo()`, `listAlgorithms()`, `healthCheck()`).
- `hooks/` — `useAlgorithms`, `useWebSocket`, `useTrainingStream`, etc.; algorithm demo components use TanStack Query + these hooks rather than calling Axios directly.
- `components/algorithm-demos/` — one folder per algorithm, each typically with `Controls.tsx`, `Documentation.tsx`, `Visualization.tsx`, `index.tsx`. **Naming is inconsistent**: some algorithms sit flat under `algorithm-demos/<name>/` (e.g. `k-means/`, `dbscan/`, `gmm/`, `logistic-regression/`), others are nested under a domain folder (e.g. `algorithm-demos/ml/RandomForest/`, `algorithm-demos/nlp/SentimentAnalysis/`). When adding a new demo, check for an existing sibling in the same domain to match its convention rather than inventing a third pattern.
- `components/common/` — shared demo scaffolding (Button, Card, ParameterControl, ResultsPanel, ErrorDisplay, LoadingSpinner). `components/ui/` — Shadcn/UI primitives.

### Adding a new algorithm

1. Add `backend/algorithms/<domain>/<name>/{model.py,schema.py,data.py}` following an existing sibling in that domain.
2. Wire an endpoint in `backend/api/routes/<domain>.py`.
3. Add a matching frontend demo folder under `frontend/src/components/algorithm-demos/`, following whichever convention (flat vs. domain-nested) the domain already uses.
