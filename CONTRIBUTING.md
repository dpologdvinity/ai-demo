# Contributing

Thanks for your interest in improving the AI Algorithms Demonstration Website. This guide covers local setup, the checks CI runs, and the commit conventions. Architecture notes for both humans and AI agents are in [AGENTS.md](AGENTS.md).

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 24 (the version CI uses) and npm
- Docker (optional, for the full stack)
- [lefthook](https://lefthook.dev), [gitleaks](https://github.com/gitleaks/gitleaks) and [typos](https://github.com/crate-ci/typos) for the pre-commit hooks

## Setup

```bash
# Backend
cd backend
uv venv
uv pip install --torch-backend cpu -r requirements.txt   # CPU torch; tests need no GPU
uv run uvicorn main:app --reload --port 8000            # http://localhost:8000/docs for Swagger UI

# Frontend (second terminal)
cd frontend
npm ci
npm run dev                                             # http://localhost:3000, proxies /api to :8000

# Or run everything in Docker
docker-compose up --build

# Git hooks (once per clone)
lefthook install
```

## Checks (same as CI)

```bash
cd backend && uv run pytest --cov=. --cov-report=term    # backend tests (~15 min)
cd frontend && npm run lint && npm run build            # eslint, then tsc typecheck and vite build
```

The pre-commit hooks run `gitleaks` on staged changes, eslint on staged TypeScript files, and `typos`. If `typos` flags a real domain term, add it to `_typos.toml` rather than skipping the hook.

## Adding an algorithm

1. Add `backend/algorithms/<domain>/<name>/` with `model.py`, `schema.py` (Pydantic) and `data.py`, modeled on an existing sibling.
2. Add the endpoint in `backend/api/routes/<domain>.py` and tests in `backend/tests/test_<name>.py`.
3. Add the demo in `frontend/src/components/algorithm-demos/<domain>/<PascalName>/` and register its route in `frontend/src/App.tsx`.

Smoke-test an endpoint with:

```bash
curl http://localhost:8000/health
```

## Branches, commits and pull requests

- Branch from `master` (for example `feat/<algorithm>` or `fix/<issue>`) and open a pull request against `master`.
- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(ml): add gradient boosting demo`, `fix(api): validate k range`, `docs: ...`, `chore: ...`.
- Keep commits small and atomic. Each one should be a single logical change.
- Never commit secrets, `.env` files (other than `.env.example`) or `*.local.*` files.

## Troubleshooting

- **Import errors in the backend:** run commands with `uv run` from `backend/` so the `.venv` is used.
- **Port conflicts:** change the ports in `docker-compose.yml` or pass `--port` to uvicorn.
- **Container debugging:** use `docker-compose logs -f backend` and `docker-compose exec backend bash`.
