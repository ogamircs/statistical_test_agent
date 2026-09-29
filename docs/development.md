# Development

## Environment Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
npm ci --prefix frontend    # Node 20.19+ (CI uses 22)
```

Add your API key to `.env`:

```dotenv
OPENAI_API_KEY=your-api-key-here
```

## Running the App

For development, run the API and the Vite dev server side by side:

```bash
uv run uvicorn app:app --reload --port 8000   # API on :8000
cd frontend && npm run dev                    # UI on http://localhost:5173 (proxies /api to :8000)
```

For a production-like single process, build the UI once and let FastAPI serve it:

```bash
npm run build --prefix frontend
uv run uvicorn app:app --port 8000            # UI + API on http://localhost:8000
```

`python app.py` also starts uvicorn (honors `HOST`/`PORT`). Interactive API docs are at `/api/docs`.

## Sample Data

Generate the default small sample dataset:

```bash
./.venv/bin/python scripts/generate_sample_data.py
```

Generate a large CSV (~500k rows) for pandas performance checks:

```bash
./.venv/bin/python scripts/generate_large_sample_data.py
```

The large generator writes `data/sample_ab_data_large.csv`, which is intentionally gitignored because it is a local verification artifact.

## Frontend Scripts

Run from `frontend/`:

| Script | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server with `/api` proxy |
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm test` | Vitest + Testing Library |
| `npm run build` | Typecheck and build `frontend/dist` |

## Repo Conventions

- `pyproject.toml` + `uv.lock` are the canonical Python dependency source; `frontend/package-lock.json` is committed for the UI.
- `requirements.txt` remains as a compatibility shim for tooling that still expects it.
- The root `README.md`, `AGENTS.md` (instructions for AI coding agents), and the curated files in `docs/` are the only Markdown docs tracked in git.
