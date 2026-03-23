# Development

## Environment Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Install the optional Spark path when you need local large-file verification:

```bash
uv sync --extra dev --extra spark
```

Add your API key to `.env`:

```dotenv
OPENAI_API_KEY=your-api-key-here
```

## Running the App

Start the Chainlit UI locally:

```bash
./.venv/bin/chainlit run app.py --host 127.0.0.1 --port 8010
```

`python app.py` also works for a quick local run, but the explicit Chainlit command is the most reliable path for browser-driven verification.

## Sample Data

Generate the default small sample dataset:

```bash
./.venv/bin/python scripts/generate_sample_data.py
```

Generate a large CSV for Spark-path testing:

```bash
./.venv/bin/python scripts/generate_large_sample_data.py
```

The large generator writes `data/sample_ab_data_large.csv`, which is intentionally gitignored because it is a local verification artifact.

## Repo Conventions

- `pyproject.toml` is the canonical dependency source.
- `requirements.txt` remains as a compatibility shim for tooling that still expects it.
- The root `README.md` plus the curated files in `docs/` are the only Markdown docs tracked in git.
