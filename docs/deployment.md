# Deployment

The repository ships a multi-stage `Dockerfile` that bundles the Chainlit UI
and LangGraph agent into a single image. The default image is intentionally
slim (no Java runtime, no Spark) and is suitable for a single-container
deployment behind a reverse proxy or PaaS such as Railway or Fly.io.

## Build

```bash
docker build -t statistical-test-agent:latest .
```

The build uses `uv` inside the builder stage to sync the locked dependency
set (`uv sync --extra dev --frozen`). The runtime stage copies only the
resolved virtualenv and project source, then runs as a non-root `appuser`.

## Run

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  statistical-test-agent:latest
```

Then open <http://localhost:8000> in a browser.

The container starts Chainlit with:

```bash
chainlit run app.py --host 0.0.0.0 --port 8000 --headless
```

### Why port 8000 in the container?

The repository's local development workflow (see `README.md`) launches
Chainlit via `python app.py`, which uses Chainlit's default port `8000`.
Other internal docs (e.g. some plan notes referencing port `8010`) describe
side-by-side dev runs where the default port may be in use; the container
has no such conflict, so it stays on `8000` and you map it to whatever you
like on the host (`-p 8080:8000`, `-p 80:8000`, etc.).

## Required environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Used by `langchain-openai` for the agent LLM calls. |

Additional Chainlit / LangChain environment variables (e.g. `CHAINLIT_AUTH_SECRET`,
`LANGCHAIN_TRACING_V2`) can be passed through with extra `-e` flags.

Do **not** bake secrets into the image — `.env` files are excluded by
`.dockerignore`.

## Exposed port

- `8000/tcp` — Chainlit web UI and websocket endpoint. This is the only
  port the image exposes.

## Volumes

For persistence across container restarts, mount these two directories:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/output":/app/output \
  statistical-test-agent:latest
```

- `/app/data` — input CSVs (drop your experiment files here, or upload via
  the UI; uploads land in Chainlit's tmp area inside the container).
- `/app/output` — generated reports and chart artifacts. Without a mount,
  these are lost when the container exits.

Both directories are owned by the non-root `appuser` (UID 1001) inside the
image. If your host bind-mount has different ownership, either run with
`--user "$(id -u):$(id -g)"` or `chown` the host directories accordingly.

## Known limits

- **No Spark in the default image.** The `[spark]` optional extra requires
  PySpark, which in turn requires a Java runtime. Including Java would push
  the image well past 1.5 GB and is unnecessary for the in-memory pandas
  backend that handles the typical workload. To enable Spark you currently
  need a custom image: install `openjdk-17-jre-headless` (or similar) in
  both stages and run `uv sync --extra dev --extra spark --frozen`. Large
  CSVs that would normally trigger the Spark backend will fall back to the
  pandas analyzer; very large inputs may exhaust container memory.
- **Single-process deployment.** Chainlit is started in `--headless` mode
  with one worker. For higher concurrency, run multiple replicas behind a
  load balancer with sticky sessions (Chainlit holds session state in
  memory).
- **Stateless filesystem.** Anything written outside `/app/data` and
  `/app/output` is ephemeral.

## Railway

Railway can build the repository's Dockerfile directly: create a new
service from the GitHub repo, set the `OPENAI_API_KEY` variable in the
service's *Variables* tab, and configure the public port to `8000`.
Railway's persistent-volume feature can be attached at `/app/output` if you
want generated reports to survive restarts; CSV uploads go through the
Chainlit UI, so a `/app/data` volume is optional.

## Fly.io

`fly launch --no-deploy` will detect the Dockerfile and scaffold a
`fly.toml`. Set the internal port to `8000` (`[http_service] internal_port = 8000`)
and provision the OpenAI key with `fly secrets set OPENAI_API_KEY=sk-...`.
For persistence, create a Fly volume and mount it at `/app/output` in the
`[mounts]` section; the default `shared-cpu-1x` machine size with 512 MB
RAM is enough for the pandas backend on small/medium CSVs, bump to 1 GB if
you regularly analyze multi-hundred-MB files.
