# aceh-nlp/backend

**API only.** The single hub between the other three repositories: it serves
data from [aceh-nlp/dataset](https://github.com/aceh-nlp/dataset),
stores evaluator ratings and uploads as append-only JSONL, and proxies
inference to [aceh-nlp/gpu](https://github.com/aceh-nlp/gpu) (open to
anyone, rate-limited). The UI
lives in [aceh-nlp/frontend](https://github.com/aceh-nlp/frontend);
this repository serves no HTML.

Pure Python standard library; no database (JSONL state only, no SQLite).
Evaluators sign in (email, or Google when configured) and every rating records
who gave it. Contract: [docs/API.md](docs/API.md).

## Run

```sh
ACEH_STATE=/path/state \
ACEH_DATASET_ROOT=/path/dataset \
ACEH_AUTH_SECRET=$(openssl rand -hex 32) \
ACEH_ADMIN_TOKEN=$(openssl rand -hex 32) \
python3 server.py
```

Optional: `ACEH_GPU_URL` + `ACEH_GPU_TOKEN` (inference), `ACEH_GOOGLE_CLIENT_ID`
(Google Sign-In), `ACEH_SPECIALISTS` (comma-separated emails allowed to review, after
Google sign-in; restart to apply changes), `ACEH_GENERATE_PER_IP` and
`ACEH_GENERATE_DAILY_CAP` (public model demo limits, default 20/hour per IP and
500/day), `ACEH_PORT` (default 8766, loopback only).

Deployment (systemd + Caddy serving the frontend and proxying `/api/*`):
[deploy/](deploy/).

## Tests

```sh
python3 -m unittest discover -s tests -v
```
