# aceh-nlp/backend

**API only.** The single hub between the other three repositories: it serves
data from [aceh-nlp/dataset](https://github.com/aceh-nlp/dataset),
stores evaluator ratings and uploads as append-only JSONL, and proxies
inference to [aceh-nlp/gpu](https://github.com/aceh-nlp/gpu) (open to
anyone, rate-limited). The UI
lives in [aceh-nlp/frontend](https://github.com/aceh-nlp/frontend);
this repository serves no HTML.

Pure Python standard library; no database (JSONL state only, no SQLite).
Reviewers sign in with a code sent to their email, and every rating records
who gave it. Contract: [docs/API.md](docs/API.md).

## Run

```sh
ACEH_STATE=/path/state \
ACEH_DATASET_ROOT=/path/dataset \
ACEH_AUTH_SECRET=$(openssl rand -hex 32) \
ACEH_ADMIN_TOKEN=$(openssl rand -hex 32) \
python3 server.py
```

Sign-in needs `ACEH_CF_ACCOUNT_ID` + `ACEH_CF_EMAIL_TOKEN` (Cloudflare Email Sending; the
token needs Email Sending: Edit). Without them every sign-in gets a 503.

Optional: `ACEH_GPU_URL` + `ACEH_GPU_TOKEN` (inference), `ACEH_SPECIALISTS`
(comma-separated emails labelled specialist, who can also upload; restart to apply
changes), `ACEH_MAIL_FROM` (default `login@aceh-nlp.net`), `ACEH_MAIL_REPLY_TO` (default `hello@aceh-nlp.net`), `ACEH_MAIL_DAILY_CAP` (default
100), `ACEH_MAIL_PER_IP` (default 30 an hour), `ACEH_REVIEWS_PER_PERSON` and `ACEH_REVIEWS_DAILY_CAP` (default 500 and 5000 a
day), `ACEH_GENERATE_PER_IP` and
`ACEH_GENERATE_DAILY_CAP` (public model demo limits, default 20/hour per IP and
500/day), `ACEH_PORT` (default 8766, loopback only).

Deployment (systemd + Caddy serving the frontend and proxying `/api/*`):
[deploy/](deploy/).

## Tests

```sh
python3 -m unittest discover -s tests -v
```
