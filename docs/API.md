# Backend API

One JSON API under `/api/*`, loopback :8766 behind Caddy. All state is
append-only JSONL under `$ACEH_STATE` — there is no database server and no SQLite.
Datasets are read from a checkout of
[aceh-nlp/dataset](https://github.com/aceh-nlp/dataset) at `$ACEH_DATASET_ROOT`.

## Auth

Evaluators sign in; every rating and upload records who did it.

- `POST /api/login` with `{email, name?, dialect?}` → `{token, profile}`.
  The token is an HMAC-signed profile; send it as `Authorization: Bearer <token>`.
- With `ACEH_GOOGLE_CLIENT_ID` set, `POST /api/login` also accepts
  `{google_id_token, dialect?}`; the token is verified server-side against
  Google's tokeninfo endpoint (audience + verified email).
- `GET /api/me` → `{profile, tier}`. `tier` is `specialist` if the email is in
  `$ACEH_SPECIALISTS`, otherwise `everyday`. It is checked on every request and
  is not stored in the token.

Email sign-in is identification, not authentication: fine for a trusted circle
of speakers; switch on Google Sign-In for verified identities.

## Review

- `GET /api/datasets` → `{datasets: [{dataset, rows, origin}]}` — every JSONL
  under the dataset repo's `data/` whose rows all have `id`, plus uploads.
- `GET /api/items?dataset=X&limit=N` (auth) → `{items, remaining, total}` —
  items this evaluator has not rated yet. Blinded: `model`, `revision`,
  `run_id`, `source`, `generator`, `producer` fields are stripped.
- `POST /api/ratings` (auth) — `{dataset, item_id, ...}` plus the fields for the
  reviewer's tier; anything else is a 400, and an unknown item is a 404:
  - `specialist`: `score: 1..5`, `correction?`, `annotation?`
  - `everyday`: `vote: 1 | -1`

  Appended to `$ACEH_STATE/ratings/<dataset>.jsonl` as one record with three
  fields. `data.sha256` is the hash of the row as reviewed. Re-reviewing an item
  adds another line:

  ```json
  {"data": {"dataset": "...", "item_id": "s001", "sha256": "..."},
   "reviewer": {"email": "...", "dialect": "...", "tier": "everyday"},
   "result": {"vote": 1, "ts": 1793700000}}
  ```
- `GET /api/export?dataset=X` with `Authorization: Bearer $ACEH_ADMIN_TOKEN`
  → `{dataset, ratings: [...]}`: every record as stored, including reviewer
  identities and older flat records. Summaries are left to consumers.

## Uploads

- `POST /api/uploads` (auth) — `{name: [A-Za-z0-9_-]{1,80}, rows: [{...}≤2000]}`.
  Rows get ids when missing; stored at `$ACEH_STATE/uploads/<name>.jsonl`;
  reviewable as dataset `upload/<name>`. Duplicate names are rejected (409).

## Inference

- `POST /api/generate` (open) — `{prompt, max_tokens?, temperature?, top_p?, seed?}`
  proxied to `$ACEH_GPU_URL/generate` on the
  [aceh-nlp/gpu](https://github.com/aceh-nlp/gpu) server
  (bearer `$ACEH_GPU_TOKEN` when set). No sign-in needed; anonymous calls are
  capped at 128 tokens, signed-in calls keep their `max_tokens`. Limits:
  `$ACEH_GENERATE_PER_IP` per client IP per hour (default 20, from
  `X-Forwarded-For`) and `$ACEH_GENERATE_DAILY_CAP` per UTC day (default 500),
  both in memory → 429. Each forwarded call appends `{ts, ip_hash,
  prompt_chars, signed_in}` to `$ACEH_STATE/generate_log.jsonl`; prompts are
  not logged. 503 with a clear message when no GPU server is configured or
  reachable.
- `GET /api/health` → `{ok, google_client_id, gpu}` (gpu block reports the
  model the GPU server has loaded, or that none is configured).
