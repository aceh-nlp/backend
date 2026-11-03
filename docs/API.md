# Backend API

One JSON API under `/api/*`, loopback :8766 behind Caddy. All state is
append-only JSONL under `$ACEH_STATE` — there is no database server and no SQLite.
Datasets are read from a checkout of
[aceh-nlp/dataset](https://github.com/aceh-nlp/dataset) at `$ACEH_DATASET_ROOT`.

## Auth

Anyone with a working email signs in with a one-time code sent to it; every rating
and upload records who did it.

- `POST /api/login/start` with `{email}` → `{challenge}`, once the code has been sent.
  400 for a bad email, 429 over a limit, 502 if the email could not be sent, and 503 until
  sending is configured. Limits: per email one a minute, five an hour and ten a day; per
  visitor `$ACEH_MAIL_PER_IP` an hour; in total `$ACEH_MAIL_DAILY_CAP` a day. A failed send
  still counts, so an address the mail service refuses can't be retried without limit.
- `POST /api/login/verify` with `{challenge, code, name?, dialect?}` → `{token, profile}`,
  or 401. A code lasts 10 minutes, works once and only with its own challenge, and the
  challenge ends after 5 wrong tries. Codes live in memory, hashed, so a restart means
  asking for a new one.
- The token is an HMAC-signed profile; send it as `Authorization: Bearer <token>`. It lasts
  30 days. Tokens from the earlier email and Google sign-ins are refused.
- `GET /api/me` → `{profile, tier}`. `tier` is `specialist` if the email is in
  `$ACEH_SPECIALISTS`, otherwise `everyday`. It is a label checked on every request and
  is not stored in the token.

## Review

- `GET /api/datasets` → `{datasets: [{dataset, rows, origin}]}` — every JSONL
  under the dataset repo's `data/` whose rows all have `id`, plus uploads.
  `layer_1_*` is left out because it is the result of review, not its input, and so
  are `raw/` source files. Both, and any name that isn't plain folders and a file
  stem, get a 403 from the two routes below.
- `GET /api/items?dataset=X&limit=N` (auth) → `{items, remaining, total}` —
  items this reviewer has not reviewed yet. Blinded: `model`, `revision`,
  `run_id`, `source`, `generator`, `producer`, `uploaded_by` fields are stripped.
- `POST /api/ratings` (auth) — `{dataset, item_id, score: 1..5, correction?,
  annotation?}`. `correction` and `annotation` can be up to 5000 characters. Anything else is a
  400, and an unknown item is a 404. Limits, in memory → 429: `$ACEH_REVIEWS_PER_PERSON` a
  day and `$ACEH_REVIEWS_DAILY_CAP` a day from everyone. `reviewer.tier` records the label
  at the time of the review.

  Appended to `$ACEH_STATE/ratings/<dataset>.jsonl` as one record with three
  fields. `data.sha256` is the hash of the row as reviewed. Re-reviewing an item
  adds another line:

  ```json
  {"data": {"dataset": "...", "item_id": "s001", "sha256": "..."},
   "reviewer": {"email": "...", "dialect": "...", "tier": "specialist"},
   "result": {"score": 4, "correction": "...", "annotation": "...", "ts": 1793700000}}
  ```
- `GET /api/export?dataset=X` with `Authorization: Bearer $ACEH_ADMIN_TOKEN`
  → `{dataset, ratings: [...]}`: every record as stored, including reviewer
  identities and older flat records. Summaries are left to consumers.

## Uploads

- `POST /api/uploads` (specialist) — `{name: [A-Za-z0-9_-]{1,80}, rows: [{...}≤2000]}`.
  Rows get ids when missing; stored at `$ACEH_STATE/uploads/<name>.jsonl`;
  reviewable as dataset `upload/<name>`. Duplicate names are rejected (409).

## Inference

- `POST /api/generate` (open) — `{prompt, max_tokens?, temperature?, top_p?, seed?}`
  proxied to `$ACEH_GPU_URL/generate` on the
  [aceh-nlp/gpu](https://github.com/aceh-nlp/gpu) server
  (bearer `$ACEH_GPU_TOKEN` when set). No sign-in needed; anonymous calls are
  capped at 128 tokens, signed-in calls keep their `max_tokens`. Limits:
  `$ACEH_GENERATE_PER_IP` per client IP per hour (default 20, from
  `X-Aceh-Client-IP`, which the Worker sets, else `X-Forwarded-For`) and `$ACEH_GENERATE_DAILY_CAP` per UTC day (default 500),
  both in memory → 429. Each forwarded call appends `{ts, ip_hash,
  prompt_chars, signed_in}` to `$ACEH_STATE/generate_log.jsonl`; prompts are
  not logged. 503 with a clear message when no GPU server is configured or
  reachable.
- `GET /api/health` → `{ok, gpu}` (gpu block reports the
  model the GPU server has loaded, or that none is configured).
