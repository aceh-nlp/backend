"""Aceh NLP backend: the one API between frontend, dataset, GPU and storage.

Pure standard library. Serves JSON under /api/*; the static frontend and TLS
belong to Caddy (deploy/Caddyfile). State is append-only JSONL (storage.py).

Env:
  ACEH_STATE             state directory (ratings/, uploads/) — required
  ACEH_DATASET_ROOT      checkout of aceh-nlp-dataset — required
  ACEH_AUTH_SECRET       HMAC secret for evaluator tokens — required
  ACEH_ADMIN_TOKEN       bearer token for /api/export — required
  ACEH_GPU_URL           base URL of the aceh-nlp-gpu server (optional)
  ACEH_GPU_TOKEN         bearer token for the gpu server (optional)
  ACEH_GOOGLE_CLIENT_ID  enables Google Sign-In, which specialists need (optional)
  ACEH_SPECIALISTS       comma-separated emails allowed to review, after Google sign-in
  ACEH_GENERATE_PER_IP   anonymous /api/generate calls per IP per hour (default 20)
  ACEH_GENERATE_DAILY_CAP  all /api/generate calls per UTC day (default 500)
  ACEH_PORT              listen port, default 8766 (loopback only)
"""
import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import storage

# Fields never shown to evaluators (blinding).
BLINDED_FIELDS = {"model", "revision", "run_id", "source", "generator", "producer"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_UPLOAD_ROWS = 2000
MAX_BODY = 5 * 1024 * 1024
# Anonymous generation gets fewer tokens than a signed-in call.
ANON_MAX_TOKENS = 128
# What a specialist may submit, besides dataset and item_id.
REVIEW_FIELDS = {"score", "correction", "annotation"}
REVIEW_CLOSED = "Review is open to specialists for now"


# ---------- auth ----------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(secret, profile):
    payload = _b64(json.dumps(profile, ensure_ascii=False).encode())
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def read_token(secret, token):
    """Returns the profile dict, or None if the token is missing/invalid."""
    try:
        payload, sig = token.split(".", 1)
        expect = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expect):
            return None
        profile = json.loads(_unb64(payload))
        return profile if EMAIL_RE.match(profile.get("email", "")) else None
    except (ValueError, json.JSONDecodeError):
        return None


def verify_google_token(id_token, client_id):
    """Verify a Google ID token via Google's tokeninfo endpoint.

    Returns {email, name} or raises ValueError.
    """
    url = "https://oauth2.googleapis.com/tokeninfo?" + urllib.parse.urlencode(
        {"id_token": id_token})
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            info = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Google verification unavailable: {exc}")
    if info.get("aud") != client_id:
        raise ValueError("Google token was issued for a different app")
    if info.get("email_verified") not in ("true", True):
        raise ValueError("Google account email is not verified")
    return {"email": info["email"], "name": info.get("name", "")}


# ---------- datasets ----------

def list_datasets(dataset_root, state_dir):
    """Reviewable datasets: JSONL files in the dataset repo plus uploads."""
    sets = []
    root = Path(dataset_root) / "data"
    for p in sorted(root.rglob("*.jsonl")):
        if p.name.startswith("."):
            continue
        name = str(p.relative_to(root)).removesuffix(".jsonl")
        if name.startswith("layer_1_"):  # already specialist-checked
            continue
        rows = storage.read_all(p)
        if rows and all("id" in r for r in rows):
            sets.append({"dataset": name,
                         "rows": len(rows), "origin": "dataset"})
    for p in sorted((Path(state_dir) / "uploads").glob("*.jsonl")):
        rows = storage.read_all(p)
        sets.append({"dataset": f"upload/{p.stem}", "rows": len(rows), "origin": "upload"})
    return sets


def load_items(dataset_root, state_dir, dataset):
    if dataset.startswith("upload/"):
        path = Path(state_dir) / "uploads" / (dataset.split("/", 1)[1] + ".jsonl")
    else:
        path = Path(dataset_root) / "data" / (dataset + ".jsonl")
        # stop path escape
        if not str(path.resolve()).startswith(str((Path(dataset_root) / "data").resolve())):
            return None
    rows = storage.read_all(path)
    return rows or None


def blind(item):
    return {k: v for k, v in item.items() if k not in BLINDED_FIELDS}


# ---------- ratings ----------

def tier_of(specialists, prof):
    """Specialist needs a verified Google sign-in: email sign-in proves nothing."""
    if prof.get("method") == "google" and prof["email"].lower() in specialists:
        return "specialist"
    return "everyday"


def reviewable(dataset, tier):
    """Specialists review everything below their own layer; pairs for others come later."""
    return tier == "specialist" and not dataset.startswith("layer_1_")


def validate_rating(body):
    """Returns ({data, result}, None) for a specialist review, or (None, error)."""
    if not isinstance(body, dict):
        return None, "Body must be a JSON object"
    dataset, item_id = body.get("dataset"), body.get("item_id")
    if not isinstance(dataset, str) or not isinstance(item_id, str):
        return None, "'dataset' and 'item_id' are required strings"
    extra = set(body) - {"dataset", "item_id"} - REVIEW_FIELDS
    if extra:
        return None, f"Reviews cannot include: {', '.join(sorted(extra))}"
    score = body.get("score")
    if type(score) is not int or not 1 <= score <= 5:
        return None, "'score' must be an integer 1..5"
    result = {"score": score}
    for opt in ("correction", "annotation"):
        val = body.get(opt)
        if val is not None:
            if not isinstance(val, str) or len(val) > 20000:
                return None, f"'{opt}' must be a string under 20000 characters"
            result[opt] = val
    return {"data": {"dataset": dataset, "item_id": item_id}, "result": result}, None


def row_sha256(row):
    """Hash of a row as reviewed, independent of key order."""
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def reviewed_items(records, email):
    """Item ids this reviewer has finished, including older flat records."""
    done = set()
    for r in records:
        if r.get("evaluator_email") == email:
            done.add(r.get("item_id"))
        elif r.get("reviewer", {}).get("email") == email:
            done.add(r.get("data", {}).get("item_id"))
    return done


def safe_name(dataset):
    return re.sub(r"[^A-Za-z0-9_-]", "_", dataset)


# ---------- generation limits ----------

class GenerateLimiter:
    """In-memory per-IP hourly and global daily limits for /api/generate."""

    def __init__(self, per_ip_hour, daily_cap):
        self.per_ip_hour, self.daily_cap = per_ip_hour, daily_cap
        self.by_ip, self.day, self.day_count = {}, None, 0

    def check(self, ip_hash, now=None):
        """Returns None when allowed (and counts the call), else an error message."""
        now = time.time() if now is None else now
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            self.day, self.day_count = day, 0
        recent = [t for t in self.by_ip.get(ip_hash, []) if now - t < 3600]
        if len(recent) >= self.per_ip_hour:
            return f"Limit of {self.per_ip_hour} requests per hour reached; try again later"
        if self.day_count >= self.daily_cap:
            return "The daily limit for the model demo is reached; try again tomorrow"
        recent.append(now)
        self.by_ip[ip_hash] = recent
        self.day_count += 1
        return None


# ---------- server ----------

def build_handler(cfg):
    secret = cfg["auth_secret"]
    limiter = GenerateLimiter(cfg.get("generate_per_ip", 20), cfg.get("generate_daily_cap", 500))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                raise ValueError(f"Body over {MAX_BODY} bytes")
            return json.loads(self.rfile.read(length) or b"{}")

        def _evaluator(self):
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                return read_token(secret, auth[7:])
            return None

        def _ip_hash(self):
            # Behind Caddy every connection is loopback; the real address is forwarded.
            forwarded = self.headers.get("X-Forwarded-For", "")
            ip = forwarded.split(",")[0].strip() or self.client_address[0]
            return hashlib.sha256(f"{secret}:{ip}".encode()).hexdigest()[:16]

        # ----- GET -----
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(url.query))
            if url.path == "/api/health":
                self._send(200, {"ok": True,
                                 "google_client_id": cfg.get("google_client_id"),
                                 "gpu": self._gpu_health()})
            elif url.path == "/api/me":
                prof = self._evaluator()
                if prof:
                    self._send(200, {"profile": prof,
                                     "tier": tier_of(cfg["specialists"], prof)})
                else:
                    self._send(401, {"error": "Sign in first"})
            elif url.path == "/api/datasets":
                self._send(200, {"datasets": list_datasets(cfg["dataset_root"], cfg["state"])})
            elif url.path == "/api/items":
                self._items(q)
            elif url.path == "/api/export":
                self._export(q)
            else:
                self._send(404, {"error": "Unknown path"})

        def _gpu_health(self):
            if not cfg.get("gpu_url"):
                return {"configured": False}
            try:
                with urllib.request.urlopen(cfg["gpu_url"] + "/health", timeout=5) as resp:
                    return {"configured": True, **json.loads(resp.read())}
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                return {"configured": True, "ok": False, "error": str(exc)}

        def _items(self, q):
            prof = self._evaluator()
            if not prof:
                self._send(401, {"error": "Sign in first"})
                return
            dataset = q.get("dataset", "")
            if not reviewable(dataset, tier_of(cfg["specialists"], prof)):
                self._send(403, {"error": REVIEW_CLOSED})
                return
            items = load_items(cfg["dataset_root"], cfg["state"], dataset)
            if items is None:
                self._send(404, {"error": f"Unknown dataset '{dataset}'"})
                return
            ratings_path = Path(cfg["state"]) / "ratings" / (safe_name(dataset) + ".jsonl")
            done = reviewed_items(storage.read_all(ratings_path), prof["email"])
            pending = [blind(i) for i in items if i["id"] not in done]
            limit = min(int(q.get("limit", 10)), 50)
            self._send(200, {"items": pending[:limit], "remaining": len(pending),
                             "total": len(items)})

        def _export(self, q):
            if self.headers.get("Authorization") != f"Bearer {cfg['admin_token']}":
                self._send(401, {"error": "Admin token required"})
                return
            dataset = q.get("dataset", "")
            path = Path(cfg["state"]) / "ratings" / (safe_name(dataset) + ".jsonl")
            self._send(200, {"dataset": dataset, "ratings": storage.read_all(path)})

        # ----- POST -----
        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"Invalid JSON body: {exc}"})
                return
            if self.path == "/api/login":
                self._login(body)
            elif self.path == "/api/ratings":
                self._rate(body)
            elif self.path == "/api/uploads":
                self._upload(body)
            elif self.path == "/api/generate":
                self._generate(body)
            else:
                self._send(404, {"error": "Unknown path"})

        def _login(self, body):
            if body.get("google_id_token"):
                if not cfg.get("google_client_id"):
                    self._send(400, {"error": "Google Sign-In is not configured"})
                    return
                try:
                    ident = verify_google_token(body["google_id_token"], cfg["google_client_id"])
                except ValueError as exc:
                    self._send(401, {"error": str(exc)})
                    return
                profile = {"email": ident["email"], "name": ident["name"],
                           "dialect": str(body.get("dialect", ""))[:100],
                           "method": "google", "iat": int(time.time())}
            else:
                email = str(body.get("email", "")).strip().lower()
                if not EMAIL_RE.match(email):
                    self._send(400, {"error": "A valid email is required"})
                    return
                profile = {"email": email, "name": str(body.get("name", ""))[:200],
                           "dialect": str(body.get("dialect", ""))[:100],
                           "method": "email", "iat": int(time.time())}
            self._send(200, {"token": make_token(secret, profile), "profile": profile})

        def _rate(self, body):
            prof = self._evaluator()
            if not prof:
                self._send(401, {"error": "Sign in first"})
                return
            tier = tier_of(cfg["specialists"], prof)
            review, err = validate_rating(body)
            if err:
                self._send(400, {"error": err})
                return
            data = review["data"]
            if not reviewable(data["dataset"], tier):
                self._send(403, {"error": REVIEW_CLOSED})
                return
            items = load_items(cfg["dataset_root"], cfg["state"], data["dataset"]) or []
            item = next((i for i in items if i.get("id") == data["item_id"]), None)
            if item is None:
                self._send(404, {"error": f"Unknown item '{data['item_id']}'"})
                return
            data["sha256"] = row_sha256(item)
            path = Path(cfg["state"]) / "ratings" / (safe_name(data["dataset"]) + ".jsonl")
            storage.append(path, {"data": data,
                                  "reviewer": {"email": prof["email"],
                                               "dialect": prof.get("dialect", ""),
                                               "tier": tier},
                                  "result": {**review["result"], "ts": int(time.time())}})
            self._send(200, {"saved": True})

        def _upload(self, body):
            prof = self._evaluator()
            if not prof:
                self._send(401, {"error": "Sign in first"})
                return
            name, rows = body.get("name"), body.get("rows")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
                self._send(400, {"error": "'name' must match [A-Za-z0-9_-]{1,80}"})
                return
            if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_UPLOAD_ROWS:
                self._send(400, {"error": f"'rows' must be a list of 1..{MAX_UPLOAD_ROWS}"})
                return
            clean = []
            for n, r in enumerate(rows):
                if not isinstance(r, dict) or not any(isinstance(v, str) and v.strip()
                                                      for v in r.values()):
                    self._send(400, {"error": f"Row {n + 1} is empty or not an object"})
                    return
                clean.append({"id": str(r.get("id") or f"{name}_{n + 1:04d}"), **r})
            if len({r["id"] for r in clean}) != len(clean):
                self._send(400, {"error": "Duplicate ids in upload"})
                return
            path = Path(cfg["state"]) / "uploads" / (name + ".jsonl")
            if path.exists():
                self._send(409, {"error": f"Upload '{name}' already exists"})
                return
            for r in clean:
                storage.append(path, {**r, "uploaded_by": prof["email"]})
            self._send(200, {"dataset": f"upload/{name}", "rows": len(clean)})

        def _generate(self, body):
            # Open to anyone; signed-in callers only get the larger token budget.
            prof = self._evaluator()
            if not cfg.get("gpu_url"):
                self._send(503, {"error": "No GPU server is configured right now"})
                return
            if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
                self._send(400, {"error": "'prompt' must be a non-empty string"})
                return
            ip_hash = self._ip_hash()
            err = limiter.check(ip_hash)
            if err:
                self._send(429, {"error": err})
                return
            params = {k: body.get(k) for k in
                      ("prompt", "max_tokens", "temperature", "top_p", "seed")
                      if body.get(k) is not None}
            if not prof:
                params["max_tokens"] = min(int(params.get("max_tokens") or ANON_MAX_TOKENS),
                                           ANON_MAX_TOKENS)
            storage.append(Path(cfg["state"]) / "generate_log.jsonl",
                           {"ts": int(time.time()), "ip_hash": ip_hash,
                            "prompt_chars": len(body["prompt"]), "signed_in": bool(prof)})
            req = urllib.request.Request(
                cfg["gpu_url"] + "/generate",
                data=json.dumps(params).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {cfg['gpu_token']}"}
                            if cfg.get("gpu_token") else {})},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    self._send(200, json.loads(resp.read()))
            except urllib.error.HTTPError as exc:
                try:
                    self._send(exc.code, json.loads(exc.read()))
                except json.JSONDecodeError:
                    self._send(502, {"error": f"GPU server error {exc.code}"})
            except (urllib.error.URLError, OSError) as exc:
                self._send(503, {"error": f"GPU server unreachable: {exc}"})

    return Handler


def config_from_env():
    cfg = {"state": os.environ.get("ACEH_STATE"),
           "dataset_root": os.environ.get("ACEH_DATASET_ROOT"),
           "auth_secret": os.environ.get("ACEH_AUTH_SECRET"),
           "admin_token": os.environ.get("ACEH_ADMIN_TOKEN"),
           "gpu_url": os.environ.get("ACEH_GPU_URL"),
           "gpu_token": os.environ.get("ACEH_GPU_TOKEN"),
           "google_client_id": os.environ.get("ACEH_GOOGLE_CLIENT_ID"),
           "generate_per_ip": int(os.environ.get("ACEH_GENERATE_PER_IP", "20")),
           "generate_daily_cap": int(os.environ.get("ACEH_GENERATE_DAILY_CAP", "500")),
           "specialists": {e.strip().lower()
                           for e in os.environ.get("ACEH_SPECIALISTS", "").split(",")
                           if e.strip()}}
    missing = [k for k in ("state", "dataset_root", "auth_secret", "admin_token")
               if not cfg[k]]
    if missing:
        raise SystemExit("Missing env: " + ", ".join("ACEH_" + m.upper() for m in missing))
    return cfg


def main():
    cfg = config_from_env()
    port = int(os.environ.get("ACEH_PORT", "8766"))
    server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(cfg))
    print(f"aceh-nlp-backend listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
