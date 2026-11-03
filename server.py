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
  ACEH_SPECIALISTS       comma-separated emails labelled specialist; they can also upload (optional)
  ACEH_CF_ACCOUNT_ID     Cloudflare account that sends sign-in codes
  ACEH_CF_EMAIL_TOKEN    Cloudflare API token with Email Sending: Edit
  ACEH_MAIL_FROM         sender, default login@aceh-nlp.net
  ACEH_MAIL_REPLY_TO     where replies go, default hello@aceh-nlp.net
  ACEH_MAIL_DAILY_CAP    sign-in emails per UTC day (default 100)
  ACEH_MAIL_PER_IP       sign-in emails per visitor per hour (default 30)
  ACEH_REVIEWS_PER_PERSON  reviews per person per UTC day (default 500)
  ACEH_REVIEWS_DAILY_CAP   reviews by everyone per UTC day (default 5000)
  ACEH_GENERATE_PER_IP   anonymous /api/generate calls per IP per hour (default 20)
  ACEH_GENERATE_DAILY_CAP  all /api/generate calls per UTC day (default 500)
  ACEH_PORT              listen port, default 8766 (loopback only)
"""
import base64
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import storage

# Fields never shown to evaluators (blinding).
BLINDED_FIELDS = {"model", "revision", "run_id", "source", "generator", "producer", "uploaded_by"}
EMAIL_RE = re.compile(r"^[^@\s<>,;:\"'()]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
# A dataset name is folders and a file stem: no dots, so no way out of its folder.
DATASET_RE = re.compile(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*")
MAX_UPLOAD_ROWS = 2000
MAX_UPLOAD_BODY = 5 * 1024 * 1024
MAX_BODY = 256 * 1024
MAX_REVIEW_TEXT = 5000
# Anonymous generation gets fewer tokens than a signed-in call.
ANON_MAX_TOKENS = 128
# What a reviewer may submit, besides dataset and item_id.
REVIEW_FIELDS = {"score", "correction", "annotation"}
NOT_REVIEWABLE = "This dataset is not open for review"
UPLOADS_CLOSED = "Uploads are open to coordinators for now"
SIGN_IN = "email_code"  # the only sign-in method tokens are accepted from
CODE_TTL = 600
CODE_ATTEMPTS = 5
TOKEN_TTL = 30 * 86400
CF_API = "https://api.cloudflare.com/client/v4"


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
    except (ValueError, TypeError):
        return None


def clean_text(value):
    """A request's string with unencodable characters dropped; anything else is empty."""
    return value.encode("utf-8", "ignore").decode() if isinstance(value, str) else ""


def signed_in(prof, now=None):
    """True for a token from the emailed-code sign-in that has not expired."""
    now = time.time() if now is None else now
    return bool(prof) and prof.get("method") == SIGN_IN and now - prof.get("iat", 0) < TOKEN_TTL


class SignInCodes:
    """One-time sign-in codes, in memory: a restart only means asking for a new code.

    Each request gets its own secret challenge, which the code only works with. So
    someone who knows an address can't guess against, or use up, another person's code.
    """

    def __init__(self, secret, daily_cap, per_ip_hour):
        self.secret, self.daily_cap, self.per_ip_hour = secret, daily_cap, per_ip_hour
        self.codes, self.sent, self.by_ip = {}, {}, {}
        self.day, self.day_count = None, 0
        self.lock = threading.Lock()

    def _digest(self, challenge, code):
        return hmac.new(self.secret.encode(), f"{challenge}:{code}".encode(),
                        hashlib.sha256).hexdigest()

    def reserve(self, email, ip_hash, now=None):
        """Counts one email against the limits. Returns False when over a limit."""
        now = time.time() if now is None else now
        with self.lock:
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if day != self.day:
                self.day, self.day_count = day, 0
            sent = [t for t in self.sent.get(email, []) if now - t < 86400]
            by_ip = [t for t in self.by_ip.get(ip_hash, []) if now - t < 3600]
            if (self.day_count >= self.daily_cap or len(by_ip) >= self.per_ip_hour
                    or len(sent) >= 10 or sum(now - t < 3600 for t in sent) >= 5
                    or (sent and now - sent[-1] < 60)):
                return False
            self.sent[email], self.by_ip[ip_hash] = sent + [now], by_ip + [now]
            self.day_count += 1
            return True

    def store(self, email, code, now=None):
        """Remembers a sent code. Returns the challenge that goes with it."""
        now = time.time() if now is None else now
        challenge = secrets.token_urlsafe(24)
        with self.lock:
            self.codes = {c: e for c, e in self.codes.items() if now < e["expires"]}
            self.codes[challenge] = {"email": email, "digest": self._digest(challenge, code),
                                     "expires": now + CODE_TTL, "attempts": 0}
        return challenge

    def check(self, challenge, code, now=None):
        """The email, once, for the right code. Too many wrong tries end the challenge."""
        now = time.time() if now is None else now
        digest = self._digest(challenge, code)
        with self.lock:
            entry = self.codes.get(challenge)
            if not entry:
                return None
            if now >= entry["expires"]:
                del self.codes[challenge]
                return None
            if hmac.compare_digest(digest, entry["digest"]):
                del self.codes[challenge]
                return entry["email"]
            entry["attempts"] += 1
            if entry["attempts"] >= CODE_ATTEMPTS:
                del self.codes[challenge]
            return None


WELCOME_SUBJECT = "Welcome to Aceh NLP / Selamat datang di Aceh NLP"
WELCOME_TEXT = """Thank you for joining. Aceh NLP is a community project to keep Acehnese alive in the tools people use.

You can help by scoring and correcting sentences at https://aceh-nlp.net/evaluate/. They are AI drafts, so please fix them the way you would really say them.

Questions, or want to help in another way? Just reply to this email.

- Aceh NLP

---

Terima kasih sudah bergabung. Aceh NLP adalah proyek komunitas untuk menjaga bahasa Aceh tetap hidup dalam alat yang digunakan orang.

Anda bisa membantu dengan menilai dan mengoreksi kalimat di https://aceh-nlp.net/evaluate/. Kalimat-kalimat itu adalah draf AI, jadi perbaiki seperti yang benar-benar Anda ucapkan.

Ada pertanyaan, atau ingin membantu dengan cara lain? Balas saja email ini.
"""


def send_email(cfg, to, subject, text):
    req = urllib.request.Request(
        f"{CF_API}/accounts/{cfg['cf_account_id']}/email/sending/send",
        data=json.dumps({"to": to, "from": cfg["mail_from"], "reply_to": cfg["mail_reply_to"],
                         "subject": subject, "text": text}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['cf_email_token']}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise ValueError(f"mail API returned {exc.code}: {exc.read()[:300]!r}")
    if not (isinstance(result, dict) and result.get("success")):
        raise ValueError(f"mail API refused: {str(result)[:300]}")


def code_email_text(code):
    minutes = CODE_TTL // 60
    return (f"Your Aceh NLP sign-in code is {code}. It expires in {minutes} minutes.\n"
            f"Kode masuk Aceh NLP Anda: {code}. Kode berlaku {minutes} menit.\n\n"
            "If you didn't ask for this, you can ignore this email.\n"
            "Jika Anda tidak memintanya, abaikan email ini.\n")


class Welcomes:
    """Who has had the welcome email, kept as keyed hashes rather than addresses."""

    def __init__(self, secret, state_dir):
        self.secret, self.path = secret, Path(state_dir) / "welcomed.jsonl"
        self.seen = {r.get("email_hash") for r in storage.read_all(self.path)}
        self.lock = threading.Lock()

    def first_time(self, email):
        """True once per address, even across restarts."""
        digest = hmac.new(self.secret.encode(), email.encode(), hashlib.sha256).hexdigest()
        with self.lock:
            if digest in self.seen:
                return False
            storage.append(self.path, {"email_hash": digest, "ts": int(time.time())})
            self.seen.add(digest)
            return True


def welcome(cfg, welcomes, email):
    """Runs on its own thread: a welcome that fails must not hold up signing in."""
    try:
        if welcomes.first_time(email):
            cfg.get("send_mail", send_email)(cfg, email, WELCOME_SUBJECT, WELCOME_TEXT)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f"welcome email failed: {exc}", flush=True)


# ---------- datasets ----------

def list_datasets(dataset_root, state_dir):
    """Reviewable datasets: JSONL files in the dataset repo plus uploads."""
    sets = []
    root = Path(dataset_root) / "data"
    for p in sorted(root.rglob("*.jsonl")):
        if p.name.startswith("."):
            continue
        name = str(p.relative_to(root)).removesuffix(".jsonl")
        if not reviewable(name):
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
    """Rows of a dataset. The caller has checked the name with reviewable()."""
    if dataset.startswith("upload/"):
        path = Path(state_dir) / "uploads" / (dataset.split("/", 1)[1] + ".jsonl")
    else:
        path = Path(dataset_root) / "data" / (dataset + ".jsonl")
    rows = [r for r in storage.read_all(path) if isinstance(r, dict) and "id" in r]
    return rows or None


def blind(item):
    return {k: v for k, v in item.items() if k not in BLINDED_FIELDS}


# ---------- ratings ----------

def tier_of(specialists, prof):
    """A label on the review, from the list. Being signed in proved the email."""
    if prof["email"].lower() in specialists:
        return "specialist"
    return "everyday"


def reviewable(dataset):
    """A well-formed name, outside layer 1 (the result of review) and raw source files."""
    return (bool(DATASET_RE.fullmatch(dataset)) and not dataset.startswith("layer_1_")
            and "/raw/" not in dataset)


def validate_rating(body):
    """Returns ({data, result}, None) for a review, or (None, error)."""
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
            if not isinstance(val, str) or len(val) > MAX_REVIEW_TEXT:
                return None, f"'{opt}' must be a string under {MAX_REVIEW_TEXT} characters"
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


class ReviewLimiter:
    """In-memory per-person and global daily limits for stored reviews."""

    def __init__(self, per_person_day, daily_cap):
        self.per_person_day, self.daily_cap = per_person_day, daily_cap
        self.day, self.by_person, self.total = None, {}, 0
        self.lock = threading.Lock()

    def check(self, email, now=None):
        """True when allowed, and counts the review."""
        now = time.time() if now is None else now
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.lock:
            if day != self.day:
                self.day, self.by_person, self.total = day, {}, 0
            if self.total >= self.daily_cap or self.by_person.get(email, 0) >= self.per_person_day:
                return False
            self.by_person[email] = self.by_person.get(email, 0) + 1
            self.total += 1
            return True


# ---------- server ----------

def guarded(handler):
    """Any failure gets a JSON reply, rather than a dropped connection."""
    def run(self):
        try:
            handler(self)
        except Exception as exc:
            print(f"{self.path.split('?')[0]} failed: {type(exc).__name__}", flush=True)
            try:
                self._send(500, {"error": "Something went wrong"})
            except OSError:
                pass
    return run


def build_handler(cfg):
    secret = cfg["auth_secret"]
    limiter = GenerateLimiter(cfg.get("generate_per_ip", 20), cfg.get("generate_daily_cap", 500))
    codes = SignInCodes(secret, cfg.get("mail_daily_cap", 100), cfg.get("mail_per_ip", 30))
    welcomes = Welcomes(secret, cfg["state"])
    review_limit = ReviewLimiter(cfg.get("reviews_per_person", 500), cfg.get("reviews_daily_cap", 5000))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

        def _body(self):
            limit = MAX_UPLOAD_BODY if self.path == "/api/uploads" else MAX_BODY
            length = int(self.headers.get("Content-Length", 0))
            if length > limit:
                raise ValueError(f"Body over {limit} bytes")
            return json.loads(self.rfile.read(length) or b"{}")

        def _evaluator(self):
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                prof = read_token(secret, auth[7:])
                return prof if signed_in(prof) else None
            return None

        def _ip_hash(self):
            # The Worker names the visitor, since Caddy's X-Forwarded-For would only name the
            # Worker. Direct callers can fake this, so per-visitor limits are never the only limit.
            forwarded = (self.headers.get("X-Aceh-Client-IP")
                         or self.headers.get("X-Forwarded-For", ""))
            ip = forwarded.split(",")[0].strip() or self.client_address[0]
            return hashlib.sha256(f"{secret}:{ip}".encode()).hexdigest()[:16]

        # ----- GET -----
        @guarded
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(url.query))
            if url.path == "/api/health":
                self._send(200, {"ok": True, "gpu": self._gpu_health()})
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
            if not reviewable(dataset):
                self._send(403, {"error": NOT_REVIEWABLE})
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
        @guarded
        def do_POST(self):
            # Only a signed-in person gets a large body read.
            if self.path == "/api/uploads" and not self._evaluator():
                self._send(401, {"error": "Sign in first"})
                return
            try:
                body = self._body()
            except (ValueError, RecursionError) as exc:
                self._send(400, {"error": f"Invalid JSON body: {exc}"})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": "Body must be a JSON object"})
                return
            if self.path == "/api/login/start":
                self._login_start(body)
            elif self.path == "/api/login/verify":
                self._login_verify(body)
            elif self.path == "/api/ratings":
                self._rate(body)
            elif self.path == "/api/uploads":
                self._upload(body)
            elif self.path == "/api/generate":
                self._generate(body)
            else:
                self._send(404, {"error": "Unknown path"})

        def _login_start(self, body):
            if not (cfg.get("send_mail") or (cfg.get("cf_account_id") and cfg.get("cf_email_token"))):
                self._send(503, {"error": "Email sign-in is not set up yet"})
                return
            email = clean_text(body.get("email")).strip().lower()
            if len(email) > 254 or not EMAIL_RE.match(email):
                self._send(400, {"error": "A valid email is required"})
                return
            if not codes.reserve(email, self._ip_hash()):
                self._send(429, {"error": "Too many codes for now; try again later"})
                return
            code = f"{secrets.randbelow(10**6):06d}"
            try:
                cfg.get("send_mail", send_email)(cfg, email, "Aceh NLP sign-in code",
                                                 code_email_text(code))
            except (OSError, ValueError, http.client.HTTPException) as exc:
                print(f"sign-in email failed: {exc}", flush=True)
                self._send(502, {"error": "The email could not be sent; try again later"})
                return
            self._send(200, {"challenge": codes.store(email, code)})

        def _login_verify(self, body):
            code = clean_text(body.get("code")).strip()
            email = None
            if re.fullmatch(r"[0-9]{6}", code):
                email = codes.check(clean_text(body.get("challenge")), code)
            if not email:
                self._send(401, {"error": "That code is wrong or has expired"})
                return
            profile = {"email": email, "name": clean_text(body.get("name"))[:200],
                       "dialect": clean_text(body.get("dialect"))[:100],
                       "method": SIGN_IN, "iat": int(time.time())}
            threading.Thread(target=welcome, args=(cfg, welcomes, email), daemon=True).start()
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
            if not reviewable(data["dataset"]):
                self._send(403, {"error": NOT_REVIEWABLE})
                return
            items = load_items(cfg["dataset_root"], cfg["state"], data["dataset"]) or []
            item = next((i for i in items if i.get("id") == data["item_id"]), None)
            if item is None:
                self._send(404, {"error": f"Unknown item '{data['item_id']}'"})
                return
            if not review_limit.check(prof["email"]):
                self._send(429, {"error": "That is a lot of reviews for one day; try again tomorrow"})
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
            if tier_of(cfg["specialists"], prof) != "specialist":
                self._send(403, {"error": UPLOADS_CLOSED})
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
                clean.append({**r, "id": str(r.get("id") or f"{name}_{n + 1:04d}")})
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
                    # The GPU's 401 is about our token, and must not read as the visitor's.
                    self._send(502 if exc.code == 401 else exc.code, json.loads(exc.read()))
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
           "cf_account_id": os.environ.get("ACEH_CF_ACCOUNT_ID"),
           "cf_email_token": os.environ.get("ACEH_CF_EMAIL_TOKEN"),
           "mail_from": os.environ.get("ACEH_MAIL_FROM", "login@aceh-nlp.net"),
           "mail_reply_to": os.environ.get("ACEH_MAIL_REPLY_TO", "hello@aceh-nlp.net"),
           "mail_daily_cap": int(os.environ.get("ACEH_MAIL_DAILY_CAP", "100")),
           "mail_per_ip": int(os.environ.get("ACEH_MAIL_PER_IP", "30")),
           "reviews_per_person": int(os.environ.get("ACEH_REVIEWS_PER_PERSON", "500")),
           "reviews_daily_cap": int(os.environ.get("ACEH_REVIEWS_DAILY_CAP", "5000")),
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
