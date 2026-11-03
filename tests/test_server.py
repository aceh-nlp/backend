"""End-to-end tests against a live handler on a temp state dir — no network, no GPU."""
import json
import queue
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage  # noqa: E402
from server import (ANON_MAX_TOKENS, CODE_ATTEMPTS, CODE_TTL, SIGN_IN, TOKEN_TTL,  # noqa: E402
                    GenerateLimiter, ReviewLimiter, SignInCodes, build_handler, make_token, read_token,
                    reviewable, row_sha256, send_code_email, signed_in, tier_of,
                    validate_rating)

SECRET = "test-secret"
ADMIN = "test-admin"


class TokenTests(unittest.TestCase):
    def test_roundtrip(self):
        token = make_token(SECRET, {"email": "a@b.co", "name": "A"})
        self.assertEqual(read_token(SECRET, token)["email"], "a@b.co")

    def test_tamper_rejected(self):
        token = make_token(SECRET, {"email": "a@b.co"})
        self.assertIsNone(read_token(SECRET, token[:-2] + "xx"))
        self.assertIsNone(read_token("other-secret", token))


class SignedInTests(unittest.TestCase):
    def test_only_fresh_code_sign_ins_count(self):
        self.assertTrue(signed_in({"method": SIGN_IN, "iat": 1000}, now=1000))
        self.assertFalse(signed_in({"method": SIGN_IN, "iat": 1000}, now=1000 + TOKEN_TTL))
        for old in ("email", "google", None):
            self.assertFalse(signed_in({"method": old, "iat": 1000}, now=1000))
        self.assertFalse(signed_in(None))


class SignInCodeTests(unittest.TestCase):
    EMAIL = "spec@example.com"

    def setUp(self):
        self.codes = SignInCodes("s", daily_cap=100, per_ip_hour=30)
        self.now = 1_000_000.0

    def _store(self, code="123456"):
        return self.codes.store(self.EMAIL, code, now=self.now)

    def test_code_works_once(self):
        challenge = self._store()
        self.assertEqual(self.codes.check(challenge, "123456", now=self.now), self.EMAIL)
        self.assertIsNone(self.codes.check(challenge, "123456", now=self.now))

    def test_code_only_works_with_its_challenge(self):
        self._store()
        other = self.codes.store("other@example.com", "654321", now=self.now)
        self.assertIsNone(self.codes.check(other, "123456", now=self.now))
        self.assertIsNone(self.codes.check("made-up", "123456", now=self.now))

    def test_code_expires(self):
        challenge = self._store()
        self.assertIsNone(self.codes.check(challenge, "123456", now=self.now + CODE_TTL))

    def test_too_many_wrong_tries_end_the_challenge(self):
        challenge = self._store()
        for _ in range(CODE_ATTEMPTS):
            self.assertIsNone(self.codes.check(challenge, "000000", now=self.now))
        self.assertIsNone(self.codes.check(challenge, "123456", now=self.now))

    def test_wrong_tries_elsewhere_leave_a_code_alone(self):
        mine = self._store()
        theirs = self.codes.store(self.EMAIL, "111111", now=self.now)
        for _ in range(CODE_ATTEMPTS):
            self.codes.check(theirs, "000000", now=self.now)
        self.assertEqual(self.codes.check(mine, "123456", now=self.now), self.EMAIL)

    def test_expired_codes_are_dropped(self):
        self._store()
        self.codes.store("other@example.com", "654321", now=self.now + CODE_TTL)
        self.assertEqual(len(self.codes.codes), 1)

    def _reserve(self, at=0.0, email=None, ip="ip"):
        return self.codes.reserve(email or self.EMAIL, ip, now=self.now + at)

    def test_per_email_limits(self):
        self.assertTrue(self._reserve())
        self.assertFalse(self._reserve(59))  # one a minute
        for i in range(1, 5):
            self.assertTrue(self._reserve(60 * i))
        self.assertFalse(self._reserve(300))  # five an hour
        for i in range(1, 6):
            self.assertTrue(self._reserve(3600 * i))
        self.assertFalse(self._reserve(3600 * 6))  # ten a day

    def test_per_visitor_limit(self):
        codes = SignInCodes("s", daily_cap=100, per_ip_hour=2)
        for i in range(2):
            self.assertTrue(codes.reserve(f"{i}@example.com", "ip", now=self.now))
        self.assertFalse(codes.reserve("2@example.com", "ip", now=self.now))
        self.assertTrue(codes.reserve("2@example.com", "other", now=self.now))
        self.assertTrue(codes.reserve("2@example.com", "ip", now=self.now + 3600))

    def test_daily_cap_covers_all_emails(self):
        codes = SignInCodes("s", daily_cap=2, per_ip_hour=30)
        self.assertTrue(codes.reserve("a@example.com", "ip", now=self.now))
        self.assertTrue(codes.reserve("b@example.com", "ip", now=self.now))
        self.assertFalse(codes.reserve("c@example.com", "ip", now=self.now))
        self.assertTrue(codes.reserve("c@example.com", "ip", now=self.now + 86400))


class SendCodeEmailTests(unittest.TestCase):
    CFG = {"cf_account_id": "acct", "cf_email_token": "tok", "mail_from": "login@aceh-nlp.net"}

    def _send(self, reply):
        resp = unittest.mock.MagicMock()
        resp.__enter__.return_value.read.return_value = json.dumps(reply).encode()
        with unittest.mock.patch("urllib.request.urlopen", return_value=resp) as call:
            send_code_email(self.CFG, "spec@example.com", "123456")
        return call.call_args[0][0]

    def test_request(self):
        req = self._send({"success": True})
        self.assertTrue(req.full_url.endswith("/accounts/acct/email/sending/send"))
        self.assertEqual(req.get_header("Authorization"), "Bearer tok")
        body = json.loads(req.data)
        self.assertEqual((body["to"], body["from"]), ("spec@example.com", "login@aceh-nlp.net"))
        self.assertIn("123456", body["text"])
        self.assertNotIn("123456", body["subject"])

    def test_refusal_raises(self):
        for reply in ({"success": False, "errors": [{"message": "no"}]}, ["odd"]):
            with self.assertRaises(ValueError):
                self._send(reply)


class RatingValidationTests(unittest.TestCase):
    ITEM = {"dataset": "d", "item_id": "i"}

    def test_score_range(self):
        for score in (0, 6, True, "4"):
            _, err = validate_rating({**self.ITEM, "score": score})
            self.assertIn("1..5", err)

    def test_accepts_correction_and_annotation(self):
        rec, err = validate_rating({**self.ITEM, "score": 4, "correction": "Lôn jak u keude.",
                                    "annotation": "spelling"})
        self.assertIsNone(err)
        self.assertEqual(rec["result"], {"score": 4, "correction": "Lôn jak u keude.",
                                         "annotation": "spelling"})

    def test_rejects_other_fields(self):
        for extra in ({"vote": 1}, {"rating": 3}):
            _, err = validate_rating({**self.ITEM, "score": 4, **extra})
            self.assertIn("cannot include", err)


class TierTests(unittest.TestCase):
    SPECIALISTS = {"spec@example.com"}

    def test_specialist_means_on_the_list(self):
        self.assertEqual(tier_of(self.SPECIALISTS, {"email": "Spec@example.com"}), "specialist")
        self.assertEqual(tier_of(self.SPECIALISTS, {"email": "o@example.com"}), "everyday")

    def test_reviewable_layers(self):
        self.assertTrue(reviewable("layer_3_synthetic/x"))
        self.assertTrue(reviewable("layer_2_everyday/x"))
        self.assertFalse(reviewable("layer_1_specialist/x"))


class ApiTests(unittest.TestCase):
    ROW_S1 = {"id": "s1", "ace_draft": "Peue haba?", "model": "secret-model"}

    mail_fails = False

    @classmethod
    def _send_mail(cls, _cfg, to, code):
        if cls.mail_fails:
            raise ValueError("mail API returned 403")
        cls.mail.put((to, code))

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "dataset/data/demo").mkdir(parents=True)
        with open(root / "dataset/data/demo/sentences.jsonl", "w") as f:
            f.write(json.dumps(cls.ROW_S1) + "\n")
            f.write(json.dumps({"id": "s2", "ace_draft": "Lôn jak u keude."}) + "\n")
        (root / "dataset/data/layer_1_specialist").mkdir(parents=True)
        with open(root / "dataset/data/layer_1_specialist/checked.jsonl", "w") as f:
            f.write(json.dumps({"id": "c1", "ace": "Peue haba?"}) + "\n")
        cls.state = root / "state"
        cls.mail = queue.Queue()
        cfg = {"state": str(cls.state), "dataset_root": str(root / "dataset"),
               "auth_secret": SECRET, "admin_token": ADMIN,
               "gpu_url": None, "gpu_token": None,
               "send_mail": cls._send_mail,
               "specialists": {"spec@example.com"}}
        cls.cfg = cfg
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(cfg))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def _raw(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _call(self, method, path, payload=None, token=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _login(self, email="eva@example.com"):
        """Token for someone signed in who is not on the specialist list."""
        return make_token(SECRET, {"email": email, "name": "Eva", "dialect": "pidie",
                                   "method": SIGN_IN, "iat": int(time.time())})

    def test_health_open(self):
        status, data = self._call("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])

    def test_login_rejects_bad_email(self):
        status, _ = self._call("POST", "/api/login/start", {"email": "nope"})
        self.assertEqual(status, 400)

    def test_old_login_is_gone(self):
        status, _ = self._call("POST", "/api/login", {"email": "spec@example.com"})
        self.assertEqual(status, 404)

    def test_code_sign_in(self):
        status, data = self._call("POST", "/api/login/start", {"email": " New@Example.com "})
        self.assertEqual(status, 200)
        self.assertEqual(set(data), {"challenge"})
        challenge = data["challenge"]
        to, code = self.mail.get(timeout=5)
        self.assertEqual(to, "new@example.com")
        status, _ = self._call("POST", "/api/login/verify", {"challenge": challenge, "code": "abc"})
        self.assertEqual(status, 401)
        status, data = self._call("POST", "/api/login/verify",
                                  {"challenge": challenge, "code": f" {code} ",
                                   "name": "New", "dialect": "pidie"})
        self.assertEqual(status, 200)
        self.assertEqual(data["profile"]["email"], "new@example.com")
        self.assertEqual(data["profile"]["dialect"], "pidie")
        _, me = self._call("GET", "/api/me", token=data["token"])
        self.assertEqual(me["tier"], "everyday")
        status, _ = self._call("POST", "/api/login/verify", {"challenge": challenge, "code": code})
        self.assertEqual(status, 401)

    def test_second_request_for_an_email_waits(self):
        self.assertEqual(self._call("POST", "/api/login/start", {"email": "w@example.com"})[0], 200)
        self.mail.get(timeout=5)
        status, _ = self._call("POST", "/api/login/start", {"email": "w@example.com"})
        self.assertEqual(status, 429)
        self.assertTrue(self.mail.empty())

    def test_failed_send_is_reported_and_still_counts(self):
        type(self).mail_fails = True
        try:
            status, data = self._call("POST", "/api/login/start", {"email": "f@example.com"})
        finally:
            type(self).mail_fails = False
        self.assertEqual(status, 502)
        self.assertNotIn("challenge", data)
        self.assertEqual(self._call("POST", "/api/login/start", {"email": "f@example.com"})[0], 429)

    def test_odd_addresses_are_refused(self):
        for email in ("name<v@x.org>", "a@b", "a@b.co.", "a b@c.org", "a,b@c.org", "a@b.c"):
            self.assertEqual(self._call("POST", "/api/login/start", {"email": email})[0], 400, email)

    def test_junk_login_bodies_get_a_reply(self):
        for path in ("/api/login/start", "/api/login/verify"):
            for body in (b"[]", b"null", b"5", b'"x"', b"[" * 200000):
                status, _ = self._raw(path, body)
                self.assertEqual(status, 400, (path, body[:9]))
        status, _ = self._raw("/api/login/verify", b'{"challenge":"\\ud800","code":"\\ud800"}')
        self.assertEqual(status, 401)
        status, _ = self._raw("/api/login/start", b'{"email":5}')
        self.assertEqual(status, 400)

    def test_junk_name_cannot_spoil_a_good_code(self):
        _, data = self._call("POST", "/api/login/start", {"email": "j@example.com"})
        _, code = self.mail.get(timeout=5)
        status, body = self._raw("/api/login/verify", json.dumps(
            {"challenge": data["challenge"], "code": code, "name": "\ud800", "dialect": ["x"]}).encode())
        self.assertEqual(status, 200)
        self.assertEqual((body["profile"]["name"], body["profile"]["dialect"]), ("", ""))

    def test_odd_bearer_token_gets_401(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/me",
                                     headers={"Authorization": "Bearer x.\u00e9"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)
        ctx.exception.close()

    def test_older_tokens_are_refused(self):
        for method in ("email", "google"):
            token = make_token(SECRET, {"email": "spec@example.com", "method": method,
                                        "iat": int(time.time())})
            status, _ = self._call("GET", "/api/me", token=token)
            self.assertEqual(status, 401)

    def _specialist(self, email="spec@example.com"):
        return make_token(SECRET, {"email": email, "dialect": "pidie", "method": SIGN_IN,
                                   "iat": int(time.time())})

    def _export_for(self, email):
        status, data = self._call("GET", "/api/export?dataset=demo/sentences", token=ADMIN)
        self.assertEqual(status, 200)
        return [r for r in data["ratings"] if r.get("reviewer", {}).get("email") == email]

    def test_items_require_sign_in_and_blind(self):
        status, _ = self._call("GET", "/api/items?dataset=demo/sentences")
        self.assertEqual(status, 401)
        status, data = self._call("GET", "/api/items?dataset=demo/sentences", token=self._login())
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 2)
        self.assertNotIn("model", data["items"][0])  # blinded

    def test_me_reports_tier(self):
        _, data = self._call("GET", "/api/me", token=self._specialist())
        self.assertEqual(data["tier"], "specialist")
        # not on the list: signed in, but labelled everyday
        _, data = self._call("GET", "/api/me", token=self._login("gone@example.com"))
        self.assertEqual(data["tier"], "everyday")

    def test_specialist_record_and_export(self):
        token = self._specialist()
        status, _ = self._call("POST", "/api/ratings",
                               {"dataset": "demo/sentences", "item_id": "s1", "score": 2,
                                "correction": "Peue haba, Bang?", "annotation": "polite"},
                               token=token)
        self.assertEqual(status, 200)
        # item disappears from this reviewer's queue
        _, data = self._call("GET", "/api/items?dataset=demo/sentences", token=token)
        self.assertEqual({i["id"] for i in data["items"]}, {"s2"})
        status, _ = self._call("GET", "/api/export?dataset=demo/sentences", token="wrong")
        self.assertEqual(status, 401)
        record = self._export_for("spec@example.com")[0]
        self.assertEqual(set(record), {"data", "reviewer", "result"})
        self.assertEqual(record["data"]["sha256"], row_sha256(self.ROW_S1))
        self.assertEqual(record["reviewer"], {"email": "spec@example.com", "dialect": "pidie",
                                              "tier": "specialist"})
        self.assertEqual(record["result"]["score"], 2)
        self.assertIsInstance(record["result"]["ts"], int)

    def test_re_review_keeps_both(self):
        token = self._specialist("again@example.com")
        self.cfg["specialists"].add("again@example.com")
        try:
            for score in (1, 5):
                status, _ = self._call("POST", "/api/ratings",
                                       {"dataset": "demo/sentences", "item_id": "s1", "score": score},
                                       token=token)
                self.assertEqual(status, 200)
        finally:
            self.cfg["specialists"].discard("again@example.com")
        self.assertEqual([r["result"]["score"] for r in self._export_for("again@example.com")], [1, 5])

    def test_anyone_signed_in_can_review_and_is_labelled_everyday(self):
        item = {"dataset": "demo/sentences", "item_id": "s2", "score": 3, "correction": "x"}
        self.assertEqual(self._call("POST", "/api/ratings", item)[0], 401)
        status, _ = self._call("POST", "/api/ratings", item, token=self._login("day@example.com"))
        self.assertEqual(status, 200)
        self.assertEqual([r["reviewer"]["tier"] for r in self._export_for("day@example.com")],
                         ["everyday"])

    def test_layer_1_is_not_reviewable_or_listed(self):
        token = self._specialist()
        _, data = self._call("GET", "/api/datasets")
        self.assertNotIn("layer_1_specialist/checked", [d["dataset"] for d in data["datasets"]])
        status, _ = self._call("GET", "/api/items?dataset=layer_1_specialist/checked", token=token)
        self.assertEqual(status, 403)
        status, _ = self._call("POST", "/api/ratings",
                               {"dataset": "layer_1_specialist/checked", "item_id": "c1", "score": 5},
                               token=token)
        self.assertEqual(status, 403)

    def test_unknown_item_rejected(self):
        token = self._specialist()
        for item in ({"dataset": "demo/sentences", "item_id": "nope"},
                     {"dataset": "demo/missing", "item_id": "s1"}):
            status, _ = self._call("POST", "/api/ratings", {**item, "score": 3}, token=token)
            self.assertEqual(status, 404)

    def test_removing_specialist_changes_the_label_on_the_next_review(self):
        token = self._specialist("former@example.com")
        item = {"dataset": "demo/sentences", "item_id": "s1", "score": 5}
        self.cfg["specialists"].add("former@example.com")
        try:
            self.assertEqual(self._call("POST", "/api/ratings", item, token=token)[0], 200)
        finally:
            self.cfg["specialists"].discard("former@example.com")
        self.assertEqual(self._call("POST", "/api/ratings", item, token=token)[0], 200)
        self.assertEqual([r["reviewer"]["tier"] for r in self._export_for("former@example.com")],
                         ["specialist", "everyday"])

    def test_older_records_still_count_as_done(self):
        storage.append(self.state / "ratings" / "demo_sentences.jsonl",
                       {"dataset": "demo/sentences", "item_id": "s1", "rating": 4,
                        "evaluator_email": "old@example.com", "ts": 1.0})
        self.cfg["specialists"].add("old@example.com")
        try:
            _, data = self._call("GET", "/api/items?dataset=demo/sentences",
                                 token=self._specialist("old@example.com"))
        finally:
            self.cfg["specialists"].discard("old@example.com")
        self.assertEqual({i["id"] for i in data["items"]}, {"s2"})

    def test_upload_and_review(self):
        status, _ = self._call("POST", "/api/uploads",
                               {"name": "nope", "rows": [{"question": "x", "answer": "y"}]},
                               token=self._login("coord@example.com"))
        self.assertEqual(status, 403)
        token = self._specialist()
        status, data = self._call("POST", "/api/uploads",
                                  {"name": "my_pairs",
                                   "rows": [{"question": "Peue nyoe?", "answer": "Boh mamplam."}]},
                                  token=token)
        self.assertEqual(status, 200)
        self.assertEqual(data["dataset"], "upload/my_pairs")
        status, data = self._call("GET", "/api/items?dataset=upload/my_pairs",
                                  token=self._specialist())
        self.assertEqual(data["total"], 1)
        # duplicate name rejected
        status, _ = self._call("POST", "/api/uploads",
                               {"name": "my_pairs", "rows": [{"question": "x", "answer": "y"}]},
                               token=token)
        self.assertEqual(status, 409)

    def test_generate_without_gpu_is_503_even_anonymously(self):
        status, data = self._call("POST", "/api/generate", {"prompt": "hai"})
        self.assertEqual(status, 503)

    def test_dataset_names_cannot_leave_their_folder(self):
        token = self._login()
        secret = Path(self.tmp.name) / "secret.jsonl"
        secret.write_text(json.dumps({"id": "x", "private": "leak"}) + "\n")
        for name in ("../../etc/passwd", "upload/../../secret", f"upload/{secret.with_suffix('')}",
                     "./layer_1_specialist/checked", "demo/../layer_1_specialist/checked",
                     "layer_1_specialist/checked", "demo//sentences", "demo/raw/x", "%00"):
            query = urllib.parse.quote(name, safe="%")
            status, data = self._call("GET", f"/api/items?dataset={query}", token=token)
            self.assertEqual(status, 403, name)
            self.assertNotIn("items", data)
            status, _ = self._call("POST", "/api/ratings",
                                   {"dataset": name, "item_id": "x", "score": 3}, token=token)
            self.assertEqual(status, 403, name)

    def test_uploader_email_is_not_shown_to_reviewers(self):
        self._call("POST", "/api/uploads", {"name": "blind_check", "rows": [{"ace": "x"}]},
                   token=self._specialist())
        _, data = self._call("GET", "/api/items?dataset=upload/blind_check", token=self._login())
        self.assertEqual(data["total"], 1)
        self.assertNotIn("uploaded_by", data["items"][0])

    def test_uploaded_ids_become_strings(self):
        self._call("POST", "/api/uploads", {"name": "ids_check", "rows": [
            {"id": 7, "ace": "x"}, {"id": None, "ace": "y"}, {"id": [1], "ace": "z"}]},
            token=self._specialist())
        _, data = self._call("GET", "/api/items?dataset=upload/ids_check", token=self._login())
        self.assertTrue(all(isinstance(i["id"], str) and i["id"] for i in data["items"]))

    def test_rows_without_ids_are_skipped(self):
        path = Path(self.tmp.name) / "dataset/data/demo/noid.jsonl"
        path.write_text(json.dumps({"ace": "x"}) + "\n")
        status, _ = self._call("GET", "/api/items?dataset=demo/noid", token=self._login())
        self.assertEqual(status, 404)

    def test_review_text_is_capped(self):
        item = {"dataset": "demo/sentences", "item_id": "s1", "score": 3, "correction": "x" * 5001}
        self.assertEqual(self._call("POST", "/api/ratings", item, token=self._login())[0], 400)

    def test_large_bodies_are_refused_before_sign_in(self):
        self.assertEqual(self._raw("/api/login/start", b" " * (256 * 1024 + 1))[0], 400)
        self.assertEqual(self._raw("/api/generate", b" " * (256 * 1024 + 1))[0], 400)
        self.assertEqual(self._raw("/api/uploads", b" " * 1024)[0], 401)

    def test_handler_failures_get_a_reply(self):
        status, data = self._call("GET", "/api/items?dataset=demo/sentences&limit=abc",
                                  token=self._login())
        self.assertEqual(status, 500)
        self.assertIn("error", data)

class ReviewLimiterTests(unittest.TestCase):
    def test_limits(self):
        limit = ReviewLimiter(per_person_day=2, daily_cap=3)
        now = 1_000_000.0
        self.assertTrue(limit.check("a", now=now))
        self.assertTrue(limit.check("a", now=now))
        self.assertFalse(limit.check("a", now=now))
        self.assertTrue(limit.check("b", now=now))
        self.assertFalse(limit.check("c", now=now))
        self.assertTrue(limit.check("a", now=now + 86400))


class StubGpu(BaseHTTPRequestHandler):
    """Echoes the request so tests can see what the backend forwarded."""
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        out = json.dumps({"text": "ok", "model": "stub", "ms": 1,
                          "received": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class PublicGenerateTests(ApiTests):
    @classmethod
    def setUpClass(cls):
        cls.gpu = ThreadingHTTPServer(("127.0.0.1", 0), StubGpu)
        threading.Thread(target=cls.gpu.serve_forever, daemon=True).start()
        super().setUpClass()
        cls.cfg["gpu_url"] = f"http://127.0.0.1:{cls.gpu.server_address[1]}"
        cls.cfg["generate_per_ip"] = 3
        cls.cfg["generate_daily_cap"] = 5
        cls.server.shutdown()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(cls.cfg))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.gpu.shutdown()
        super().tearDownClass()

    def _gen(self, payload, ip, token=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/generate", data=json.dumps(payload).encode(),
            method="POST", headers={"Content-Type": "application/json", "X-Forwarded-For": ip,
                                    **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_generate_without_gpu_is_503_even_anonymously(self):
        pass  # this class has a GPU stub

    def test_anonymous_allowed_with_token_cap_and_limits(self):
        status, data = self._gen({"prompt": "Peue haba?", "max_tokens": 400}, "10.0.0.1")
        self.assertEqual(status, 200)
        self.assertEqual(data["received"]["max_tokens"], ANON_MAX_TOKENS)
        status, data = self._gen({"prompt": "Peue haba?", "max_tokens": 400}, "10.0.0.2",
                                 token=self._login("gen@example.com"))
        self.assertEqual(status, 200)
        self.assertEqual(data["received"]["max_tokens"], 400)
        status, _ = self._gen({"prompt": "  "}, "10.0.0.3")
        self.assertEqual(status, 400)
        # per-IP limit: 3 per hour (two used above by .1? no: one each) -> use a fresh IP
        for _ in range(3):
            self.assertEqual(self._gen({"prompt": "x"}, "10.0.0.9")[0], 200)
        status, data = self._gen({"prompt": "x"}, "10.0.0.9")
        self.assertEqual(status, 429)
        self.assertIn("per hour", data["error"])
        # daily cap of 5 is now reached: 1 + 1 + 3 = 5 counted calls
        status, data = self._gen({"prompt": "x"}, "10.0.0.10")
        self.assertEqual(status, 429)
        self.assertIn("daily", data["error"])
        log = storage.read_all(self.state / "generate_log.jsonl")
        self.assertEqual(len(log), 5)
        self.assertEqual(set(log[0]), {"ts", "ip_hash", "prompt_chars", "signed_in"})
        self.assertNotIn("Peue", json.dumps(log))


class StorageTests(unittest.TestCase):
    def test_latest_by(self):
        records = [{"e": "a", "i": "1", "v": 1}, {"e": "a", "i": "1", "v": 2},
                   {"e": "b", "i": "1", "v": 3}]
        latest = storage.latest_by(records, ["e", "i"])
        self.assertEqual(sorted(r["v"] for r in latest), [2, 3])


if __name__ == "__main__":
    unittest.main()
