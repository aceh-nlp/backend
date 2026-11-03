"""End-to-end tests against a live handler on a temp state dir — no network, no GPU."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage  # noqa: E402
from server import (ANON_MAX_TOKENS, GenerateLimiter, build_handler, make_token,  # noqa: E402
                    read_token, row_sha256, validate_rating)

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


class RatingValidationTests(unittest.TestCase):
    ITEM = {"dataset": "d", "item_id": "i"}

    def test_specialist_score_range(self):
        for score in (0, 6, True, "4"):
            _, err = validate_rating({**self.ITEM, "score": score}, "specialist")
            self.assertIn("1..5", err)

    def test_specialist_accepts_correction_and_annotation(self):
        rec, err = validate_rating({**self.ITEM, "score": 4, "correction": "Lôn jak u keude.",
                                    "annotation": "spelling"}, "specialist")
        self.assertIsNone(err)
        self.assertEqual(rec["result"], {"score": 4, "correction": "Lôn jak u keude.",
                                         "annotation": "spelling"})

    def test_specialist_cannot_vote(self):
        _, err = validate_rating({**self.ITEM, "score": 4, "vote": 1}, "specialist")
        self.assertIn("vote", err)

    def test_everyday_vote_values(self):
        for vote in (0, 2, True, "1"):
            _, err = validate_rating({**self.ITEM, "vote": vote}, "everyday")
            self.assertIn("1 or -1", err)
        for vote in (1, -1):
            rec, err = validate_rating({**self.ITEM, "vote": vote}, "everyday")
            self.assertIsNone(err)
            self.assertEqual(rec["result"], {"vote": vote})

    def test_everyday_cannot_score_or_correct(self):
        for extra in ({"score": 3}, {"correction": "x"}, {"annotation": "x"}, {"rating": 3}):
            _, err = validate_rating({**self.ITEM, "vote": 1, **extra}, "everyday")
            self.assertIn("cannot include", err)


class ApiTests(unittest.TestCase):
    ROW_S1 = {"id": "s1", "ace_draft": "Peue haba?", "model": "secret-model"}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "dataset/data/demo").mkdir(parents=True)
        with open(root / "dataset/data/demo/sentences.jsonl", "w") as f:
            f.write(json.dumps(cls.ROW_S1) + "\n")
            f.write(json.dumps({"id": "s2", "ace_draft": "Lôn jak u keude."}) + "\n")
        cls.state = root / "state"
        cfg = {"state": str(cls.state), "dataset_root": str(root / "dataset"),
               "auth_secret": SECRET, "admin_token": ADMIN,
               "gpu_url": None, "gpu_token": None, "google_client_id": None,
               "specialists": {"spec@example.com"}}
        cls.cfg = cfg
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(cfg))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

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
        status, data = self._call("POST", "/api/login",
                                  {"email": email, "name": "Eva", "dialect": "pidie"})
        self.assertEqual(status, 200)
        return data["token"]

    def test_health_open(self):
        status, data = self._call("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])

    def test_login_rejects_bad_email(self):
        status, _ = self._call("POST", "/api/login", {"email": "nope"})
        self.assertEqual(status, 400)

    def test_items_require_auth_and_blind(self):
        status, _ = self._call("GET", "/api/items?dataset=demo/sentences")
        self.assertEqual(status, 401)
        token = self._login()
        status, data = self._call("GET", "/api/items?dataset=demo/sentences", token=token)
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 2)
        self.assertNotIn("model", data["items"][0])  # blinded

    def _export_for(self, email):
        status, data = self._call("GET", "/api/export?dataset=demo/sentences", token=ADMIN)
        self.assertEqual(status, 200)
        return [r for r in data["ratings"] if r.get("reviewer", {}).get("email") == email]

    def test_me_reports_tier(self):
        _, data = self._call("GET", "/api/me", token=self._login("spec@example.com"))
        self.assertEqual(data["tier"], "specialist")
        _, data = self._call("GET", "/api/me", token=self._login("day@example.com"))
        self.assertEqual(data["tier"], "everyday")

    def test_specialist_record_and_export(self):
        token = self._login("spec@example.com")
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
        self.assertEqual(record["data"]["dataset"], "demo/sentences")
        self.assertEqual(record["data"]["item_id"], "s1")
        self.assertEqual(record["data"]["sha256"], row_sha256(self.ROW_S1))
        self.assertEqual(record["reviewer"], {"email": "spec@example.com", "dialect": "pidie",
                                              "tier": "specialist"})
        self.assertEqual(record["result"]["score"], 2)
        self.assertEqual(record["result"]["correction"], "Peue haba, Bang?")
        self.assertEqual(record["result"]["annotation"], "polite")
        self.assertIsInstance(record["result"]["ts"], int)

    def test_everyday_vote_and_re_review_keeps_both(self):
        token = self._login("voter@example.com")
        for vote in (1, -1):
            status, _ = self._call("POST", "/api/ratings",
                                   {"dataset": "demo/sentences", "item_id": "s1", "vote": vote},
                                   token=token)
            self.assertEqual(status, 200)
        records = self._export_for("voter@example.com")
        self.assertEqual([r["result"]["vote"] for r in records], [1, -1])
        self.assertEqual(records[0]["reviewer"]["tier"], "everyday")

    def test_tier_format_enforced_over_http(self):
        everyday, specialist = self._login("wrong1@example.com"), self._login("spec@example.com")
        item = {"dataset": "demo/sentences", "item_id": "s2"}
        status, _ = self._call("POST", "/api/ratings", {**item, "score": 3}, token=everyday)
        self.assertEqual(status, 400)
        status, _ = self._call("POST", "/api/ratings", {**item, "vote": 1, "correction": "x"},
                               token=everyday)
        self.assertEqual(status, 400)
        status, _ = self._call("POST", "/api/ratings", {**item, "vote": 1}, token=specialist)
        self.assertEqual(status, 400)
        self.assertEqual(self._export_for("wrong1@example.com"), [])

    def test_unknown_item_rejected(self):
        token = self._login("lost@example.com")
        for item in ({"dataset": "demo/sentences", "item_id": "nope"},
                     {"dataset": "demo/missing", "item_id": "s1"}):
            status, _ = self._call("POST", "/api/ratings", {**item, "vote": 1}, token=token)
            self.assertEqual(status, 404)

    def test_removing_specialist_applies_to_next_review(self):
        token = self._login("former@example.com")
        item = {"dataset": "demo/sentences", "item_id": "s1"}
        self.cfg["specialists"].add("former@example.com")
        try:
            status, _ = self._call("POST", "/api/ratings", {**item, "score": 5}, token=token)
            self.assertEqual(status, 200)
        finally:
            self.cfg["specialists"].discard("former@example.com")
        status, _ = self._call("POST", "/api/ratings", {**item, "score": 5}, token=token)
        self.assertEqual(status, 400)
        status, _ = self._call("POST", "/api/ratings", {**item, "vote": 1}, token=token)
        self.assertEqual(status, 200)
        tiers = [r["reviewer"]["tier"] for r in self._export_for("former@example.com")]
        self.assertEqual(tiers, ["specialist", "everyday"])

    def test_older_records_still_count_as_done(self):
        storage.append(self.state / "ratings" / "demo_sentences.jsonl",
                       {"dataset": "demo/sentences", "item_id": "s1", "rating": 4,
                        "evaluator_email": "old@example.com", "ts": 1.0})
        token = self._login("old@example.com")
        _, data = self._call("GET", "/api/items?dataset=demo/sentences", token=token)
        self.assertEqual({i["id"] for i in data["items"]}, {"s2"})

    def test_upload_and_review(self):
        token = self._login("coord@example.com")
        status, data = self._call("POST", "/api/uploads",
                                  {"name": "my_pairs",
                                   "rows": [{"question": "Peue nyoe?", "answer": "Boh mamplam."}]},
                                  token=token)
        self.assertEqual(status, 200)
        self.assertEqual(data["dataset"], "upload/my_pairs")
        status, data = self._call("GET", "/api/items?dataset=upload/my_pairs", token=token)
        self.assertEqual(data["total"], 1)
        # duplicate name rejected
        status, _ = self._call("POST", "/api/uploads",
                               {"name": "my_pairs", "rows": [{"question": "x", "answer": "y"}]},
                               token=token)
        self.assertEqual(status, 409)

    def test_generate_without_gpu_is_503_even_anonymously(self):
        status, data = self._call("POST", "/api/generate", {"prompt": "hai"})
        self.assertEqual(status, 503)

    def test_path_escape_blocked(self):
        token = self._login()
        status, _ = self._call("GET", "/api/items?dataset=../../etc/passwd", token=token)
        self.assertEqual(status, 404)


class GenerateLimiterTests(unittest.TestCase):
    def test_per_ip_hour_and_daily_cap(self):
        lim = GenerateLimiter(per_ip_hour=2, daily_cap=3)
        t0 = 1_800_000_000
        self.assertIsNone(lim.check("a", t0))
        self.assertIsNone(lim.check("a", t0 + 1))
        self.assertIn("per hour", lim.check("a", t0 + 2))
        self.assertIsNone(lim.check("a", t0 + 3601))  # window has moved on
        self.assertIn("daily", lim.check("b", t0 + 3602))  # third call used the day
        self.assertIsNone(lim.check("b", t0 + 90_000))  # next UTC day resets


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
