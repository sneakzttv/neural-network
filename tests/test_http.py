"""Real loopback HTTP tests ensure API errors are visible, not silent failures."""

import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import unittest

from app import Handler, TRAINING_LOCK


class QuietHandler(Handler):
    def log_message(self, *args):
        # Expected 400/403 responses should not clutter the test report.
        pass


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Port zero asks the OS for a free loopback port; never bind publicly.
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def post(self, body):
        return self.request("POST", "/api/train", body, {"Content-Type": "application/json"})

    def test_health_and_all_assets(self):
        status, body, _ = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        for path in ("/", "/style.css", "/app.js"):
            status, body, headers = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertGreater(len(body), 100)
            self.assertIn("Content-Security-Policy", headers)

    def test_short_real_training_request(self):
        status, body, _ = self.post('{"epochs": 2, "samples": 80}')
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["history"][-1]["epoch"], 2)
        self.assertEqual(result["train_count"], 60)

    def test_invalid_payloads_and_routes(self):
        for body in ('not json', '[]', '{"epochs": false}', '{"learning_rate": NaN}'):
            status, response, _ = self.post(body)
            self.assertEqual(status, 400)
            self.assertIn("error", json.loads(response))
        self.assertEqual(self.post(" " * 8193)[0], 413)
        self.assertEqual(self.request("POST", "/api/train", "{}")[0], 415)
        self.assertEqual(self.request("GET", "/../network.py")[0], 404)
        self.assertEqual(self.request("POST", "/missing", "{}")[0], 404)

    def test_rebinding_and_cross_origin_are_blocked(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Origin": "https://evil.example"})[0], 403)

    def test_concurrent_training_surfaces_busy_error(self):
        # Occupy the gate without a second expensive training process.
        TRAINING_LOCK.acquire()
        try:
            status, body, _ = self.post("{}")
            self.assertEqual(status, 429)
            self.assertIn("progress", json.loads(body)["error"])
        finally:
            TRAINING_LOCK.release()


if __name__ == "__main__":
    unittest.main()
