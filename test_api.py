
import unittest

from app import app


class TestPrescriptionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "running")

    def test_homepage(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_known_medicine_text(self):
        response = self.client.post(
            "/verify-text",
            json={"text": "Medicine: Paracetamol 500 mg tablet"},
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("matches", data)
        self.assertTrue(data["matches"])

        result = data["matches"][0]
        self.assertEqual(
            result["suggested_medicine"], "Paracetamol"
        )
        self.assertEqual(
            result["verification_status"], "candidate_match"
        )
        self.assertTrue(result["requires_human_confirmation"])

    def test_unknown_medicine_text(self):
        response = self.client.post(
            "/verify-text",
            json={"text": "Medicine: UnknownDrug 20 mg capsule"},
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("matches", data)
        self.assertTrue(data["matches"])

        result = data["matches"][0]
        self.assertIsNone(result["suggested_medicine"])
        self.assertEqual(
            result["verification_status"], "unknown_medicine"
        )
        self.assertTrue(result["requires_human_confirmation"])

    def test_missing_text_returns_400(self):
        response = self.client.post("/verify-text", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_empty_text_returns_400(self):
        response = self.client.post(
            "/verify-text",
            json={"text": "   "},
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_json_content_type(self):
        response = self.client.post(
            "/verify-text",
            data="not json",
            content_type="text/plain",
        )
        self.assertIn(response.status_code, (400, 415))


if __name__ == "__main__":
    unittest.main(verbosity=2)
