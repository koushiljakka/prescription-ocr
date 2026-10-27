
import unittest

from app import app, match_medicines


class TestPrescriptionVerification(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def assert_human_confirmation(self, result):
        self.assertTrue(result["requires_human_confirmation"])

    def test_exact_medicine_match(self):
        results = match_medicines(
            "Medicine: Paracetamol 500 mg tablet"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["suggested_medicine"], "Paracetamol"
        )
        self.assertEqual(
            results[0]["verification_status"], "candidate_match"
        )
        self.assert_human_confirmation(results[0])

    def test_alias_match(self):
        results = match_medicines(
            "Medicine: acetaminophen 650 mg tablet"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["suggested_medicine"], "Paracetamol"
        )
        self.assertEqual(results[0]["match_type"], "alias_match")
        self.assertEqual(
            results[0]["verification_status"], "candidate_match"
        )

    def test_strength_conflict(self):
        results = match_medicines(
            "Medicine: Paracetamol 100 mg tablet"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["strength_status"], "conflict")
        self.assertEqual(
            results[0]["verification_status"], "attribute_conflict"
        )
        self.assert_human_confirmation(results[0])

    def test_unknown_medicine(self):
        results = match_medicines(
            "Medicine: UnknownDrug 20 mg capsule"
        )
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["suggested_medicine"])
        self.assertEqual(
            results[0]["verification_status"], "unknown_medicine"
        )
        self.assert_human_confirmation(results[0])

    def test_missing_strength_and_form(self):
        results = match_medicines("Medicine: Paracetamol")
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["verification_status"],
            "incomplete_information",
        )
        self.assertEqual(results[0]["strength_status"], "not_detected")
        self.assertEqual(results[0]["form_status"], "not_detected")
        self.assert_human_confirmation(results[0])

    def test_patient_details_are_ignored(self):
        self.assertEqual(
            match_medicines("Patient: Test Patient"), []
        )

    def test_multiple_medicines(self):
        text = (
            "Medicine: Paracetamol 500 mg tablet\n"
            "Medicine: Cetirizine 10 mg tablet"
        )
        results = match_medicines(text)
        self.assertEqual(len(results), 2)

        names = {
            item["suggested_medicine"] for item in results
        }
        self.assertEqual(names, {"Paracetamol", "Cetirizine"})

        for result in results:
            self.assert_human_confirmation(result)

    def test_multiline_attributes(self):
        text = (
            "Medicine: Paracetamol\n"
            "Strength: 500 mg\n"
            "Form: Tablet"
        )
        results = match_medicines(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["detected_strengths"], ["500 mg"])
        self.assertEqual(results[0]["detected_forms"], ["tablet"])
        self.assertEqual(results[0]["strength_status"], "match")
        self.assertEqual(results[0]["form_status"], "match")
        self.assertEqual(
            results[0]["verification_status"], "candidate_match"
        )

    def test_fuzzy_medicine_candidate(self):
        results = match_medicines(
            "Medicine: Paracetemol 500 mg tablet"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["suggested_medicine"], "Paracetamol"
        )
        self.assertEqual(
            results[0]["verification_status"], "possible_match"
        )
        self.assert_human_confirmation(results[0])

    def test_unknown_text_does_not_match_catalogue(self):
        results = match_medicines(
            "Medicine: CompletelyUnknownMedicine"
        )
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["suggested_medicine"])
        self.assertEqual(
            results[0]["verification_status"], "unknown_medicine"
        )

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"], "running"
        )

    def test_homepage(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
