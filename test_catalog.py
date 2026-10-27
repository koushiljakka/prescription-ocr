
import csv
import tempfile
import unittest
from pathlib import Path

from validate_catalog import validate_catalogue


VALID_CSV = """generic_name,aliases,strengths,forms
Paracetamol,acetaminophen,500 mg|650 mg,tablet|syrup
Cetirizine,zyrtec,10 mg,tablet
Amoxicillin,amoxil,250 mg|500 mg,capsule|tablet
Ibuprofen,advil,200 mg|400 mg,tablet
"""


class TestMedicineCatalogue(unittest.TestCase):

    def validate_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "medicine_catalog.csv"
            path.write_text(text, encoding="utf-8")
            return validate_catalogue(path)

    def test_current_catalogue_is_valid(self):
        result = validate_catalogue(
            Path(__file__).resolve().parent / "medicine_catalog.csv"
        )
        self.assertEqual(result["errors"], [])

    def test_valid_catalogue_fixture(self):
        result = self.validate_text(VALID_CSV)
        self.assertEqual(result["errors"], [])

    def test_duplicate_generic_names_are_detected(self):
        text = VALID_CSV + (
            "Paracetamol,para,100 mg,tablet\n"
        )
        result = self.validate_text(text)
        self.assertTrue(result["errors"])

    def test_duplicate_alias_conflict_is_detected(self):
        text = VALID_CSV + (
            "ExampleDrug,acetaminophen,20 mg,tablet\n"
        )
        result = self.validate_text(text)
        self.assertTrue(result["errors"])

    def test_missing_required_header_is_detected(self):
        text = """generic_name,aliases,strengths
Paracetamol,acetaminophen,500 mg
"""
        result = self.validate_text(text)
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
