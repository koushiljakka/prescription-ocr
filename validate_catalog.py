#!/usr/bin/env python3
"""Validate semicolon-separated medicine strengths and forms in medicine_catalog.csv."""

import csv
import re
import sys
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent / "medicine_catalog.csv"

SUPPORTED_FORMS = {
    "tablet", "capsule", "syrup", "suspension", "oral solution", "solution",
    "powder", "powder for oral solution", "granules", "chewable tablet",
    "effervescent tablet", "extended-release tablet", "sublingual tablet",
    "cream", "ointment", "gel", "lotion", "topical solution", "shampoo",
    "injection", "inhaler", "nasal spray", "nebulizer suspension",
    "eye drops", "ear drops", "oral drops", "drops", "suppository", "patch",
    "lozenge", "mouthwash", "spray", "inhalation solution", "vial", "ampoule",
}

# Examples: 500 mg, 0.1%, 100 mg/5 mL, 20 mcg/actuation, 100 units/mL.
STRENGTH_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:mcg|μg|ug|mg|g|kg|iu|units?|%|mmol|mol|meq)"
    r"(?:\s*/\s*(?:\d+(?:\.\d+)?\s*)?"
    r"(?:mcg|μg|ug|mg|g|kg|ml|l|units?|actuation|dose|puff|tablet|capsule))?\s*$",
    re.IGNORECASE,
)


def split_values(value):
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def valid_strength(value):
    value = value.strip()
    if STRENGTH_RE.fullmatch(value):
        return True
    # Combination strengths such as 80 mg/480 mg or 2.25 g/0.5 g.
    parts = re.split(r"\s*/\s*(?=\d)", value)
    return len(parts) > 1 and all(STRENGTH_RE.fullmatch(part) for part in parts)


def validate_catalog(path=CATALOG_PATH):
    errors = []
    warnings = []
    records = []

    if not path.is_file():
        print(f"[ERROR] Catalogue file not found: {path}")
        return 1

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"generic_name", "aliases", "strengths", "forms"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                print("[ERROR] Missing required CSV columns: " + ", ".join(sorted(missing)))
                return 1

            for row_number, row in enumerate(reader, start=2):
                name = (row.get("generic_name") or "").strip()
                if not name:
                    errors.append(f"Row {row_number}: missing generic_name")
                    continue

                records.append(row)
                strengths = split_values(row.get("strengths"))
                forms = split_values(row.get("forms"))

                if not strengths:
                    errors.append(f"Row {row_number} ({name}): no strengths provided")
                for strength in strengths:
                    if not valid_strength(strength):
                        errors.append(
                            f"Row {row_number} ({name}): malformed strength "
                            f"'{strength}'. Example: 500 mg."
                        )

                if not forms:
                    errors.append(f"Row {row_number} ({name}): no forms provided")
                for form in forms:
                    if form.casefold() not in SUPPORTED_FORMS:
                        errors.append(
                            f"Row {row_number} ({name}): unsupported form "
                            f"'{form}'. Add it to SUPPORTED_FORMS if appropriate."
                        )

                if not (row.get("aliases") or "").strip():
                    warnings.append(f"Row {row_number} ({name}): no aliases provided")
    except (OSError, csv.Error) as exc:
        print(f"[ERROR] Could not read catalogue: {exc}")
        return 1

    print("=" * 60)
    print("SMART PRESCRIPTION VERIFICATION")
    print("MEDICINE CATALOGUE VALIDATION")
    print("=" * 60)
    print(f"File: {path}")
    print(f"\nMedicine records: {len(records)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nERRORS")
        print("-" * 60)
        for error in errors:
            print(f"[ERROR] {error}")
    if warnings:
        print("\nWARNINGS")
        print("-" * 60)
        for warning in warnings:
            print(f"[WARNING] {warning}")

    print()
    if errors:
        print("RESULT: FAILED — fix the errors before using this catalogue.")
        return 1
    print("RESULT: PASSED — catalogue format and supported forms are valid.")
    print("Note: format validation does not confirm clinical correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(validate_catalog())
