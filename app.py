
import csv
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import pytesseract
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from ocr_engine import process_prescription


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "medicine_catalog.csv"

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tesseract_path = os.environ.get("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
REQUIRED_COLUMNS = {"generic_name", "aliases", "strengths", "forms"}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "SmartPrescription Verification System",
        "ocr_engine": "Tesseract",
        "catalogue_entries": len(MEDICINE_CATALOG),
        "catalogue_file_exists": CATALOG_PATH.exists(),
        "catalogue_valid": bool(MEDICINE_CATALOG),
    }), 200


# ============================================================
# MEDICINE CATALOGUE
# ============================================================

def split_values(value):
    """Split catalogue lists on semicolons or legacy pipe separators."""
    if not value:
        return []

    separator = ";" if ";" in value else "|"
    return [item.strip() for item in value.split(separator) if item.strip()]


def load_medicine_catalog():
    medicines = []

    if not CATALOG_PATH.exists():
        logger.error("Catalogue not found: %s", CATALOG_PATH)
        return medicines

    try:
        with open(CATALOG_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                logger.error("Catalogue has no headers.")
                return []

            reader.fieldnames = [
                (field or "").strip().lower()
                for field in reader.fieldnames
            ]

            required = {"generic_name", "aliases", "strengths", "forms"}
            if not required.issubset(set(reader.fieldnames)):
                logger.error("Catalogue must contain: %s", sorted(required))
                return []

            for row in reader:
                name = (row.get("generic_name") or "").strip()
                if not name:
                    continue

                medicines.append({
                    "generic_name": name,
                    "aliases": split_values(row.get("aliases")),
                    "strengths": split_values(row.get("strengths")),
                    "forms": split_values(row.get("forms")),
                })

    except (OSError, csv.Error):
        logger.exception("Unable to read medicine catalogue.")
        return []

    logger.info("Loaded %d catalogue entries.", len(medicines))
    return medicines


MEDICINE_CATALOG = load_medicine_catalog()


def match_medicines(text):
    """Identify possible medicine candidates for human review."""
    if not isinstance(text, str) or not text.strip():
        return []

    # Split explicitly labelled medicine entries into separate records.
    entries = re.split(
        r"(?im)^\s*Medicine\s*:\s*",
        text
    )

    if len(entries) > 1:
        records = []
        for entry in entries[1:]:
            records.append(entry.strip())
    elif re.match(r"(?im)^\s*(Patient|Name|Age|Address)\s*:", text):
        return []
    else:
        records = [text.strip()]

    results = []

    for record in records:
        if not record:
            continue

        lines = [line.strip() for line in record.splitlines() if line.strip()]
        if not lines:
            continue

        first_line = lines[0]
        medicine_text = re.sub(
            r"(?i)^\s*Medicine\s*:\s*",
            "",
            first_line
        ).strip()

        # Extract strength and dosage form from the whole record.
        combined_text = " ".join(lines).lower()

        detected_strengths = []
        for strength in re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|iu|units?)"
            r"(?:\s*/\s*\d+(?:\.\d+)?\s*(?:ml|mL))?\b",
            combined_text,
            flags=re.IGNORECASE,
        ):
            normalized = re.sub(r"\s+", " ", strength.strip()).lower()
            if normalized not in detected_strengths:
                detected_strengths.append(normalized)

        detected_forms = []
        for form in (
            "tablet", "capsule", "syrup", "suspension",
            "injection", "cream", "ointment", "drops",
            "solution", "inhaler"
        ):
            if re.search(r"\b" + re.escape(form) + r"s?\b", combined_text):
                detected_forms.append(form)

        suggested = None
        match_type = "unknown"
        best_ratio = 0.0

        # Compare only the medicine-name portion, not the strength/form.
        name_to_check = re.sub(
            r"(?i)\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b",
            " ",
            medicine_text,
        )
        name_to_check = re.sub(
            r"(?i)\b(tablets?|capsules?|syrups?|suspensions?|"
            r"injections?|creams?|ointments?|drops?|solutions?|inhalers?)\b",
            " ",
            name_to_check,
        ).strip(" :,-")

        for medicine in MEDICINE_CATALOG:
            generic = medicine.get("generic_name", "").strip()
            aliases = medicine.get("aliases", [])
            candidates = [(generic, "exact_match")] + [
                (alias, "alias_match") for alias in aliases
            ]

            for candidate, candidate_type in candidates:
                if not candidate:
                    continue

                if re.search(
                    r"(?<![a-z0-9])" + re.escape(candidate.lower()) + r"(?![a-z0-9])",
                    name_to_check.lower(),
                ):
                    suggested = medicine
                    match_type = candidate_type
                    best_ratio = 1.0
                    break

                ratio = SequenceMatcher(
                    None,
                    name_to_check.lower(),
                    candidate.lower(),
                ).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    if ratio >= 0.78:
                        suggested = medicine
                        match_type = "fuzzy_match"

            if best_ratio == 1.0:
                break

        # Avoid fuzzy matching an entire prescription sentence.
        if suggested is None and best_ratio < 0.78:
            results.append({
                "suggested_medicine": None,
                "match_type": "unknown",
                "detected_strengths": detected_strengths,
                "detected_forms": detected_forms,
                "strength_status": "not_detected" if not detected_strengths else "unknown",
                "form_status": "not_detected" if not detected_forms else "unknown",
                "verification_status": "unknown_medicine",
                "requires_human_confirmation": True,
            })
            continue

        if suggested is None:
            results.append({
                "suggested_medicine": None,
                "match_type": "unknown",
                "detected_strengths": detected_strengths,
                "detected_forms": detected_forms,
                "strength_status": "not_detected" if not detected_strengths else "unknown",
                "form_status": "not_detected" if not detected_forms else "unknown",
                "verification_status": "unknown_medicine",
                "requires_human_confirmation": True,
            })
            continue

        generic_name = suggested["generic_name"]
        known_strengths = [
            re.sub(r"\s+", " ", s.strip()).lower()
            for s in suggested.get("strengths", [])
        ]
        known_forms = [
            f.strip().lower()
            for f in suggested.get("forms", [])
        ]

        if not detected_strengths:
            strength_status = "not_detected"
        elif any(s in known_strengths for s in detected_strengths):
            strength_status = "match"
        else:
            strength_status = "conflict"

        if not detected_forms:
            form_status = "not_detected"
        elif any(f in known_forms for f in detected_forms):
            form_status = "match"
        else:
            form_status = "conflict"

        if match_type == "fuzzy_match":
            verification_status = "possible_match"
        elif strength_status == "conflict" or form_status == "conflict":
            verification_status = "attribute_conflict"
        elif strength_status == "not_detected" or form_status == "not_detected":
            verification_status = "incomplete_information"
        else:
            verification_status = "candidate_match"

        results.append({
            "suggested_medicine": generic_name,
            "match_type": match_type,
            "detected_strengths": detected_strengths,
            "detected_forms": detected_forms,
            "strength_status": strength_status,
            "form_status": form_status,
            "verification_status": verification_status,
            "requires_human_confirmation": True,
        })

    return results
# ============================================================
# TEXT AND ATTRIBUTE HELPERS
# ============================================================

def normalize_text(text):
    text = str(text or "").casefold().replace("µ", "u")
    return re.sub(r"[^a-z0-9./+\-\s]", " ", text).strip()


def extract_strengths(text):
    pattern = (
        r"(?<![a-z0-9])\d+(?:\.\d+)?\s*"
        r"(?:mcg|ug|mg|g|ml|iu|units?)(?![a-z])"
    )

    found = re.findall(pattern, normalize_text(text), re.IGNORECASE)

    results = set()
    for item in found:
        item = re.sub(r"\s+", "", item.lower())
        item = item.replace("ug", "mcg")
        results.add(item)

    return results


FORM_ALIASES = {
    "tablet": ("tablet", "tablets", "tab", "tabs"),
    "capsule": ("capsule", "capsules", "cap", "caps"),
    "syrup": ("syrup",),
    "suspension": ("suspension",),
    "injection": ("injection", "injectable"),
    "cream": ("cream",),
    "ointment": ("ointment",),
    "drops": ("drop", "drops"),
    "gel": ("gel",),
    "solution": ("solution",),
    "inhaler": ("inhaler",),
}


def extract_forms(text):
    normalized = normalize_text(text)
    detected = set()

    for canonical, aliases in FORM_ALIASES.items():
        for alias in aliases:
            if re.search(
                r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])",
                normalized,
            ):
                detected.add(canonical)
                break

    return detected


def catalogue_forms(medicine):
    forms = set()

    for form in medicine["forms"]:
        normalized = normalize_text(form)

        for canonical, aliases in FORM_ALIASES.items():
            if normalized == canonical or normalized in aliases:
                forms.add(canonical)
                break
        else:
            forms.add(normalized)

    return forms


def compare_attributes(medicine, text):
    detected_strengths = extract_strengths(text)
    detected_forms = extract_forms(text)

    expected_strengths = {
        re.sub(r"\s+", "", normalize_text(value)).replace("ug", "mcg")
        for value in medicine["strengths"]
    }
    expected_forms = catalogue_forms(medicine)

    if not detected_strengths:
        strength_status = "not_detected"
    elif detected_strengths & expected_strengths:
        strength_status = "match"
    else:
        strength_status = "conflict"

    if not detected_forms:
        form_status = "not_detected"
    elif detected_forms & expected_forms:
        form_status = "match"
    else:
        form_status = "conflict"

    return {
        "strengths": medicine["strengths"],
        "forms": medicine["forms"],
        "detected_strengths": sorted(detected_strengths),
        "detected_forms": sorted(detected_forms),
        "strength_status": strength_status,
        "form_status": form_status,
    }


# ============================================================
# MEDICINE MATCHING
# ============================================================

MEDICINE_LABEL = re.compile(
    r"^\s*(?:medicine|medication|drug|rx)\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)


def find_exact_match(text, medicine):
    normalized = normalize_text(text)

    names = [
        (medicine["generic_name"], "exact_name_match"),
        *[(alias, "alias_match") for alias in medicine["aliases"]],
    ]

    names.sort(key=lambda item: len(normalize_text(item[0])), reverse=True)

    for name, match_type in names:
        normalized_name = normalize_text(name)
        if not normalized_name:
            continue

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_name)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, normalized):
            return name, match_type

    return None, None


def find_fuzzy_candidate(line):
    """
    Return a cautious candidate for human review.
    A similarity score does not prove medicine identity.
    """
    normalized_line = normalize_text(line)
    if not normalized_line:
        return None

    # Remove strengths and dosage forms before comparing names.
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mcg|ug|mg|g|ml|iu|units?)\b",
        " ",
        normalized_line,
    )

    for aliases in FORM_ALIASES.values():
        for alias in aliases:
            cleaned = re.sub(
                r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])",
                " ",
                cleaned,
            )

    ignored = {
        "medicine", "medication", "drug", "rx", "take", "daily",
        "once", "twice", "after", "before", "food", "morning",
        "night", "dose", "dosage", "patient", "prescription",
    }

    words = [word for word in cleaned.split() if word not in ignored]

    if not words:
        return None

    phrases = set()
    for size in (1, 2, 3):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            if len(phrase) >= 5:
                phrases.add(phrase)

    best = None

    for medicine in MEDICINE_CATALOG:
        for name in [medicine["generic_name"], *medicine["aliases"]]:
            normalized_name = normalize_text(name)

            if len(normalized_name) < 5:
                continue

            for phrase in phrases:
                score = SequenceMatcher(
                    None, phrase, normalized_name
                ).ratio()

                threshold = 0.88 if len(normalized_name) >= 8 else 0.92

                if score >= threshold and (
                    best is None or score > best["similarity"]
                ):
                    best = {
                        "medicine": medicine,
                        "candidate_name": name,
                        "similarity": round(score, 3),
                    }

    return best
# ============================================================
# IMAGE OCR
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "Please select a prescription image."}), 400

    extension = uploaded.filename.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported image format."}), 400

    document_type = request.form.get("document_type", "printed").lower()
    if document_type not in {"printed", "handwritten"}:
        return jsonify({"error": "Choose printed or handwritten."}), 400

    try:
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded file is empty."}), 400

        ocr_result = process_prescription(image_bytes, document_type)

        # IMPORTANT: keep text available at BOTH locations:
        # data.extracted_text and data.ocr.text.
        extracted_text = str(ocr_result.get("text") or "").strip()
        confidence = ocr_result.get("confidence", 0.0)
        word_count = ocr_result.get("word_count", 0)
        quality = ocr_result.get("quality", "unknown")
        ocr_message = ocr_result.get("message", "")

        matches = match_medicines(extracted_text)

        warnings = []
        if ocr_message:
            warnings.append(ocr_message)

        if document_type == "handwritten":
            warnings.append(
                "Handwriting recognition is experimental. Check the "
                "recognized text against the original prescription."
            )

        warnings.append(
            "Medicine suggestions are not clinical verification. "
            "A qualified person must confirm the medicine and details."
        )

        if not extracted_text:
            warnings.append(
                "No text was extracted. Enter readable text manually "
                "in the editable field and verify it."
            )

        # Preserve useful OCR metadata while ensuring predictable fields.
        ocr_metadata = dict(ocr_result)
        ocr_metadata["text"] = extracted_text
        ocr_metadata["confidence"] = confidence
        ocr_metadata["word_count"] = word_count
        ocr_metadata["quality"] = quality
        ocr_metadata["message"] = ocr_message

        return jsonify({
            "success": True,
            "text": extracted_text,
            "extracted_text": extracted_text,
            "recognized_text": extracted_text,
            "ocr_text": extracted_text,
            "confidence": confidence,
            "word_count": word_count,
            "quality": quality,
            "ocr": ocr_metadata,
            "matches": matches,
            "warnings": warnings,
            "verification_status": "manual_review_required",
            "message": (
                "OCR completed. Review the text and correct any errors."
                if extracted_text
                else "OCR returned no text. Enter or correct the text manually."
            ),
            "catalogue_entries": len(MEDICINE_CATALOG),
            "catalogue_valid": bool(MEDICINE_CATALOG),
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Prescription OCR failed.")
        return jsonify({
            "error": "Prescription processing failed. Check the Flask terminal."
        }), 500


# ============================================================
# CORRECTED TEXT VERIFICATION
# ============================================================

@app.route("/verify-text", methods=["POST"])
def verify_text():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Expected JSON containing a text field."}), 400

    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter prescription text to verify."}), 400

    matches = match_medicines(text)

    if not MEDICINE_CATALOG:
        message = "The medicine catalogue is unavailable. Check medicine_catalog.csv."
    elif matches:
        message = "Possible catalogue candidates are listed. Confirm them against the original prescription."
    else:
        message = (
            "No catalogue matches were found. This does not mean no medicine "
            "is present. Correct the text or check the catalogue."
        )

    return jsonify({
        "success": True,
        "text": text,
        "extracted_text": text,
        "matches": matches,
        "verification_status": "manual_review_required",
        "message": message,
        "catalogue_entries": len(MEDICINE_CATALOG),
        "catalogue_valid": bool(MEDICINE_CATALOG),
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({"error": "File exceeds the 8 MB upload limit."}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
