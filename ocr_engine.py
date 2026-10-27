import os
import re
import shutil

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def configure_tesseract():
    """Find Tesseract automatically on Windows."""
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    return False


TESSERACT_AVAILABLE = configure_tesseract()


def decode_image(image_bytes):
    """Decode and resize uploaded prescription images."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image. Upload JPG or PNG.")

    height, width = image.shape[:2]

    if height < 80 or width < 80:
        raise ValueError("Image is too small. Upload a clearer image.")

    max_dimension = 2800
    scale = min(1.0, max_dimension / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    return image


def deskew_image(gray):
    """Correct small rotations in the prescription."""
    inverted = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    coords = np.column_stack(np.where(inverted > 0))

    if len(coords) < 100:
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3 or abs(angle) > 10:
        return gray

    height, width = gray.shape

    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def crop_medicine_region(image, top=0.30, bottom=0.95):
    """
    Return a crop of the lower prescription area for inspection.
    The crop is only a starting point; adjust it to the actual layout.
    """
    if image is None or image.size == 0:
        raise ValueError("A valid image is required.")

    height, width = image.shape[:2]

    y1 = max(0, min(height - 1, int(height * top)))
    y2 = max(y1 + 1, min(height, int(height * bottom)))

    return image[y1:y2, 0:width].copy()


def prepare_variants(image, document_type="printed"):
    """Prepare different image versions for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if min(gray.shape[:2]) < 1400:
        gray = cv2.resize(
            gray,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = deskew_image(gray)

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=5 if document_type == "handwritten" else 8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(denoised)

    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)

    sharpened = cv2.addWeighted(
        clahe,
        1.2,
        blurred,
        -0.2,
        0,
    )

    variants = [
        gray,
        denoised,
        clahe,
        sharpened,
    ]

    if document_type == "printed":
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

        otsu = cv2.threshold(
            clahe,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        variants.extend([adaptive, otsu])

    prepared = []

    for variant in variants:
        bordered = cv2.copyMakeBorder(
            variant,
            24,
            24,
            24,
            24,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        prepared.append(bordered)

    return prepared


def assess_text_plausibility(text):
    """Estimate whether OCR output looks obviously corrupted."""
    compact = re.sub(r"\s+", "", text)

    if not compact:
        return {
            "alnum_ratio": 0.0,
            "word_ratio": 0.0,
            "garbage_penalty": 40.0,
        }

    alphanumeric = sum(
        character.isalnum() for character in compact
    )

    alnum_ratio = alphanumeric / len(compact)

    words = re.findall(r"[A-Za-z0-9]+", text)

    alphabetic_words = [
        word.lower()
        for word in words
        if any(character.isalpha() for character in word)
    ]

    vowel_words = sum(
        any(vowel in word for vowel in "aeiouy")
        for word in alphabetic_words
    )

    word_ratio = vowel_words / max(len(alphabetic_words), 1)

    penalty = 0.0

    if alnum_ratio < 0.70:
        penalty += (0.70 - alnum_ratio) * 35

    if alphabetic_words and word_ratio < 0.45:
        penalty += (0.45 - word_ratio) * 30

    # Long consonant sequences can indicate corrupted text.
    consonant_sequences = re.findall(
        r"(?i)[bcdfghjklmnpqrstvwxz]{5,}",
        text,
    )

    letter_count = max(
        sum(character.isalpha() for character in compact),
        1,
    )

    consonant_count = sum(
        len(sequence) for sequence in consonant_sequences
    )

    consonant_ratio = consonant_count / letter_count

    if consonant_ratio > 0.12:
        penalty += min(
            20.0,
            (consonant_ratio - 0.12) * 80,
        )

    if len(words) < 2:
        penalty += 8.0

    return {
        "alnum_ratio": round(alnum_ratio, 3),
        "word_ratio": round(word_ratio, 3),
        "garbage_penalty": round(penalty, 2),
    }



def extract_candidate(variant, psm):
    """Extract OCR text and score its readability and OCR confidence."""
    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        variant,
        config=config,
        lang="eng",
        output_type=Output.DICT,
    )

    words = []
    confidences = []

    for raw_text, raw_confidence in zip(
        data["text"],
        data["conf"],
    ):
        word = raw_text.strip()

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue

        if word and confidence >= 0:
            words.append(word)
            confidences.append(confidence)

    text = " ".join(words).strip()

    if not text:
        return None

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    high_confidence_ratio = (
        sum(confidence >= 50 for confidence in confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    plausibility = assess_text_plausibility(text)

    score = (
        mean_confidence * 0.55
        + high_confidence_ratio * 15
        + min(len(words), 35) * 0.20
        + plausibility["alnum_ratio"] * 6
        + plausibility["word_ratio"] * 3
        - plausibility["garbage_penalty"]
    )

    return {
        "text": text,
        "confidence": round(mean_confidence, 2),
        "high_confidence_ratio": round(high_confidence_ratio, 3),
        "word_count": len(words),
        "psm": psm,
        "score": round(score, 2),
        **plausibility,
    }


def determine_quality(candidate, document_type="printed"):
    """Estimate OCR output quality; this is not a measure of medical accuracy."""
    text = candidate.get("text", "").strip()
    confidence = float(candidate.get("confidence", 0.0))
    word_count = int(candidate.get("word_count", 0))

    if not text or word_count == 0:
        return (
            "unreadable",
            "No usable text was recognized. Try a clearer image."
        )

    if document_type == "handwritten":
        if confidence >= 70 and word_count >= 3:
            return (
                "review",
                "Some text was recognized, but handwriting requires manual review."
            )
        return (
            "low",
            "Handwriting recognition is uncertain. Manually check the original prescription."
        )

    if confidence >= 80 and word_count >= 3:
        return (
            "good",
            "Text recognition looks reasonably clear. Verify it against the original image."
        )

    if confidence >= 50:
        return (
            "moderate",
            "Some text was recognized. Review unclear words against the original."
        )

    return (
        "low",
        "OCR confidence is low. Try a clearer image and review the result manually."
    )


def recognize_text(image, document_type="printed"):
    """Recognize printed or experimental handwritten text."""
    if document_type not in {"printed", "handwritten"}:
        raise ValueError(
            "document_type must be 'printed' or 'handwritten'."
        )

    if not TESSERACT_AVAILABLE:
        return {
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "quality": "unreadable",
            "message": (
                "Tesseract was not found. Install Tesseract or set "
                "TESSERACT_CMD to the full executable path."
            ),
        }

    variants = prepare_variants(image, document_type)

    if document_type == "handwritten":
        psm_modes = [6, 11, 12]
    else:
        psm_modes = [6, 11, 3]

    candidates = []

    for variant_index, variant in enumerate(
        variants,
        start=1,
    ):
        for psm in psm_modes:
            try:
                candidate = extract_candidate(variant, psm)
            except (
                pytesseract.TesseractError,
                OSError,
            ):
                continue

            if candidate:
                candidate["variant"] = variant_index
                candidates.append(candidate)

    if not candidates:
        return {
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "quality": "unreadable",
            "message": (
                "No text could be recognized. Upload a clearer "
                "image or manually review the prescription."
            ),
        }

    best = max(
        candidates,
        key=lambda candidate: candidate["score"],
    )

    quality, message = determine_quality(
        best,
        document_type,
    )

    if document_type == "handwritten":
        message += (
            " Tesseract is not a handwriting-specific model. "
            "Never rely on this output alone for medication decisions."
        )

    return {
        "text": best["text"],
        "confidence": best["confidence"],
        "word_count": best["word_count"],
        "quality": quality,
        "message": message,
        "selected_psm": best["psm"],
        "selected_variant": best["variant"],
        "candidate_score": best["score"],
        "text_plausibility": {
            "alnum_ratio": best["alnum_ratio"],
            "word_ratio": best["word_ratio"],
        },
        "engine_note": (
            "OCR confidence is not the same as recognition accuracy."
        ),
    }


def process_prescription(
    image_bytes,
    document_type="printed",
):
    """Public interface used by app.py."""
    image = decode_image(image_bytes)

    return recognize_text(
        image,
        document_type,
    )
