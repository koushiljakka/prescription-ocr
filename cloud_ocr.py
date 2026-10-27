
import os
import requests


VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def recognize_handwriting_online(image_bytes):
    """
    Return an OCR result when Google Cloud Vision is configured.
    Return None when unavailable so the caller can use Tesseract.
    """
    api_key = os.environ.get("GOOGLE_VISION_API_KEY")

    if not api_key:
        return None

    try:
        import base64

        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        response = requests.post(
            VISION_URL,
            params={"key": api_key},
            json={
                "requests": [
                    {
                        "image": {"content": encoded_image},
                        "features": [
                            {"type": "DOCUMENT_TEXT_DETECTION"}
                        ],
                    }
                ]
            },
            timeout=12,
        )

        response.raise_for_status()
        payload = response.json()
        result = payload["responses"][0]

        if result.get("error"):
            return None

        text = (
            result.get("fullTextAnnotation", {}).get("text", "")
            .strip()
        )

        if not text:
            return None

        confidences = []

        for page in result.get("fullTextAnnotation", {}).get("pages", []):
            for block in page.get("blocks", []):
                for paragraph in block.get("paragraphs", []):
                    for word in paragraph.get("words", []):
                        confidence = word.get("confidence")
                        if isinstance(confidence, (int, float)):
                            confidences.append(confidence * 100)

        confidence = (
            round(sum(confidences) / len(confidences), 2)
            if confidences else 0
        )

        return {
            "text": text,
            "confidence": confidence,
            "word_count": len(text.split()),
            "quality": (
                "readable" if confidence >= 75
                else "review"
            ),
            "message": (
                "Online OCR completed. Verify every medicine name, "
                "strength and instruction against the original."
            ),
            "selected_psm": "google-cloud-vision",
            "selected_variant": 0,
        }

    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
