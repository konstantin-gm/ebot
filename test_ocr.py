import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import cv2  # type: ignore
except Exception as e:
    print("OpenCV not installed. pip install -r requirements.txt", file=sys.stderr)
    raise

try:
    import pytesseract  # type: ignore
except Exception as e:
    print("pytesseract not installed. pip install -r requirements.txt", file=sys.stderr)
    raise

from bot import _ocr_number_from_image, _OCR_AVAILABLE


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OCR on a meter image")
    parser.add_argument("image", help="Path to the meter image (local file)")
    parser.add_argument("--expect", type=str, default=None, help="Expected 5-digit reading (e.g., 23653)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory to dump OCR debug images")
    args = parser.parse_args()

    if args.debug_dir:
        os.environ["EBOT_OCR_DEBUG_DIR"] = args.debug_dir

    if not _OCR_AVAILABLE:
        print("OCR is not available. Ensure Tesseract is installed and TESSERACT_CMD is set if needed.", file=sys.stderr)
        return 2

    img_path = args.image
    if not os.path.isfile(img_path):
        print(f"Image not found: {img_path}", file=sys.stderr)
        return 2

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}", file=sys.stderr)
        return 2

    # Helpful info
    try:
        print("Tesseract:", pytesseract.get_tesseract_version())
    except Exception:
        pass

    value = _ocr_number_from_image(img)
    if value is None:
        print("Detected: None")
        if args.expect is not None:
            print("FAIL: No value detected, expected", args.expect)
            return 1
        return 0

    # Normalize printing: prefer integer formatting if whole number
    value_str = str(int(value)) if float(value).is_integer() else str(value)
    print("Detected:", value_str)

    if args.expect is not None:
        if value_str == args.expect:
            print("PASS")
            return 0
        else:
            print(f"FAIL: expected {args.expect} but got {value_str}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
