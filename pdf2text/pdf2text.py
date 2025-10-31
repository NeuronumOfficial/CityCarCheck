import os
import re
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

# === SETTINGS ===
PDF_FOLDER = r"D:\PM\klasifikator\klas\dokumenty_new\dokumenty"
OUTPUT_FOLDER = r"D:\PM\klasifikator\klas\dokumenty_new\text"
LANG = "slk+eng"
DPI = 300
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === INITIALIZATION ===
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    """Cleans OCR output"""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text_from_pdf(path: str) -> str:
    """Extracts text from normal (non-scanned) PDF"""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""


def extract_text_ocr(path: str) -> str:
    """Runs OCR on scanned PDF using Tesseract"""
    try:
        pages = convert_from_path(path, dpi=DPI, poppler_path=POPPLER_PATH)
        text = ""
        for i, page in enumerate(pages, start=1):
            print(f"  Performing OCR on page {i}/{len(pages)}...")
            txt = pytesseract.image_to_string(
                page.convert("L"),
                lang=LANG,
                config="--oem 3 --psm 6"
            )
            text += txt + "\n"
        text = clean_text(text)
        print(f"  OCR extracted {len(text)} characters.")
        return text
    except Exception as e:
        print(f"OCR error while processing {path}: {e}")
        return ""


def process_pdf(file_path: str, output_path: str):
    """Main processing logic"""
    print(f"\nProcessing: {file_path}")

    # Step 1: Try normal text extraction
    text = extract_text_from_pdf(file_path)

    # Step 2: If not enough text, run OCR
    if len(text) < 100:
        print("  Looks like a scanned file - starting OCR...")
        text = extract_text_ocr(file_path)

    # Step 3: Always create a .txt file, even if empty
    if not text:
        print("  Warning: No text extracted, creating empty file.")
        text = ""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved text to: {output_path}")
    except Exception as e:
        print(f"  Error saving file {output_path}: {e}")


def main():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.\n")

    for file in pdf_files:
        input_path = os.path.join(PDF_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(file)[0] + ".txt")
        process_pdf(input_path, output_path)

    print("\nAll PDF files processed. Results saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
