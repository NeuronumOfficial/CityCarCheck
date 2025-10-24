import os
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pandas as pd
import ollama

# === SETTINGS ===
PDF_FOLDER = "pdf"        # Folder with PDF files
CSV_FILE = "results.csv"   # Output CSV file
MODEL_NAME = "llama2"      # Ollama model

# === FUNCTIONS ===

def extract_text_from_pdf(path):
    """Extracts text from PDF (if not a scan)"""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f" Error reading {path}: {e}")
        return ""

def extract_text_ocr(path): # Optional
    """Recognizes text via OCR (for scanned PDFs)"""
    try:
        pages = convert_from_path(path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page, lang="slk+eng") + "\n"
        return text.strip()
    except Exception as e:
        print(f" OCR error while processing {path}: {e}")
        return ""

def ask_llama(prompt):
    """Helper function for querying the model"""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a document classifier."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        print(f" LLaMA error: {e}")
        return ""

def analyze_full_text(text):
    """Checks the entire document"""
    # prompt = f"""
    # Determine whether the following text contains a technical specification 
    # of a motor vehicle or machine – for example, data such as power (kW), engine, fuel, 
    # transmission, dimensions, weight, load capacity, engine displacement, body type, etc.

    # If the text contains at least two such technical values, answer “Yes”. 
    # Otherwise, answer “No”.

    # Text:
    # {text}
    # """

    prompt = f"""
    Urči, či nasledujúci text obsahuje technickú špecifikáciu 
    motorového vozidla alebo stroja – napríklad údaje ako výkon (kW), motor, palivo, 
    prevodovka, rozmery, hmotnosť, nosnosť, objem motora, karoséria atď.

    Ak text obsahuje aspoň dve takéto technické hodnoty, odpovedz „Áno“. 
    Inak odpovedz „Nie“.

    Text:
    {text}
    """
    
    answer = ask_llama(prompt)
    return "Yes" if any(x in answer for x in ["áno", "yes", "да", "ano"]) else "No"

def analyze_chunk(chunk, i):
    """Checks one fragment of a long document"""
    # prompt = f"""
    # Determine whether the following text contains a technical specification 
    # of a motor vehicle or machine – for example, data such as power (kW), engine, fuel, 
    # transmission, dimensions, weight, load capacity, engine displacement, body type, etc.

    # If the text contains at least two such technical values, answer “Yes”. 
    # Otherwise, answer “No”.

    # Document part no.{i}:


    
    # {chunk}
    # """
    
    prompt = f"""
    Urči, či nasledujúci text obsahuje technickú špecifikáciu 
    motorového vozidla alebo stroja – napríklad údaje ako výkon (kW), motor, palivo, 
    prevodovka, rozmery, hmotnosť, nosnosť, objem motora, karoséria atď.

    Ak text obsahuje aspoň dve takéto technické hodnoty, odpovedz „Áno“. 
    Inak odpovedz „Nie“.

    Časť dokumentu č.{i}:
    {chunk}
    """
    answer = ask_llama(prompt)
    if any(x in answer for x in ["áno", "yes", "да", "ano"]):
        return True
    return False

def process_pdf(file_path):
    """Main logic for processing a single PDF"""
    print(f"\n Processing: {file_path}")
    text = extract_text_from_pdf(file_path)
    if len(text) < 200:
        text = extract_text_ocr(file_path)

    if not text:
        print("Empty or corrupted file.")
        return "Error"

    # --- Automatic decision: whole or chunked ---
    if len(text) <= 12000:
        result = analyze_full_text(text)
        print(f"  Result: {result}")
        return result
    else:
        print(f"Long document ({len(text)} characters) — analyzing by parts...")
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for i, chunk in enumerate(chunks, start=1):
            if analyze_chunk(chunk, i):
                print(f"  Specification found in part {i} of {file_path}")
                print(f"  Result: Yes")
                return "Yes"
        print("  Specification not found")
        return "No"

# === MAIN LOOP ===
def main():
    # Load previous results
    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE)
        processed = set(df_old["file"])
        print(f" Found {CSV_FILE}, already processed: {len(processed)} files.")
    else:
        df_old = pd.DataFrame(columns=["file", "contains_specs"])
        processed = set()
        print("CSV not found — creating a new one.")
        df_old.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

    # Iterate over PDF folder
    for file in os.listdir(PDF_FOLDER):
        if not file.lower().endswith(".pdf"):
            continue
        if file in processed:
            print(f" Skipping: {file} (already processed)")
            continue

        path = os.path.join(PDF_FOLDER, file)
        result = process_pdf(path)

        # Save result immediately after processing each file
        new_row = pd.DataFrame([{"file": file, "contains_specs": result}])
        new_row.to_csv(CSV_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
        print(f"Saved result for: {file}")

    print("\nProcessing complete. All results saved to CSV.")

if __name__ == "__main__":
    main()