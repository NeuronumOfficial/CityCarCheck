import os
import re
import json
import pandas as pd
import ollama

# ============== SETTINGS ==============
TEXT_FOLDER = r"D:\PM\klasifikator\klas\dokumenty_new\100txtfiles"
CSV_FILE = r"D:\PM\klasifikator\klas\dokumenty_new\results_chunked.csv"
MODEL_NAME = "gpt-oss:120b-cloud"
CHUNK_SIZE = 4000     # characters per chunk
STEP = 3000           # overlapping step (to not lose context)
DEBUG = True


# ============== PROMPT ==============
PROMPT_SK = """
Úloha: Urči, či nasledujúci text obsahuje PODROBNÚ technickú špecifikáciu motorového vozidla alebo stroja.

🔹 "Technická špecifikácia" znamená opis viacerých technických parametrov ako:
   výkon motora (kW, HP), objem motora (cm3), typ paliva, prevodovka, spotreba,
   hmotnosť, rozmery (dĺžka, šírka, výška), nosnosť, kapacita, farba, emisná norma (EURO),
   alebo iné merateľné technické údaje.

🔹 Text sa považuje za **technickú špecifikáciu (yes)** len vtedy, ak obsahuje
   viacero takýchto parametrov (minimálne dva rôzne technické údaje).

🔹 Ak text obsahuje iba základné identifikačné informácie ako:
   značka (napr. Škoda, Peugeot), model (Octavia, Boxer), VIN číslo,
   číslo motora, evidenčné číslo, alebo len jeden technický údaj (napr. objem motora),
   považuj ho za **"no"**.

🔹 Ak text obsahuje iba administratívne, finančné, právne alebo komunikačné údaje
   (napr. faktúra, cena, zmluva, objednávka, e-mailová komunikácia),
   tiež odpovedz **"no"**.

🔹 Ak si nie si istý, odpovedz **"no"**.

Výstup:
Vráť IBA platný JSON vo formáte:
{"answer":"yes|no","evidence":["param1","param2"],"confidence":0..1}

Text:
"""


# ============== HELPER FUNCTIONS ==============
def clean_text(text: str) -> str:
    """Normalize text and remove unwanted spacing."""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def read_text(path: str) -> str:
    """Safely read a UTF-8 text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return clean_text(f.read())
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 4000, step: int = 3000):
    """Split long text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks


# ============== MODEL CALL ==============
def ask_model(text: str):
    """Ask Ollama model and parse its JSON output."""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content":
                    "You are a strict JSON classifier. Respond only valid JSON with keys: "
                    '{"answer":"yes|no","evidence":["..."],"confidence":0-1}. '
                    "If uncertain, answer 'no'."},
                {"role": "user", "content": PROMPT_SK + text},
            ],
        )
        raw = response["message"]["content"]
        if DEBUG:
            print(f"🔍 Raw output: {raw[:300]}...")

        # Extract JSON safely
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response.")
        data = json.loads(match.group(0))

        ans = data.get("answer", "").strip().lower()
        conf = float(data.get("confidence", 0) or 0)
        evidence = data.get("evidence", [])
        return ans, evidence, conf

    except Exception as e:
        print(f"⚠️ Model error: {e}")
        return "no", [], 0.0


# ============== CLASSIFICATION LOGIC ==============
def classify_document(text: str):
    """Analyze text by chunks. Return Yes if any chunk is technical."""
    chunks = chunk_text(text, CHUNK_SIZE, STEP)
    print(f"📄 Document has {len(chunks)} chunks to analyze.")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"   🔹 Chunk {idx}/{len(chunks)} → sending to model...")
        ans, evidence, conf = ask_model(chunk)
        if ans == "yes" and conf >= 0.5:
            print(f"   ✅ Chunk {idx}: YES (conf={conf:.2f}, evidence={evidence})")
            return "Yes"
        else:
            print(f"   ❌ Chunk {idx}: NO (conf={conf:.2f})")
    return "No"


# ============== MAIN LOOP ==============
def main():
    if os.path.exists(CSV_FILE):
        try:
            df_old = pd.read_csv(CSV_FILE)
            processed = set(df_old["file"])
            print(f"📘 Loaded CSV: {len(processed)} already processed.")
        except Exception:
            print("⚠️ CSV corrupted → starting new.")
            df_old = pd.DataFrame(columns=["file", "contains_specs"])
            processed = set()
    else:
        df_old = pd.DataFrame(columns=["file", "contains_specs"])
        processed = set()

    for file in os.listdir(TEXT_FOLDER):
        if not file.lower().endswith(".txt"):
            continue
        if file in processed:
            print(f"⏩ Skipping {file} (already processed)")
            continue

        path = os.path.join(TEXT_FOLDER, file)
        text = read_text(path)
        if not text or len(text) < 50:
            print(f"⚠️ {file} is empty or too short → skipped.")
            result = "No"
        else:
            result = classify_document(text)

        # Save immediately after each file
        pd.DataFrame([{"file": file, "contains_specs": result}]).to_csv(
            CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0,
            index=False, encoding="utf-8-sig"
        )
        print(f"💾 Saved: {file} → {result}")

    print("\n✅ All files processed. Results saved to CSV.")


if __name__ == "__main__":
    main()
