import csv
import os
import json
import ollama
import codecs

DEBUG = True

PROMPT_SK_EXTRACTOR = """
Úloha: Z nasledujúceho textu extrahuj všetky informácie, ktoré predstavujú PODROBNÚ technickú špecifikáciu motorového vozidla alebo stroja.

"Technická špecifikácia" znamená opis viacerých technických parametrov ako:
výkon motora (kW, HP), objem motora (cm3), typ paliva, prevodovka, spotreba,
hmotnosť, rozmery (dĺžka, šírka, výška), nosnosť, kapacita, farba, emisná norma (EURO),
rok výroby, typ karosérie, pohon (napr. 4x4), počet dverí, počet miest, alebo iné merateľné technické údaje.

Úlohou je:
1. Nájsť a extrahovať všetky technické údaje, ktoré sa v texte vyskytujú.
2. V prípade, že údaje sú neúplné alebo nejednoznačné, uveď ich v poli "other".
3. Neuvádzaj žiadne komentáre ani vysvetlenia mimo JSON.

Výstup:
Vráť IBA platný JSON v nasledujúcom formáte:

{
  "brand": "",
  "model": "",
  "engine": {
    "power_kw": "",
    "power_hp": "",
    "displacement_cm3": "",
    "fuel": ""
  },
  "transmission": "",
  "consumption": "",
  "weight_kg": "",
  "dimensions_mm": {
    "length": "",
    "width": "",
    "height": ""
  },
  "capacity": "",
  "emission_standard": "",
  "drive_type": "",
  "year": "",
  "color": "",
  "features": [],
  "other": []
}

Ak žiadne údaje nie sú nájdené, vráť:
{"no_data": true}

Text:
"""

CSV_PATH = r"D:\PM\klasifikator\klas\dokumenty_new\results_chunked.csv"
TEXTS_DIR = r"D:\PM\klasifikator\klas\dokumenty_new\100txtfiles"
OUTPUT_PATH = r"D:\PM\klasifikator\klas\dokumenty_new\extracted_data.json"

results = []

with codecs.open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row.get('contains_specs', '').strip().lower()
        filename = row.get('file', '').strip()

        if label != 'yes' or not filename:
            if DEBUG:
                print(f"Skipping: {filename} (label={label})")
            continue

        candidate_path = None
        for root, _, files in os.walk(TEXTS_DIR):
            for f_name in files:
                if f_name.lower() == filename.lower():
                    candidate_path = os.path.join(root, f_name)
                    break
            if candidate_path:
                break

        if not candidate_path:
            print(f"File {filename} not found in directory {TEXTS_DIR}.")
            continue

        if DEBUG:
            print(f"Processing: {filename} ({candidate_path})")

        with open(candidate_path, encoding='utf-8') as tf:
            content = tf.read().strip()

        if len(content) < 50:
            print(f"File {filename} is empty or too short.")
            continue

        prompt = PROMPT_SK_EXTRACTOR + "\n" + content

        try:
            response = ollama.chat(
                model="gpt-oss:120b-cloud",
                messages=[
                    {"role": "system", "content": "Si odborník na automobilové technické špecifikácie."},
                    {"role": "user", "content": prompt}
                ]
            )

            if not response or "message" not in response or "content" not in response["message"]:
                print(f"No response from model for {filename}")
                continue

            raw_output = response["message"]["content"].strip()
            if DEBUG:
                print(f"Model output ({filename}): {raw_output[:200]}...")

            try:
                data = json.loads(raw_output)
            except json.JSONDecodeError:
                data = {"raw_text": raw_output}

            results.append({"filename": filename, "extracted": data})
            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print(f"\nDone! Results saved to {OUTPUT_PATH}")
print(f"Total processed files: {len(results)}")
