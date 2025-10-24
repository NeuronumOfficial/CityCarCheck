# Klasifikátor PDF dokumentov – Technické špecifikácie vozidiel

Tento Python skript analyzuje PDF súbory a zisťuje, či obsahujú technické špecifikácie motorového vozidla.  
Podporuje textové PDF dokumenty.

---

## Popis

Program:
- Extrahuje text z bežných PDF súborov.  
- Využíva OCR na rozpoznanie textu v naskenovaných dokumentoch.  
- Na analýzu textu používa lokálny AI model (LLaMA 2 (Meta) prostredníctvom Ollama).  
- Výsledky ukladá do súboru `results.csv`.  
- Podporuje rozpoznávanie textu v slovenskom a anglickom jazyku.

---

## Princíp činnosti

1. Skript prehľadá priečinok `pdf/` a nájde všetky PDF súbory.  
2. Pre každý súbor:
   - Extrahuje text alebo ho rozpozná pomocou OCR.  
   - Odošle text lokálnemu modelu LLaMA 2 na analýzu.  
   - Určí, či dokument obsahuje technické údaje ako výkon, motor, palivo, rozmery alebo hmotnosť.  
3. Výsledky sa priebežne zapisujú do súboru `results.csv`.

---

## Poznámky

- Výsledky sa ukladajú po spracovaní každého súboru.  
- Ak text nie je možné priamo extrahovať, automaticky sa použije OCR.  
- Dlhšie dokumenty sa analyzujú po častiach (po 2000 znakov).  
- Jazyk OCR: slovenský a anglický (`slk+eng`).  
- Skript pracuje s textovými aj naskenovanými PDF dokumentmi.  
