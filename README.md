```python

import pandas as pd
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import csv
from tqdm import tqdm

# --- CONFIGURATION CONSTANTS ---

# Adjust the number of parallel requests to send to your vLLM instance
MAX_CONCURRENT_REQUESTS = 512

# Adjust the number of rows to read from the CSV and process in each batch
CHUNK_SIZE = 1024

BASE_BIG_SPACE = "/ephemeral/private_jurisprudence/"
# File paths
INPUT_CSV_PATH = Path(BASE_BIG_SPACE+"new_chunked_smart_correct.csv")  # <--- IMPORTANT: SET YOUR INPUT FILENAME
OUTPUT_CSV_PATH = Path(BASE_BIG_SPACE+"output_with_analysis.csv")

# API and Model details
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# --- END OF CONFIGURATION ---


# Increase the field size limit for CSV parsing, as some 'part_content' rows might be very long
# This is a good safety measure for large, complex CSVs.
try:
    max_int = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
except Exception as e:
    print(f"Could not set CSV field size limit: {e}")


# Load the system prompt from your description
# Note: The user prompt is now loaded from the specified file `prompt_extract_citations.md`
# This script will now use the content of each row as the user prompt.
SYSTEM_PROMPT = """
Vous êtes un expert en droit suisse.

DONNEZ TOUTES LES CITATIONS DANS UN FORMAT JSON : {"doctrine": [], "jurisprudence":[], "articles de loi":[]} LES ELEMENTS DANS LES LISTES DOIVENT ÊTRES DES SUBSTRINGS EXACTS DU TEXTE, PAS DE REFORMULATIONS. LES CITATIONS DOIVENT ÊTRES COMPLÈTES, PAR EXEMPLE "47 al. 1 let. f" N'EST PAS COMPLET COMME IL N'Y A PAS LA LOI. donc si par exemple il y a "47 al. 1 let. f et 50 al. 2 CPC" dans le texte, il faut donner "47 al. 1 let. f et 50 al. 2 CPC" en entier comme citations d'articles de loi.

DANS LES CAS où il y a des mots comme "précité", il faut bien donner la citation en entier, par exemple "(arrêt du TF du 11.10.2018 précité, cons. 1.2.2)" donne la citation "arrêt du TF du 11.10.2018 précité, cons. 1.2.2"

ATTENTION SI DANS LE TEXTE ILS DISENT "article 259e CO" VOTRE CITATION DOIT ÊTRE EXACTEMENT "article 259e CO" pas "art. 259e CO". ON NE REFORMULE PAS. TOUTES LES RÉPONSES DOIVENT ÊTRE UN SUBSTRING EXACT DU TEXTE DE BASE. SI IL Y A DANS LE TEXTE "art. 32 CO" alors on cite "art. 32 CO" si il y a dans le texte "article 260b CPC" alors on cite "article 260b CPC".

Les citations de doctrine c'est quand un livre écrit par un juriste est cité, par exemple : "Bohnet, Code de procédure civile commenté, op. cit., n. 23 ad art. 47 CPC et les références" c'est une citation de doctrine.

NE DONNEZ QUE DES CITATIONS QUI SONT DANS LE TEXTE QUE JE VOUS DONNE. LES EXEMPLES DONNÉES DANS LES INSTRUCTIONS NE FONT PAS PARTI DU TEXTE À ANALYSER. VERIFIEZ BIEN QUE VOUS CITES DES SUBSTRINGS EXACTS (PAR EXEMPLE C'EST DÉLICAT article vs art. il ne faut pas reformuler)
"""

def analyze_text(text_to_analyze: str) -> str:
    """
    Sends a single text snippet to the LLM API for analysis and returns the result.
    Handles potential errors and returns an error message string if the request fails.
    """
    # If the input text is empty or not a string, return an empty JSON object
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return json.dumps({"error": "Input text was empty or invalid."})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_to_analyze},
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120) # 2-minute timeout
        response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
        content = response.json()["choices"][0]["message"]["content"].replace("\n", "")
        return content
    except requests.exceptions.RequestException as e:
        error_message = f"API Request Error: {e}"
        print(f"\n{error_message}\n")
        return json.dumps({"error": error_message})
    except (KeyError, IndexError) as e:
        error_message = f"Invalid API Response format: {e} - Response: {response.text}"
        print(f"\n{error_message}\n")
        return json.dumps({"error": error_message})


def main():
    """
    Main function to read the CSV in chunks, process each chunk with concurrency,
    and write the results to a new CSV file.
    """
    print(f"Starting analysis of '{INPUT_CSV_PATH}'...")
    print(f"Reading in chunks of {CHUNK_SIZE} rows.")
    print(f"Using up to {MAX_CONCURRENT_REQUESTS} concurrent requests.")
    print(f"Output will be saved to '{OUTPUT_CSV_PATH}'.")

    is_first_chunk = True

    try:
        # Create a pandas TextFileReader to read the CSV in chunks
        reader = pd.read_csv(INPUT_CSV_PATH, chunksize=CHUNK_SIZE, on_bad_lines='warn')

        # Process each chunk
        for i, chunk_df in enumerate(reader):
            print(f"\n--- Processing chunk {i + 1} ---")
            
            # Get the list of texts to analyze from the 'part_content' column
            texts_to_process = chunk_df['part_content'].tolist()
            
            # Use ThreadPoolExecutor to make API calls concurrently
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                # executor.map applies the function to each item in the list and returns results in order
                # tqdm adds a progress bar
                results = list(tqdm(
                    executor.map(analyze_text, texts_to_process),
                    total=len(texts_to_process),
                    desc=f"Analyzing chunk {i + 1}"
                ))
            
            # Add the results as a new 'analysis' column to the DataFrame chunk
            chunk_df['analysis'] = results

            # --- Write the processed chunk to the output file ---
            if is_first_chunk:
                # For the first chunk, create the file and write the header
                chunk_df.to_csv(OUTPUT_CSV_PATH, index=False, mode='w', quoting=csv.QUOTE_MINIMAL)
                is_first_chunk = False
                print(f"Created '{OUTPUT_CSV_PATH}' and wrote the first chunk.")
            else:
                # For subsequent chunks, append to the file without the header
                chunk_df.to_csv(OUTPUT_CSV_PATH, index=False, mode='a', header=False, quoting=csv.QUOTE_MINIMAL)
                print(f"Appended chunk {i + 1} to the output file.")

    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_CSV_PATH}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return

    print("\nAnalysis complete. All chunks have been processed.")


if __name__ == "__main__":
    main()
```

```
parts.csv"...
Filtering values > 6500 and normalizing.

--- Statistics ---
Count: 823250
Mean: 4.791, Std: 4.916
Min: 1.000, Q25: 2.275, Median: 3.610, Q75: 5.687, Max: 406.668
95% CI for mean: [4.781, 4.802]
```

```
venv) [s@qr text_content]$ python text_content_analysis.py
Starting analysis of 'deduplicated_fixed.csv'...
Using 16 worker processes and a chunk size of 1000 rows.
Processing Chunks: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1854/1854 [00:00<00:00, 4470.77it/s]

--- Statistics on Character Lengths of Text Parts (>= 5 chars) ---
Total valid parts analyzed: 1,853,338
Average (Mean) Length:      14790.06
Median Length:              3042.00
Standard Deviation:         25864.61
95th Percentile:            54193.00 (95% of parts are shorter than this)
Average Length in Top 1%:   170261.30
Average Length in Top 0.1%: 385385.12

Generating plots...
Plots saved to 'length_analysis_plots.png'
```


# ETH-CLUSTER-NOTES

Note : court decision neuchatel in .md + quite big system prompt is 14474 input tokens for bytedance model

Using the tokenizer for qwen3-235B, one token is on average 3.4 characters

run bytedance on single a100 80GB

```
vllm serve "ByteDance-Seed/Seed-OSS-36B-Instruct"   --max-model-len 20000  --swap-space 32 --gpu_memory_utilization=0.95
```

## CHECK ACCOUNT STATE

https://slurm-jobs-webgui.euler.hpc.ethz.ch/

## DESTROY EVERYTHING

```
scancel --user=$USER
```

## Create new instance

https://jupyter.euler.hpc.ethz.ch/hub/spawn

-> 4x a100 did not work
-> 2x a100 did work

I requested 128GB RAM, but got 999GB

Be careful: download big models to  /cluster/scratch/saliechti

OTHERWISE ERRORS WHEN MORE THAN 50GB ON /home
