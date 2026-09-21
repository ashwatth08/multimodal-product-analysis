import os
import pandas as pd

# Paths based on your folder structure
BASE_DIR = "MVSA_Single"
DATA_DIR = os.path.join(BASE_DIR, "data")
LABEL_FILE = os.path.join(BASE_DIR, "labelResultAll.txt")

records = []

# Class mapping: 0 = Negative, 1 = Neutral, 2 = Positive
label_map = {"negative": 0, "neutral": 1, "positive": 2}

print(f"Reading label file from: {LABEL_FILE}")
print(f"Looking for images/text inside: {DATA_DIR}")

with open(LABEL_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Skip the header line ('ID\ttext,image\n')
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    
    # Split by tab: ID and 'text,image'
    parts = line.split("\t")
    if len(parts) >= 2:
        sample_id = parts[0].strip()
        sentiment_parts = parts[1].strip().split(",")
        
        if len(sentiment_parts) >= 2:
            text_sentiment = sentiment_parts[0].strip().lower()
            image_sentiment = sentiment_parts[1].strip().lower()

            # Agreement consensus filter
            if text_sentiment == image_sentiment and text_sentiment in label_map:
                img_path = os.path.join(DATA_DIR, f"{sample_id}.jpg")
                txt_path = os.path.join(DATA_DIR, f"{sample_id}.txt")

                if os.path.exists(img_path) and os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
                            text_content = tf.read().strip()
                        
                        if len(text_content) > 0:
                            records.append({
                                "id": sample_id,
                                "image_path": img_path,
                                "text": text_content,
                                "label": label_map[text_sentiment]
                            })
                    except Exception:
                        continue

df = pd.DataFrame(records)

print(f"\n--- Preprocessing Complete ---")
print(f"Total valid paired samples matched: {len(df)}")

if len(df) > 0:
    print("\nClass distribution:")
    print(df["label"].value_counts().rename({0: "Negative (0)", 1: "Neutral (1)", 2: "Positive (2)"}))

    output_csv = "clean_mvsa_paired.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved clean paired data to: {output_csv}")
else:
    print("Error: Still 0 samples matched. Please check if files inside MVSA_Single/data have .jpg and .txt extensions.")