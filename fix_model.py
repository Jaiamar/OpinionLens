import shutil
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# 1. CLEANUP: Find and delete the broken model folders
cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
targets = [
    "models--cardiffnlp--twitter-roberta-base-sentiment",
    "models--cardiffnlp--twitter-roberta-base-sentiment-latest"
]

print("🧹 Cleaning up old corrupted models...")
found = False
for target in targets:
    path = cache_dir / target
    if path.exists():
        print(f"   - Deleting corrupted folder: {path}")
        try:
            shutil.rmtree(path)
            found = True
        except Exception as e:
            print(f"   ⚠️ Could not delete {path}. Error: {e}")
            print("   Please delete it manually if possible.")

if not found:
    print("   - No old cache found (or already deleted).")

# 2. REDOWNLOAD: Force a fresh download
print("\n⬇️ Downloading fresh AI Model (This may take a minute)...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("✅ Download Complete!")
except Exception as e:
    print(f"❌ Download Failed: {e}")
    exit()

# 3. TEST: Verify it actually works
print("\n🧪 Running Diagnostic Test...")
test_text = "This product is absolute garbage and I hate it."
print(f"   Input: '{test_text}'")

encoded_input = tokenizer(test_text, return_tensors='pt')
output = model(**encoded_input)
scores = softmax(output.logits[0].detach().numpy())

# Get labels
labels = ['Negative', 'Neutral', 'Positive']
ranking = np.argsort(scores)[::-1]
top_sentiment = labels[ranking[0]]

print(f"   Result: {top_sentiment.upper()}")
print(f"   Scores: Neg={scores[0]:.4f}, Neu={scores[1]:.4f}, Pos={scores[2]:.4f}")

if top_sentiment == "Negative":
    print("\n✅ SUCCESS! The model is fixed. You can run your app now.")
else:
    print("\n❌ STILL BROKEN. The model is confused.")