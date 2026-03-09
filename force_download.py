import os
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the model we want
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
save_directory = "./my_local_model"

# Delete existing local folder if it exists (to be safe)
if os.path.exists(save_directory):
    shutil.rmtree(save_directory)

print(f"⬇️ Downloading '{model_name}' to {save_directory}...")
print("⏳ This might take 1-2 minutes. Please wait...")

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("✅ SUCCESS! Model saved to 'my_local_model'.")