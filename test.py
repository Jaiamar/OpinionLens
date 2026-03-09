from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Test the model on a clearly negative sentence
text = "Unfortunately In Latin America is just complete shit"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

print("⏳ Downloading/Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = softmax(output.logits[0].detach().numpy())

labels = ['Negative', 'Neutral', 'Positive']
print(f"\nTest Sentence: '{text}'")
for i in range(len(labels)):
    print(f"{labels[i]}: {scores[i]:.4f}")