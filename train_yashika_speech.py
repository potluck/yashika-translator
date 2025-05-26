import os
from pathlib import Path
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2ForCTC

# Set paths
DATA_DIR = "/Users/potluck/code/yashika-translator/data"

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

model.eval()

def extract_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling across time
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Load all data
X, y = [], []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file_name in os.listdir(label_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(label_path, file_name)
            try:
                emb = extract_embedding(file_path)
                X.append(emb)
                y.append(label)
            except Exception as e:
                print(f"Error with {file_path}: {e}")

print(f"Loaded {len(X)} samples.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Remove any rows with NaN or infinite values
valid_mask = np.isfinite(X).all(axis=1)
X = X[valid_mask]
y = y[valid_mask]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


def predict_sample(audio_path):
    emb = extract_embedding(audio_path)
    pred = clf.predict([emb])[0]
    print(f"Predicted label: {pred}")

# Example usage:
predict_sample("/Users/potluck/code/yashika-translator/test_audio3.wav")