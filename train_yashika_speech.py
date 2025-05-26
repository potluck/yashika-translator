import os
from pathlib import Path
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set paths
DATA_DIR = "/Users/potluck/code/yashika-translator/data"

def extract_features(audio_path):
    """Extract simple audio features using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract basic features
        features = []
        
        # 1. Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # 3. Zero crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zero_crossings), np.std(zero_crossings)])
        
        # 4. Root mean square energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Convert to numpy array
        features = np.array(features)
        
        # Check for NaN or infinite values
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"Warning: NaN or infinite values found in features for {audio_path}")
            return None
            
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Load all data
X, y = [], []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file_name in os.listdir(label_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(label_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)

print(f"Loaded {len(X)} valid samples.")

if len(X) == 0:
    raise ValueError("No valid samples were loaded!")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Print class distribution
unique_labels, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for label, count in zip(unique_labels, counts):
    print(f"{label}: {count} samples")

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the classifier
clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)

# Perform cross-validation
n_splits = 3  # Use 3-fold cross-validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Get cross-validation scores
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train the final model on all data
clf.fit(X, y)

def predict_sample(audio_path):
    """Predict the class of a new audio sample"""
    features = extract_features(audio_path)
    if features is None:
        return "Error processing audio file"
    
    # Scale the features
    features = scaler.transform(features.reshape(1, -1))
    
    # Get prediction probabilities
    probs = clf.predict_proba(features)[0]
    pred = clf.predict(features)[0]
    
    # Get confidence score
    confidence = probs.max()
    
    print(f"Predicted label: {pred}")
    print(f"Confidence: {confidence:.2%}")
    
    # Print top 3 predictions
    top_3_idx = np.argsort(probs)[-3:][::-1]
    print("\nTop 3 predictions:")
    for idx in top_3_idx:
        print(f"{clf.classes_[idx]}: {probs[idx]:.2%}")
    
    return pred, confidence

# Example usage:
predict_sample("/Users/potluck/code/yashika-translator/test_audio1.wav")