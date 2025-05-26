import os
from pathlib import Path
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import soundfile as sf

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
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # 3. Zero crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]
        features.extend([np.mean(zero_crossings), np.std(zero_crossings)])
        
        # 4. Root mean square energy
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # 5. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=256, n_mels=64)
        features.extend([np.mean(mel_spec), np.std(mel_spec)])
        
        # 6. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])
        
        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512, hop_length=256)
        features.extend([np.mean(contrast), np.std(contrast)])
        
        # 8. Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend([np.mean(tonnetz), np.std(tonnetz)])
        
        # 9. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256)
        features.extend([np.mean(rolloff), np.std(rolloff)])
        
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
clf = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance'
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

def split_audio_into_segments(audio_path, min_silence_duration=0.1):
    """Split audio into segments based on silence detection"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        print(f"Loaded audio: {audio_path}, length: {len(y)} samples, duration: {len(y)/sr:.2f} sec, max amplitude: {np.max(np.abs(y)):.4f}")
        
        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
        print(f"Detected intervals: {intervals}")
        
        # Filter out very short segments (likely noise)
        min_samples = int(min_silence_duration * sr)
        valid_intervals = [interval for interval in intervals if interval[1] - interval[0] > min_samples]
        print(f"Valid intervals (>{min_samples} samples): {valid_intervals}")
        
        # Extract segments
        segments = []
        for start, end in valid_intervals:
            segment = y[start:end]
            print(f"Segment: start={start}, end={end}, length={end-start} samples, duration={(end-start)/sr:.2f} sec")
            segments.append(segment)
            
        return segments, sr
        
    except Exception as e:
        print(f"Error splitting audio {audio_path}: {e}")
        return None, None

def predict_sample(audio_path):
    """Predict the class of a new audio sample, handling multiple words"""
    # Split audio into segments
    segments, sr = split_audio_into_segments(audio_path)
    if segments is None:
        return "Error processing audio file"
    
    results = []
    for i, segment in enumerate(segments):
        # Save segment to temporary file
        temp_path = f"temp_segment_{i}.wav"
        sf.write(temp_path, segment, sr)
        
        # Extract features for this segment
        features = extract_features(temp_path)
        if features is None:
            continue
            
        # Scale the features
        features = scaler.transform(features.reshape(1, -1))
        
        # Get prediction probabilities
        probs = clf.predict_proba(features)[0]
        pred = clf.predict(features)[0]
        
        # Get confidence score
        confidence = probs.max()
        
        print(f"\nSegment {i+1}:")
        print(f"Predicted label: {pred}")
        print(f"Confidence: {confidence:.2%}")
        
        # Print top 3 predictions
        top_3_idx = np.argsort(probs)[-3:][::-1]
        print("Top 3 predictions:")
        for idx in top_3_idx:
            print(f"{clf.classes_[idx]}: {probs[idx]:.2%}")
            
        results.append((pred, confidence))
        
        # Clean up temporary file
        os.remove(temp_path)
    
    return results

# Example usage:
print("\nStarting prediction...")
results = predict_sample("/Users/potluck/code/yashika-translator/test_audio3.wav")
print("\nPrediction results:", results)