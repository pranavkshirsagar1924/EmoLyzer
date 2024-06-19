import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
# Emotion mapping based on a known convention (you may need to adjust this based on your filename pattern)
emotion_map = {
    'a': 'anger',
    'b': 'boredom',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    's': 'sadness',
    'n': 'neutral'
}

# Extract pitch using PYIN
def extract_features(file_path):
    audio, sr = librosa.load(file_path)
    
    # Extract pitch using PYIN
    fe, voice_flag, voice_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    nan_indices = np.isnan(fe)
    fe = fe[~nan_indices]
    mean_pitch = np.mean(fe) if len(fe) > 0 else 0

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)  # Calculate mean along the axis of coefficients

    # Compute zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    mean_zcr = np.mean(zcr)

    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    mean_spectral_centroid = np.mean(spectral_centroid)

    # Compute spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    mean_spectral_band_width = np.mean(spectral_bandwidth)

    # Compute spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    mean_spectral_contrast = np.mean(spectral_contrast, axis=1)  # Calculate mean along the axis of contrasts

    # Estimate tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # Prepare feature vector
    mean_mfccs_index = mean_mfccs[0]
    mean_spectral_contrast_index = mean_spectral_contrast[0]

    # Return the feature vector as a list
    return [mean_pitch, mean_mfccs_index, mean_zcr, mean_spectral_centroid, mean_spectral_band_width, mean_spectral_contrast_index, tempo[0]]

# Manually create a small dataset with two files
file_paths = []  # Example paths to your two files
for f in os.listdir('t'):
    file_paths.append('t/'+f)
emotions = [ "anger","boredom","anxiety","happiness", "sadness","disgust", "neutral"]  # Corresponding emotions

X = []
y = []

for file_path, emotion in zip(file_paths, emotions):
    features = extract_features(file_path)
    X.append(features)
    y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Unique labels found:", np.unique(y))  # Debugging statement

# Ensure y has more than one unique class
if len(np.unique(y)) <= 1:
    raise ValueError("Not enough classes to train the model. Please check the labels.")

# Directly train and predict on the small dataset
svm = SVC(kernel="linear", C=1.0, random_state=42)
svm.fit(X, y)
y_pred = svm.predict(X)
accuracy = accuracy_score(y, y_pred)

while True:
 try:
    filename = input("FIlemame")
    features = extract_features('t/'+filename)
    features= np.array(features).reshape(1,-1)
    pre_emo = svm.predict(features)[0]
    print("predicted emotion is ",pre_emo)
 except Exception as e:
    print("Looping x")
    print("Accuracy:", accuracy)


