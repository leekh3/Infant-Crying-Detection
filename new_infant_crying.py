import os
import numpy as np
import librosa
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Configuration
data_directory = '~/data/'
label_file = '~/data/1942/1942.csv'
fs = 22050  # Sample rate for audio files
results_directory = 'analysis/analysis-0624204'

# Ensure the directory exists
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Audio Preprocessing
def preprocess_audio(data):
    if is_silent(data):
        return None
    return bandpass_filter(data, 300, 3000, fs)

# Bandpass Filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Silence Detection
def is_silent(data, threshold=0.01):
    return np.max(np.abs(data)) < threshold

# Feature Extraction
def extract_features(data):
    mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Load and Label Data
def load_and_label_data(data_directory, label_file):
    df = pd.read_csv(label_file)
    df['label'] = df['Final Code'].apply(lambda x: 1 if x in [1, 3] else 0)
    df['file_path'] = df.apply(lambda row: os.path.join(data_directory, f"{row['folder_name']}", f"{row['filename']}"), axis=1)
    return df

# Data Processing
def process_data(df):
    features = []
    labels = []
    for _, row in df.iterrows():
        data, sr = librosa.load(os.path.expanduser(row['file_path']), sr=fs)
        processed_data = preprocess_audio(data)
        if processed_data is not None:
            features.append(extract_features(processed_data))
            labels.append(row['label'])
    return np.array(features), np.array(labels)

# Train Model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    return model, scaler

# Predict
def predict(model, scaler, X_test):
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # Probability for class 1
    return y_pred, y_probs

# Evaluate
def evaluate(y_test, y_pred, y_probs):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    print(report)
    # Save the classification report to a text file
    with open(os.path.join(results_directory, 'classification_report.txt'), 'w') as f:
        f.write(report)
    plot_roc_curve(y_test, y_probs)

# ROC Curve Plotting
def plot_roc_curve(y_test, y_probs):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_directory, 'ROC_Curve.png'))  # Save the ROC curve
    plt.close()

# Main Execution
if __name__ == "__main__":
    df = load_and_label_data(os.path.expanduser(data_directory), os.path.expanduser(label_file))
    features, labels = process_data(df)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
    model, scaler = train_model(X_train, y_train)
    y_pred, y_probs = predict(model, scaler, X_test)
    evaluate(y_test, y_pred, y_probs)
