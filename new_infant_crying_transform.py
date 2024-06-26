import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Configuration
data_directory = '~/data/'
label_file = '~/data/1942/1942.csv'
fs = 16000  # Adjusted to match wav2vec 2.0 expected input
results_directory = 'analysis/analysis-06242024'

# Ensure the directory exists
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Load wav2vec 2.0 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# To enable outputting hidden states, modify the model loading line:
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)

# Silence Detection
def is_silent(data, threshold=0.01):
    return torch.max(torch.abs(data)) < threshold

# Extract features using wav2vec 2.0
def extract_features(data):
    inputs = processor(data, sampling_rate=fs, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Select different layers, e.g., the last four layers
    last_layers = outputs.hidden_states[-4:]  # Accessing hidden states if output_hidden_states=True in config
    features = torch.cat([layer.mean(dim=1) for layer in last_layers], dim=-1)  # Concatenate means of last four layers
    
    return features.squeeze()

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
        waveform, sr = torchaudio.load(os.path.expanduser(row['file_path']))
        if sr != fs:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fs)(waveform)
        waveform = waveform.mean(0)  # Assume mono input
        if not is_silent(waveform):
            features.append(extract_features(waveform).numpy())
            labels.append(row['label'])
    return np.array(features), np.array(labels)

# Train Model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Setting class_weight to 'balanced' to automatically adjust weights inversely proportional to class frequencies
    classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')
    classifier.fit(X_train, y_train)
    return classifier, scaler

# Predict
def predict(classifier, scaler, X_test):
    X_test = scaler.transform(X_test)
    y_pred = classifier.predict(X_test)
    y_probs = classifier.predict_proba(X_test)[:, 1]  # Probability for class 1
    return y_pred, y_probs

# Evaluate
def evaluate(y_test, y_pred, y_probs):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    print(report)
    with open(os.path.join(results_directory, 'classification_report_transform.txt'), 'w') as f:
        f.write(report)
    plot_roc_curve(y_test, y_probs)

# ROC Curve Plotting
def plot_roc_curve(y_test, y_probs):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_directory, 'ROC_Curve_transform.png'))
    plt.close()

# Main Execution
# if __name__ == "__main__":
df = load_and_label_data(data_directory, label_file)
features, labels = process_data(df)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
classifier, scaler = train_model(X_train, y_train)
y_pred, y_probs = predict(classifier, scaler, X_test)
evaluate(y_test, y_pred, y_probs)
   
