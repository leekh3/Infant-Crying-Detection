from os.path import expanduser, exists, join
import glob
import re
import pandas as pd
import os
import librosa
import soundfile as sf
from preprocessing import preprocessing
from predict import predict

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

def extract_number_from_path(path):
    match = re.search(r'/?(\d+)[^\d/]*\.wav$', path)
    if match:
        return match.group(1)
    else:
        print(f"Cannot extract the subject name from this file: {path}")
        return None

def create_folder_if_not_exists(folder):
    if not exists(folder):
        os.makedirs(folder)

def segment_wav_file(input_wav, tmp_folder, segment_duration=2*60):
    try:
        y, sr = librosa.load(input_wav, sr=None)
        segment_samples = segment_duration * sr
        for i in range(5):
            start_sample = i * segment_samples
            end_sample = (i + 1) * segment_samples
            output_file = join(tmp_folder, f'{sdan}_segment_{i}.wav')
            sf.write(output_file, y[start_sample:end_sample], sr, format='WAV', subtype='PCM_16')
    except Exception as e:
        print(f"Failed to process {input_wav} due to error: {e}. Skipping...")

def process_files(tmp_folder, preprocessedFolder, predictedFolder, probFolder):
    inFiles = sorted(glob.glob(join(tmp_folder, "*.wav")), key=sort_key)
    for idx, inFile in enumerate(inFiles):
        preprocessedFile = join(preprocessedFolder, f'{idx}.csv')
        predictedFile = join(predictedFolder, f'{idx}.csv')
        probFile = join(probFolder, f'{idx}.csv')

        preprocessing(inFile, preprocessedFile)
        predict(inFile, preprocessedFile, predictedFile, probFile)

def concatenate_csv_in_folder(folder_path, output_file):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = [pd.read_csv(join(folder_path, file)) for file in csv_files]
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)

#################################
import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import glob

# Function to detect infant crying in an audio file
def detect_infant_crying(wav_file_path, low_freq, high_freq, nyquist_freq):
    # Load the audio file
    fs, data = wavfile.read(wav_file_path)
    if data.ndim > 1:  # Convert to mono if stereo
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    # Apply bandpass filter
    b, a = sig.butter(4, [low_freq / nyquist_freq, high_freq / nyquist_freq], btype='band')
    filtered_data = sig.filtfilt(b, a, data)

    # Compute energy of the filtered signal using a sliding window
    window_size = int(fs * 0.1)  # 100 ms window
    step_size = int(fs * 0.05)  # 50 ms step
    energy = np.array([np.sum(filtered_data[i:i + window_size] ** 2) for i in
                       range(0, len(filtered_data) - window_size, step_size)])

    # Compute short-term and long-term average energy
    short_term_avg_energy = np.mean(energy)
    long_term_avg_energy = np.mean(energy[-10:])

    # Compute energy ratio
    energy_ratio = short_term_avg_energy / long_term_avg_energy

    return energy_ratio > 1.5

# Function to count the number of detections in an audio folder
def detection_on_5sec_by_kyunghun(inFolder):
    inFiles = glob.glob(inFolder + '/*.wav')
    inFiles.sort()

    detected_count = 0
    for audio_file in inFiles:
        is_crying = detect_infant_crying(audio_file, 250, 3000, 22050 // 2)
        if is_crying:
            detected_count += 1
    return detected_count

# Function to segment a large audio file into smaller segments
def segment_audio(file_path, segment_duration, output_dir):
    audio = AudioSegment.from_wav(file_path)
    num_segments = len(audio) // segment_duration
    os.makedirs(output_dir, exist_ok=True)

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[start_time:end_time]
        output_filename = f"segment_{i + 1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        segment.export(output_path, format="wav")
        segments.append(output_path)

    return segments

# Function to calculate the number of detections for a segment
def calculateNumOfDetect(outFile):
    segment_duration = 5000  # 5 seconds in milliseconds
    output_dir = "tmp"
    segments = segment_audio(outFile, segment_duration, output_dir)
    num_detection = detection_on_5sec_by_kyunghun(output_dir)
    return num_detection

# Function to calculate the quality of each segment
def calculate_quality(segments):
    quality = []
    for segment in segments:
        num_detection = calculateNumOfDetect(segment)
        quality.append(1 if num_detection >= 10 else 0)
    return quality

# Function to generate a CSV file from the quality data
def generate_csv(quality, output_csv):
    seconds_quality = []
    for q in quality:
        seconds_quality.extend([q] * 600)  # 600 seconds in 10 minutes

    df = pd.DataFrame(seconds_quality)
    df.to_csv(output_csv, index=False, header=False)

input_wav_path = "/Users/leek13/data/LENA/1180_LENA/AN1/e20171121_094647_013506.wav"
segments = segment_audio(input_wav_path, 600000, "segments_dir")  # 10 minutes in milliseconds
quality = calculate_quality(segments)
generate_csv(quality, "output_quality.csv")
####################################


# Main execution
home = expanduser("~")
input_wav = '2828.wav'
sdan = extract_number_from_path(input_wav)

if sdan:
    tmp_folder = join(home, f'tmp/{sdan}')
    create_folder_if_not_exists(tmp_folder)
    segment_wav_file(input_wav, tmp_folder)

    preprocessedFolder = join(tmp_folder, 'preprocessed')
    predictedFolder = join(tmp_folder, 'predicted')
    probFolder = join(tmp_folder, 'prob')
    create_folder_if_not_exists(preprocessedFolder)
    create_folder_if_not_exists(predictedFolder)
    create_folder_if_not_exists(probFolder)

    process_files(tmp_folder, preprocessedFolder, predictedFolder, probFolder)

    output_file = join(tmp_folder, "prediction.csv")
    concatenate_csv_in_folder(predictedFolder, output_file)
    print("Processing complete.")
