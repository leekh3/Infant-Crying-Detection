"""
Infant Cry Detection and Audio Processing Pipeline (for ACNP presentation)

This script provides a comprehensive solution for processing audio files, particularly focusing on the detection
of infant cries. It includes functions for audio segmentation, signal processing, machine learning predictions, and
data consolidation. The key components of this pipeline involve:

1. Segmenting large audio files into smaller, manageable segments.
2. Applying bandpass filters and calculating energy ratios to detect infant crying.
3. Predicting various aspects from the audio data using pre-trained models.
4. Aggregating results and generating concise reports in CSV format.

Author: Kyunghun Lee
Date: Tue Nov 28 01:30:17 EST 2023
"""

from os.path import exists, join
import re
import librosa
import soundfile as sf
from preprocessing import preprocessing
from predict import predict
import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import glob
import getpass
import sys
import platform

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
            output_file = join(tmp_folder, f'segment_{i}.wav')
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
    # Read CSV files without headers and select only the second column
    dataframes = [pd.read_csv(join(folder_path, file), header=None, usecols=[1]) for file in csv_files]
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    # Write to CSV without a header
    concatenated_df.to_csv(output_file, index=False, header=False)

#################################
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
def detection_on_5sec(inFolder):
    inFiles = glob.glob(inFolder + '/*.wav')
    inFiles.sort()

    detected_count = 0
    for audio_file in inFiles:
        is_crying = detect_infant_crying(audio_file, 10, 350, 22050 // 2)
        # is_crying = detect_infant_crying(audio_file, 10, 350, 22050 // 2)
        if is_crying:
            detected_count += 1
    # print(detected_count)
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
def calculateNumOfDetect(tmp_folder,outFile):
    segment_duration = 5000  # 5 seconds in milliseconds
    output_dir = f"{tmp_folder}/tmp_for_quality_testing/"
    segments = segment_audio(outFile, segment_duration, output_dir)
    num_detection = detection_on_5sec(output_dir)
    return num_detection

# Function to calculate the quality of each segment
def calculate_quality(tmp_folder,segments):
    quality = []
    for segment in segments:
        num_detection = calculateNumOfDetect(tmp_folder,segment)
        quality.append(1 if num_detection >= 10 else 0)
    return quality

# Function to generate a CSV file from the quality data
def generate_csv(quality, output_csv):
    seconds_quality = []
    for q in quality:
        seconds_quality.extend([q] * 600)  # 600 seconds in 10 minutes

    df = pd.DataFrame(seconds_quality)
    df.to_csv(output_csv, index=False, header=False)
    print(f"output quality file generated:{output_csv}")

def InfantCryDetectionPipeline_ACNP(input_wav_path,output_quality_file_path,output_prediction_file_path):

    # input_wav_path = "/Users/leek13/data/LENA/1180_LENA/AN1/e20171121_094647_013506.wav"
    from os.path import expanduser
    home = expanduser("~")
    tmp_folder_path = f"{home}/tmp_infant_crying/"

    # Check operating system
    if platform.system() == "Darwin":  # Darwin is the system name for macOS
        home = os.path.expanduser('~')  # Gets the home directory path
        tmp_folder_path = f"{home}/tmp_infant_crying/"
    else:
        tmp_folder_path = f"/lscratch/{os.path.expanduser('~').split('/')[-1]}"

    # tmp_folder_path = f"/scratch/{getpass.getuser()}"
    # output_quality_file_path = "output_quality.csv"
    # output_prediction_file_path = "output_prediction.csv"

    tmp_folder = tmp_folder_path + os.path.splitext(os.path.basename(input_wav_path))[0]
    # segments = segment_audio(input_wav_path, 600000, "segments_dir")  # 10 minutes in milliseconds
    segments = segment_audio(input_wav_path, 600000, tmp_folder)  # 10 minutes in milliseconds
    quality = calculate_quality(tmp_folder,segments)
    generate_csv(quality, output_quality_file_path)
    ####################################

    # Main execution
    for segment in segments:
        segment_folder = tmp_folder + '/' + os.path.splitext(os.path.basename(segment))[0]
        create_folder_if_not_exists(segment_folder)
        segment_wav_file(segment, segment_folder)

        preprocessedFolder = join(segment_folder, 'preprocessed')
        predictedFolder = join(segment_folder, 'predicted')
        probFolder = join(segment_folder, 'prob')
        create_folder_if_not_exists(preprocessedFolder)
        create_folder_if_not_exists(predictedFolder)
        create_folder_if_not_exists(probFolder)

        process_files(segment_folder, preprocessedFolder, predictedFolder, probFolder)
        prediction_segment_file = join(segment_folder, "prediction.csv")
        concatenate_csv_in_folder(predictedFolder, prediction_segment_file)
        print("Processing complete: segment")

    prediction_files = sorted(glob.glob(f"{tmp_folder}/segment_*/prediction.csv"),key=sort_key)
    concatenated_df = pd.concat([pd.read_csv(file, header=None) for file in prediction_files], axis=0, ignore_index=True)
    concatenated_df.to_csv(f'{output_prediction_file_path}', index=False, header=False)

# Extract arguments from sys.argv
input_wav_path = sys.argv[1]
output_quality_file_path = sys.argv[2]
output_prediction_file_path = sys.argv[3]

# Call the main function with the provided arguments
InfantCryDetectionPipeline_ACNP(input_wav_path, output_quality_file_path, output_prediction_file_path)

# python3 InfantCryDetectionPipeline_ACNP.py /Users/leek13/data/LENA/1180_LENA/AN1/e20171121_094647_013506.wav output_quality.csv output_prediction.csv
