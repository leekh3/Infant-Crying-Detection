import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

#for the CNN
import matplotlib
matplotlib.use('Agg')
import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
from math import sqrt, pi, exp
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import random
import h5py
import copy

# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, model_from_json, load_model
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Input
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import backend as K

n_fft = 980
#1764
hop_length = 490
#882
n_mels = 225
img_rows, img_cols = 225, 225
batch_size = 128
num_classes = 2

import sys
sys.path.insert(1, 'yao_training') # Path to the directory that contains the file, not the file itself
from train_alex_kyunghun import CustomPyTorchModel

def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
	mel_spectrogram = np.asarray(mel_spectrogram)
	for i in range(time_mask_num):
		t = np.random.randint(low = 0, high = time_masking_para)
		t0 = np.random.randint(low = 0, high = tau - t)
		# mel_spectrogram[:, t0:t0 + t] = 0
		mel_spectrogram[:, t0:(t0 + t)] = 0
	return list(mel_spectrogram)

def whatIsAnEvent(data, event_thre):
    previous = (-1, -1)
    start = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1 and previous[1] == -1:
            previous = (i, data[i, 0])
        elif data[i, 1] == 0 and previous[1] != -1 and data[i - 1, 1] == 1:
            start = (i, data[i, 0])
            if start[1] - previous[1] <= event_thre:
                data[previous[0] : start[0], 1] = 0
            previous = (-1, -1)
            start = (-1, -1)

    if previous[1] != -1 and data[-1, 0] - previous[1] + 1 <= event_thre:
        data[previous[0] :, 1] = 0
    return data

def combineIntoEvent(data, time_thre):
    previous = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1:
            start = (i, data[i, 0])
            if previous[1] > 0 and start[1] - previous[1] <= time_thre:
                data[previous[0] : start[0], 1] = 1
            previous = start

    if previous[1] > 0 and data[i - 1, 0] - previous[1] <= time_thre:
        data[previous[0] : i, 1] = 1

    return data


def label_to_num(input_label):
	if input_label == 'other' or input_label == 'notcry':
		return 0
	elif input_label == 'fuss':
		return 1
	elif input_label == 'cry':
		return 2
	elif input_label == 'scream':
		return 3
	else:
		return 4

# # data_folder = './ahsans/'
# data_folder = "/Users/leek13/data/deBarbaroCry/"
# label_folder = './ahsans_labels_filtered/'
#
# test_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
# test_folders = ['P31']

# alex_model_path = 'pics_alex_noflip_ahsans' + session_num + '_distress.h5'
# real_label_folder = './ahsans_labels/'

home_directory = os.path.expanduser("~")
data_folder = home_directory + "/data/deBarbaroCry/kyunghun-10min-data/"
label_folder = home_directory + "/data/deBarbaroCry/kyunghun-10min-label/"
test_folders = ['P30','P38']
model_out_folder = '.trained'
real_label_folder = './ahsans_labels/'
user_folders = ['P38', 'P31', 'P32', 'P30']
# def train_alex_svm(data_folder,label_folder,test_folders,model_out_folder,real_label_folder):
def train_alex_svm(data_folder,user_folders,label_folder,model_out_folder,real_label_folder):
	import os
	import numpy as np
	import matplotlib as plt
	
	test_folder = test_folders[0]
	# for test_folder in test_folders:
	print(test_folder)

	episodes = []
	all_data = []
	all_labels = []
	all_feature_data = []
	
	# Find the folder starting with 'P' from data_folder.
	# For example, user_folders = ['P38', 'P31', 'P32', 'P30']
	# user_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]


	# Find all the subfolder if user_folder is not test_folder.
	# For example,
	# episodes = 
	# ['P38/P38_25', 'P38/P38_13', 'P38/P38_23', 'P38/P38_26', 'P38/P38_10', 'P38/P38_28', 
	# 'P38/P38_19', 'P38/P38_22', 'P38/P38_17', 'P38/P38_2', 'P38/P38_0', 'P38/P38_15', 
	# 'P38/P38_14', 'P38/P38_7', 'P38/P38_21', 'P38/P38_9', 'P38/P38_18', 'P38/P38_29', 
	# 'P38/P38_5', 'P38/P38_27', 'P38/P38_1', 'P38/P38_24', 'P38/P38_8', 'P38/P38_6', 
	# 'P38/P38_4', 'P38/P38_12', 'P38/P38_3', 'P38/P38_16', 'P38/P38_20', 'P38/P38_11', 
	# 'P31/P31_6', 'P31/P31_26', 'P31/P31_0', 'P31/P31_33', 'P31/P31_28', 'P31/P31_17', 
	# 'P31/P31_10', 'P31/P31_4', 'P31/P31_36', 'P31/P31_23', 'P31/P31_24', 'P31/P31_30', 
	# 'P31/P31_20', 'P31/P31_41', 'P31/P31_21', 'P31/P31_3', 'P31/P31_37', 'P31/P31_19', 
	# 'P31/P31_13', 'P31/P31_34', 'P31/P31_27', 'P31/P31_32', 'P31/P31_25', 'P31/P31_11', 
	# 'P31/P31_9', 'P31/P31_38', 'P31/P31_29', 'P31/P31_15', 'P31/P31_31', 'P31/P31_39', 
	# 'P31/P31_12', 'P31/P31_2', 'P31/P31_40', 'P31/P31_14', 'P31/P31_8', 'P31/P31_1', 
	# 'P31/P31_35', 'P31/P31_7', 'P31/P31_18', 'P31/P31_22', 'P31/P31_5', 'P31/P31_42', 
	# 'P31/P31_16', 'P32/P32_12', 'P32/P32_27', 'P32/P32_24', 'P32/P32_14', 'P32/P32_29', 
	# 'P32/P32_28', 'P32/P32_17', 'P32/P32_2', 'P32/P32_18', 'P32/P32_26', 'P32/P32_1', 
	# 'P32/P32_16', 'P32/P32_31', 'P32/P32_10', 'P32/P32_9', 'P32/P32_22', 'P32/P32_6', 
	# 'P32/P32_13', 'P32/P32_32', 'P32/P32_15', 'P32/P32_19', 'P32/P32_21', 'P32/P32_5', 
	# 'P32/P32_33', 'P32/P32_7', 'P32/P32_30', 'P32/P32_4', 'P32/P32_3', 'P32/P32_8', 
	# 'P32/P32_23', 'P32/P32_0', 'P32/P32_11', 'P32/P32_25', 'P32/P32_20']

	for user_folder in user_folders:
		if user_folder != test_folder:
			user_episodes = [file for file in os.listdir(data_folder + user_folder) if file.endswith('.wav')]
			for user_episode in user_episodes:
					episodes.append(user_folder + '/' + user_episode[:-4])

	#get 5s windows with 1s overlap
	"""
	This for loop in your code is performing several key steps in processing and preparing audio data for a machine learning task. Here's a breakdown of its major components:
	Iterating Over Episodes: The loop iterates over a list of episodes. Each episode is likely a unique identifier for an audio recording.
	Loading Audio Data: For each episode, it constructs the path to the corresponding audio file (audio_filename) and annotation file (annotation_filename_ra). It then loads the audio file using librosa.load, which returns the audio time series y and its sampling rate sr. It also calculates the duration of the audio file.
	Processing Annotations: The annotations (presumably labels for segments of the audio) are read from a CSV file. The code expects each row in the annotation file to contain at least three elements. It appears to convert the third element of each row into a numerical label using label_to_num. It then filters and processes these annotations based on certain conditions (e.g., ensuring the start time of an annotation is within a certain range of the audio duration). The processed annotations are stored in ra_annotations.
	Generating Windows: The script creates 5-second windows from the audio, with each window starting 1 second apart (assuming a 1s overlap). For each annotation in ra_annotations, it checks if the duration of the annotation is at least 5 seconds and then generates these windows. It also assigns labels to these windows, depending on the condition in the annotation data.
	Extracting Audio Features: Two types of features are extracted from each window:
	Spectrogram Features: A Mel spectrogram is computed using librosa.feature.melspectrogram, which is then converted to a decibel scale. A slice of this spectrogram corresponding to each window is appended to image_windows.
	Statistical Features: Short-term audio features are extracted using a function (assumed to be ShortTermFeatures.feature_extraction). Statistical features (mean, median, standard deviation) of these are computed for each window and appended to feature_windows.
	Aggregating Data: Finally, 
	the labels, spectrogram slices, and statistical features for each window 
	are added to the lists all_labels, all_data, and all_feature_data, respectively. 
	These aggregated lists likely represent the dataset that will be used for 
	training or evaluating a machine learning model.
	"""
	for episode in episodes:
		print(episode)
		audio_filename = data_folder + episode + '.wav'
		annotation_filename_ra = label_folder + episode + '.csv'

		y, sr = librosa.load(audio_filename)
		duration = librosa.get_duration(y = y, sr = sr)
		previous = 0

		#ra_annotations: [[0, 1.0, 'other'], [1.0, 243.0, 'fuss']...]
		ra_annotations = []
		with open(annotation_filename_ra, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if len(row) > 0:
					row[2] = label_to_num(row[2])
					if float(row[0]) - previous > 0 and int(row[2]) <= 2 and int(row[0]) <= duration // 10 :
						ra_annotations.append([float(row[0]), min(duration // 10, float(row[1])), int(row[2])])
		#windows = {'other': [[243.0, 248.0], [244.0, 249.0] ....}
		windows = []
		labels = []

		for item in ra_annotations:
			if item[1] - item[0] >= 5:
				for i in range(int(item[1]) - int(item[0]) - 4):
					windows.append([item[0] + i, item[0] + i + 5])
					if item[2] != 0:
						labels.append(1)
					else:
						labels.append(item[2])

		#print ra_annotations, duration
		#print windows
		#print len(labels)


		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
		S = librosa.power_to_db(S, ref=np.max) + 80

		#[Fs, x] = audioBasicIO.read_audio_file(audio_filename)
		#print(Fs, x)
		F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr)
		F = F[:, 0::2]

		image_windows = []
		feature_windows = []
		for item in windows:
			spec = S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)]
			F_window = F[:, int(item[0]) : int(item[1])]
			F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
			image_windows.append(spec)
			feature_windows.append(F_feature)
		#image_windows = np.asarray(image_windows)

		#print image_windows.shape #(number of windows, n_mels, 5 * sr / hop_length)

		all_labels.extend(labels)
		all_data.extend(image_windows)
		all_feature_data.extend(feature_windows)

	all_data = np.asarray(all_data)
	all_feature_data = np.asarray(all_feature_data)
	print(all_data.shape) #(number of windows, n_mels, 5 * sr / hop_length)
	print(Counter(all_labels))
	#import pickle
	#pickle.dump(all_data, open( "save.p", "wb" ) )


	all_data = all_data.reshape(all_data.shape[0], img_rows * img_cols)
	all_data = np.concatenate((all_data, all_feature_data), axis = 1)

	rus = RandomUnderSampler(random_state=42)
	# all_data, all_labels = rus.fit_resample(all_data, all_labels)
	if len(np.unique(all_labels)) > 1:
		all_data, all_labels = rus.fit_resample(all_data, all_labels)
	else:
		print("Warning: all_labels contains only one class. Skipping resampling.")

	#idx = np.random.choice(np.arange(len(all_data)), len(all_data) // 2, replace=False)
	#all_data = all_data[idx, :]
	#all_labels = np.asarray(all_labels)[idx]

	all_feature_data = all_data[:, img_rows * img_cols : ]
	all_data = all_data[:, : img_rows * img_cols]


	all_data = all_data.reshape(all_data.shape[0], img_rows, img_cols, 1)
	all_data = all_data.astype('float32')
	all_data /= 80.0

	print('all_data shape:', all_data.shape)
	print(all_data.shape[0], 'train samples')
	print(Counter(all_labels))

	# session_num = test_folder[1:]
	model_output_folder = '.trained/'
	model = CustomPyTorchModel(num_classes=2)
	model.load_state_dict(torch.load(model_output_folder + 'pics_alex_noflip_torch_distress.h5'))
	model.eval()  # Set the model to evaluation mode

	# Convert all_data to PyTorch tensor
	all_data_tensor = torch.tensor(all_data, dtype=torch.float32).permute(0, 3, 1, 2)	
	from torch.utils.data import TensorDataset, DataLoader

	# Assuming all_data_tensor is your input data and doesn't need labels for prediction
	dataset = TensorDataset(all_data_tensor)
	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction

	# Disable gradient computation for efficiency and to reduce memory usage during inference
	with torch.no_grad():
		all_predictions = []
		for inputs in data_loader:
			inputs = inputs[0]  # DataLoader wraps each batch in a tuple

			# If you have a GPU, move the data to the GPU
			# inputs = inputs.to('cuda')
			
			outputs = model(inputs)
			
			# Convert outputs to probabilities; for example, using softmax for classification
			probabilities = torch.softmax(outputs, dim=1)
			
			all_predictions.extend(probabilities.cpu().numpy())

	# Convert list of arrays to a single NumPy array
	image_vector = np.vstack(all_predictions)
	svm_input = np.concatenate((image_vector, all_feature_data), axis = 1)
	from sklearn.svm import SVC
	clf = SVC(kernel = 'rbf', probability = True)
	clf.fit(svm_input, all_labels)
	from joblib import dump, load
	import os
	if not os.path.exists(model_out_folder):
		try:
			# Create the directory
			# The 'exist_ok' parameter is set to True, which will not raise an exception if the directory already exists.
			os.makedirs(model_out_folder, exist_ok=True)
			print("Directory '%s' created successfully" % model_out_folder)
		except OSError as error:
			print("Directory '%s' can not be created. Error: %s" % (model_out_folder, error))
	else:
		print("Directory '%s' already exists" % model_out_folder)


	dump(clf, model_out_folder + '/svm_noflip.joblib')
######################

data_folder = home_directory + "/data/deBarbaroCry/kyunghun-10min-data/"
label_folder = home_directory + "/data/deBarbaroCry/kyunghun-10min-label/"
test_folders = ['P30','P38']
test_folder = 'P30'
model_out_folder = '.trained'
real_label_folder = './ahsans_labels/'
real_label_folder = label_folder

def test_alex_svm(data_folder,label_folder,test_folders,model_out_folder,real_label_folder):
	from joblib import load
	session_num = '0'
	import os
	home_directory = os.path.expanduser("~")
	model_path = model_out_folder + '/svm_noflip.joblib'
	clf = load(model_path)

	saved_model = load_model(model_out_folder + '/pics_alex_noflip_ahsans' + session_num + '_distress.h5')



	# session_num = test_folder[1:]
	model_output_folder = '.trained/'
	model = CustomPyTorchModel(num_classes=2)
	model.load_state_dict(torch.load(model_output_folder + 'pics_alex_noflip_torch_distress.h5'))
	model.eval()  # Set the model to evaluation mode

	test_episodes = []
	episodes = [file for file in os.listdir(data_folder + test_folder) if file.endswith('.wav')]
	for episode in episodes:
		test_episodes.append(test_folder + '/' + episode[:-4])


	# real_label_folder = './ahsans_labels/'
	#get 5s windows with 1s overlap

	all_groundtruth = []
	all_predictions = []

	for ind, test_episode in enumerate(test_episodes):
		audio_filename = data_folder + test_episode + ".wav"
		annotation_filename_ra =  real_label_folder + test_episode + ".csv"
		annotation_filename_filtered = label_folder + test_episode + ".csv"

		y, sr = librosa.load(audio_filename)
		duration = librosa.get_duration(y = y, sr = sr)

		previous = 0

		#ra_annotations: ['other', 'other', 'other', 'other', 'other', 'fuss' ....]
		ra_annotations = []
		with open(annotation_filename_ra, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if len(row) > 0:
					row[2] = label_to_num(row[2])
					if float(row[0]) - previous > 0:
						ra_annotations.extend([0] * int(float(row[0]) - previous))
				previous = float(row[1])
				ra_annotations.extend([1] * int(float(row[1]) - float(row[0])))
		if duration - previous > 0:
			ra_annotations.extend([0] * int(duration - previous))
		print(duration, len(ra_annotations))


		previous = 0
		filtered_annotations = []
		with open(annotation_filename_filtered, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if float(row[0]) - previous > 0:
					filtered_annotations.extend([0] * int(float(row[0]) - previous))
				previous = float(row[1])
				filtered_annotations.extend([1] * int(float(row[1]) - float(row[0])))
		if duration - previous > 0:
			filtered_annotations.extend([0] * int(duration - previous))
		print(duration, len(filtered_annotations))


		#windows = [[0, 5], [1, 6],......]
		windows = []
		for i in range(0, int(duration) - 4):
			windows.append([i, i + 5])
		print(len(windows))


		#y, sr = librosa.load(audio_filename)
		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
		S = librosa.power_to_db(S, ref=np.max) + 80
		#print S.shape

		#[Fs, x] = audioBasicIO.read_audio_file(audio_filename)
		F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr)
		F = F[:, 0::2]

		image_windows = []
		feature_windows = []
		for item in windows:
			image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])
			F_window = F[:, item[0] : item[1]]
			F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
			feature_windows.append(F_feature)

		image_windows = np.asarray(image_windows)
		feature_windows = np.asarray(feature_windows)
		image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1)
		image_windows = image_windows.astype('float32')
		image_windows /= 80.0

		print(image_windows.shape) #(number of windows, n_mels, 5 * sr / hop_length, 1)

		# Convert all_data to PyTorch tensor
		all_data_tensor = torch.tensor(image_windows, dtype=torch.float32).permute(0, 3, 1, 2)	
		from torch.utils.data import TensorDataset, DataLoader

		# Assuming all_data_tensor is your input data and doesn't need labels for prediction
		dataset = TensorDataset(all_data_tensor)
		data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction

		# Disable gradient computation for efficiency and to reduce memory usage during inference
		with torch.no_grad():
			# all_predictions = []
			for inputs in data_loader:
				inputs = inputs[0]  # DataLoader wraps each batch in a tuple

				# If you have a GPU, move the data to the GPU
				# inputs = inputs.to('cuda')
				
				outputs = model(inputs)
				
				# Convert outputs to probabilities; for example, using softmax for classification
				probabilities = torch.softmax(outputs, dim=1)
				
				all_predictions.extend(probabilities.cpu().numpy())

		image_features = np.vstack(all_predictions)
		print(image_features.shape)
		svm_test_input = np.concatenate((image_features, feature_windows), axis = 1)
		predictions = clf.predict(svm_test_input)
		print("pred shape: ", predictions.shape)

		for ind, val in enumerate(filtered_annotations):
			if val >= 1:
				min_ind = max(ind - 4, 0)
				max_ind = min(len(predictions), ind + 1)
				#print(Counter(predictions[min_ind : max_ind]).most_common(1))
				#filtered_annotations[ind] = Counter(predictions[min_ind : max_ind]).most_common(1)[0][0]
				if sum(predictions[min_ind : max_ind]) >= 1:
					filtered_annotations[ind] = 1
				else:
					filtered_annotations[ind] = 0

		print(len(filtered_annotations), len(ra_annotations))

		timed_filted = np.stack([np.arange(len(filtered_annotations)), filtered_annotations], axis = 1)
		timed_filted = combineIntoEvent(timed_filted, 5 )
		timed_filted = whatIsAnEvent(timed_filted, 5 )

		filtered_annotations = timed_filted[:, 1]

		print(len(filtered_annotations), len(ra_annotations))

		all_groundtruth.extend(ra_annotations)
		all_predictions.extend(filtered_annotations)

	print(confusion_matrix(all_groundtruth, all_predictions))
	print(accuracy_score(all_groundtruth, all_predictions))
	print(classification_report(all_groundtruth, all_predictions, target_names=['other', 'distress']))

	'''
	plt.figure(figsize=(12, 8))
	plt.subplot(2, 1, 1)
	plt.plot(np.arange(len(all_predictions)), all_predictions)
	plt.subplot(2, 1, 2)
	plt.plot(np.arange(len(all_groundtruth)), all_groundtruth)
	plt.savefig(str(test_folder) + '_alex.png')
		'''


