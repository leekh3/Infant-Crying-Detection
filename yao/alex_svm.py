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

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K




n_fft = 980
#1764
hop_length = 490
#882
n_mels = 225
img_rows, img_cols = 225, 225
batch_size = 128
num_classes = 2





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
	if input_label == 'other':
		return 0
	elif input_label == 'fuss':
		return 1
	elif input_label == 'cry':
		return 2
	elif input_label == 'scream':
		return 3
	else:
		return 4



data_folder = './ahsans/'
label_folder = './ahsans_labels_filtered/'

test_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
test_folders = ['P31']

'''
test_folders.pop(test_folders.index('P15'))
test_folders.pop(test_folders.index('P34'))
test_folders.pop(test_folders.index('P23'))
test_folders.pop(test_folders.index('P19'))
test_folders.pop(test_folders.index('P33'))
test_folders.pop(test_folders.index('P26.1'))
test_folders.pop(test_folders.index('P6.1'))
test_folders.pop(test_folders.index('P36'))
test_folders.pop(test_folders.index('P4'))
test_folders.pop(test_folders.index('P27.1'))
test_folders.pop(test_folders.index('P20'))
test_folders.pop(test_folders.index('P17'))
test_folders.pop(test_folders.index('P7'))
test_folders.pop(test_folders.index('P32'))
test_folders.pop(test_folders.index('P22'))
test_folders.pop(test_folders.index('P12'))
test_folders.pop(test_folders.index('P14'))
'''



for test_folder in test_folders:
	print(test_folder)

	episodes = []
	all_data = []
	all_labels = []
	all_feature_data = []
	user_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
	#user_folders = ['P27.1']
	for user_folder in user_folders:
		if user_folder != test_folder:
			user_episodes = [file for file in os.listdir(data_folder + user_folder) if file.endswith('.wav')]
			for user_episode in user_episodes:
					episodes.append(user_folder + '/' + user_episode[:-4])	


	#get 5s windows with 1s overlap
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
	all_data, all_labels = rus.fit_resample(all_data, all_labels)

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




	session_num = test_folder[1:]
	#model.save('pics_alex_svm_' + session_num + '_distress.h5')
	saved_model = load_model('pics_alex_noflip_ahsans' + session_num + '_distress.h5')

	model = Sequential()
	for layer in saved_model.layers[:-1]:
		model.add(layer)

	for layer in model.layers:
		layer.trainable = False

	model.summary()
	image_vector = model.predict(all_data, batch_size=batch_size, verbose=1, steps=None)

	print(image_vector.shape)

	svm_input = np.concatenate((image_vector, all_feature_data), axis = 1)
	from sklearn.svm import SVC
	clf = SVC(kernel = 'rbf', probability = True)
	clf.fit(svm_input, all_labels)
	from joblib import dump, load
	#dump(clf, 'svm_noaug50' + session_num + '.joblib')
	dump(clf, 'svm_noflip_ahsans_'+session_num + 'p.joblib')




	test_episodes = []
	episodes = [file for file in os.listdir(data_folder + test_folder) if file.endswith('.wav')]
	for episode in episodes:
		test_episodes.append(test_folder + '/' + episode[:-4])


	real_label_folder = './ahsans_labels/'
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
		

		image_features = model.predict(image_windows, batch_size=batch_size, verbose=1, steps=None)

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

		
