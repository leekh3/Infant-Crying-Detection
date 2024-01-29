# for the CNN
import numpy as np
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

# from skimage import feature

from sklearn.preprocessing import minmax_scale

from scipy.signal import find_peaks

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score

from math import sqrt, pi, exp

from collections import Counter

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

import random

import h5py

import copy

from tensorflow import keras

import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json, load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras import backend as K

n_fft = 980

# 1764

hop_length = 490

# 882

n_mels = 225

img_rows, img_cols = 225, 225

batch_size = 64

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

# data_folder = './ahsans/'
# label_folder = './ahsans_labels/'
from os.path import expanduser
import os

# data_folder = home + '/data/deBarbaroCry/kyunghun-10min-data/'
# label_folder = home + '/data/deBarbaroCry/kyunghun-10min-label/'
# test_folders = ['P31']
# real_label_folder = './ahsans_labels/'
def train_alex(data_folder,label_folder,test_folders,real_label_folder,model_output_folder):
	import numpy as np
	import matplotlib as plt
	import matplotlib
	# home = expanduser("~")
	# data_folder = home + '/data/deBarbaroCry/kyunghun-10min-data/'
	# label_folder = home + '/data/deBarbaroCry/kyunghun-10min-label/'
	# test_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
	# test_folders = ['P31']

	'''

	test_folders.pop(test_folders.index('P30'))

	test_folders.pop(test_folders.index('P38'))



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

	test_folders.pop(test_folders.index('P31'))

	test_folders.pop(test_folders.index('P21'))

	test_folders.pop(test_folders.index('P29'))

	test_folders.pop(test_folders.index('P11'))

	test_folders.pop(test_folders.index('P25.6'))

	'''

	for test_folder in test_folders:

		print(test_folder)
		episodes = []
		all_data = []
		all_labels = []
		import os
		user_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]

		for user_folder in user_folders:

			if user_folder != test_folder:

				user_episodes = [file for file in os.listdir(data_folder + user_folder) if file.endswith('.wav')]

				for user_episode in user_episodes:

						episodes.append(user_folder + '/' + user_episode[:-4])


		#get 5s windows with 1s overlap

		for episode in episodes:

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

						if float(row[0]) - previous > 0:

							ra_annotations.append([float(previous), float(row[0]), 0])



						ra_annotations.append([float(row[0]), float(row[1]), int(row[2])])

						previous = float(row[0])

			if duration - previous > 0:

				ra_annotations.append([previous, float(duration), 0])

			#windows = {'other': [[243.0, 248.0], [244.0, 249.0] ....}

			windows = []

			labels = []



			for item in ra_annotations:

				if item[1] - item[0] >= 5:

					for i in range(int(item[1]) - int(item[0]) - 4):

						windows.append([item[0] + i, item[0] + i + 5])

						if item[2] == 1 or item[2] == 2:

							labels.append(1)

						else:

							labels.append(0)



			#print ra_annotations, duration

			#print windows

			#print len(labels)





			S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)

			S = librosa.power_to_db(S, ref=np.max) + 80

			#sigma_est = estimate_sigma(S, multichannel=False, average_sigmas=True)

			#S = denoise_wavelet(S, multichannel=False, convert2ycbcr=True,method='BayesShrink', mode='soft')

			#S = denoise_tv_chambolle(S, weight=0.1, multichannel=True)

			#print S.shape



			image_windows = []

			for item in windows:

				image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])

			#image_windows = np.asarray(image_windows)



			#print image_windows.shape #(number of windows, n_mels, 5 * sr / hop_length)



			all_labels.extend(labels)

			all_data.extend(image_windows)


		all_data = np.asarray(all_data)

		print(all_data.shape) #(number of windows, n_mels, 5 * sr / hop_length)

		print(Counter(all_labels))

		import pickle

		#pickle.dump(all_data, open("all_data.p", "wb" ) )

		#pickle.dump(all_labels, open("all_labels.p", "wb" ) )









		epochs = 50







		# the data, split between train and test sets

		#x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0, random_state=0)

		x_train, y_train = all_data, all_labels

		all_data, all_labels = None, None

		print(Counter(y_train))





		idx = np.random.choice(np.arange(len(x_train)), len(x_train), replace=False)

		x_train = x_train[idx, :]

		y_train = np.asarray(y_train)[idx]

		y_train = list(y_train)



		x_train = list(x_train)

		additional_labels = []

		for y_train_ind, y_train_val in enumerate(y_train):

			if y_train_val == 1:

				temp1 = copy.deepcopy(x_train[y_train_ind])

				temp2 = copy.deepcopy(x_train[y_train_ind])

				x_train.append(time_masking(temp1, tau = n_mels, time_masking_para = 10))

				#x_train.append(np.fliplr(temp2))

				#additional_labels.append(1)

				additional_labels.append(1)



		y_train.extend(additional_labels)

		additional_labels = None

		print(Counter(y_train))

		x_train = np.asarray(x_train)



		#upsample downsample for x_train, y_train

		x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)


		rus = RandomUnderSampler(random_state=42)

		x_train, y_train = rus.fit_resample(x_train, y_train)

		print(Counter(y_train))



		#sm = SMOTE(random_state=42)

		#x_train, y_train = sm.fit_resample(x_train, y_train)



		#ros = RandomOverSampler(random_state=42)

		#x_train, y_train = ros.fit_resample(x_train, y_train)

		#print(Counter(y_train))



		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

		#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

		input_shape = (img_rows, img_cols, 1)

















		x_train = x_train.astype('float32')

		#x_test = x_test.astype('float32')

		x_train /= 80.0

		#x_test /= 80.0



		print('x_train shape:', x_train.shape)

		print(x_train.shape[0], 'train samples')

		#print(x_test.shape[0], 'test samples')









		# convert class vectors to binary class matrices

		y_train = keras.utils.to_categorical(y_train, num_classes)

		#y_test = keras.utils.to_categorical(y_test, num_classes)







		#model = Sequential()

		#model.add(Conv2D(32, kernel_size=(7, 7), activation='relu',input_shape=input_shape))

		#model.add(BatchNormalization())



		#model.add(Conv2D(32, kernel_size = (7, 7), activation = 'relu'))

		#model.add(BatchNormalization())



		#model.add(MaxPooling2D(pool_size=(2, 2)))



		#model.add(Conv2D(64, (3, 3), activation='relu'))

		#model.add(BatchNormalization())

		#model.add(MaxPooling2D(pool_size=(2, 2)))

		#model.add(Dropout(0.25))



		#model.add(Conv2D(64, (3, 3), activation = 'relu'))

		#model.add(MaxPooling2D(pool_size = (2, 2)))



		#model.add(Flatten())

		#model.add(Dense(128, activation='relu'))

		#model.add(Dropout(0.5))

		#model.add(Dense(64, activation = 'relu'))

		#model.add(Dense(num_classes, activation='softmax'))



		#model.compile(loss=keras.losses.categorical_crossentropy,

		#              optimizer=keras.optimizers.Adadelta(),

		#              metrics=['accuracy'])







		model = Sequential()

		model.add(Conv2D(96, kernel_size=(11, 11),input_shape=input_shape, strides = (4, 4), padding = 'valid'))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2), padding = 'valid'))





		model.add(Conv2D(256, kernel_size = (5, 5), strides = (1, 1), padding = 'valid'))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2), padding = 'valid'))





		model.add(Conv2D(384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

		model.add(BatchNormalization())

		model.add(Activation('relu'))



		model.add(Conv2D(384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

		model.add(BatchNormalization())

		model.add(Activation('relu'))



		model.add(Conv2D(256, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2), padding = 'valid'))





		model.add(Flatten())

		model.add(Dense(4096))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(Dropout(0.5))



		model.add(Dense(4096))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(Dropout(0.5))



		model.add(Dense(1000))

		model.add(BatchNormalization())

		model.add(Activation('relu'))

		model.add(Dropout(0.5))



		model.add(Dense(num_classes, activation='softmax'))



		model.compile(loss=keras.losses.categorical_crossentropy,

					  optimizer = 'adam',

					  metrics=['accuracy'])





		print(model.summary())



		history = model.fit(x_train, y_train,

				  batch_size=batch_size,

				  epochs=epochs,

				  #validation_data=(x_test, y_test),

				  verbose=1)

		#score = model.evaluate(x_test, y_test, verbose=1)

		#y_pred = model.predict(x_test, batch_size=batch_size, verbose=1, steps=None)



		#con_mat = confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

		#print(con_mat)







		#print('Test loss:', score[0])

		#print('Test accuracy:', score[1])





		#plt.plot(history.history['acc'])

		#plt.plot(history.history['val_acc'])

		#plt.title('model accuracy')

		#plt.ylabel('accuracy')

		#plt.xlabel('epoch')

		#plt.legend(['train', 'test'], loc='upper left')

		#plt.savefig('accuracy.png')

		# summarize history for loss

		#plt.plot(history.history['loss'])

		#plt.plot(history.history['val_loss'])

		#plt.title('model loss')

		#plt.ylabel('loss')

		#plt.xlabel('epoch')

		#plt.legend(['train', 'test'], loc='upper left')

		#plt.savefig('loss.png')





		session_num = test_folder[1:]

		import os
		if not os.path.exists(model_output_folder):
			try:
				# Create the directory
				# The 'exist_ok' parameter is set to True, which will not raise an exception if the directory already exists.
				os.makedirs(model_output_folder, exist_ok=True)
				print("Directory '%s' created successfully" % model_output_folder)
			except OSError as error:
				print("Directory '%s' can not be created. Error: %s" % (model_output_folder, error))
		else:
			print("Directory '%s' already exists" % model_output_folder)

		model.save(model_output_folder + 'pics_alex_noflip_ahsans' + session_num + '_distress.h5')
		# model.save(model_output_path)

		#model = load_model('pics_alex_' + session_num + '_distress.h5')





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

			if ind <= -1:

				ra_annotations = []

				with open(annotation_filename_filtered, 'r') as csvfile:

					csvreader = csv.reader(csvfile, delimiter=',')

					for row in csvreader:

						if float(row[0]) - previous > 0 and int(row[2]) <= 2:

							ra_annotations.append([float(row[0]), float(row[1]), int(row[2])])

				windows = []

				labels = []



				for item in ra_annotations:

					if item[1] - item[0] >= 5:

						for i in range(int(item[1]) - int(item[0]) - 4):

							windows.append([item[0] + i, item[0] + i + 5])

							labels.append(item[2])





				S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)

				S = librosa.power_to_db(S, ref=np.max) + 80



				image_windows = []

				for item in windows:

					image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])

				#image_windows = np.asarray(image_windows)



				image_windows = np.asarray(image_windows)

				image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1)

				image_windows = image_windows.astype('float32')

				image_windows /= 80.0



				#upsample downsample for x_train, y_train

				x_train = image_windows

				y_train = labels



				if len(set(y_train)) == 3:



					x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)

					rus = RandomUnderSampler(sampling_strategy={0: Counter(y_train)[0] // 3}, random_state=42)

					x_train, y_train = rus.fit_resample(x_train, y_train)

					x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)



					y_train = keras.utils.to_categorical(y_train, num_classes)





					model.fit(x_train, y_train, batch_size=batch_size, epochs=15, verbose=1)



			else:



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



				image_windows = []

				for item in windows:

					image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])

				image_windows = np.asarray(image_windows)

				image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1)

				image_windows = image_windows.astype('float32')

				image_windows /= 80.0



				print(image_windows.shape) #(number of windows, n_mels, 5 * sr / hop_length, 1)



				predictions = list(np.argmax(model.predict(image_windows, batch_size=batch_size, verbose=1, steps=None), axis = 1)) ## Understanding of this model!!! (To-do: Kyunghun)

				print("Sum: ", sum(predictions))



				for ind, val in enumerate(filtered_annotations):

					if val >= 0:

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





		#plt.figure(figsize=(12, 8))

		#plt.subplot(2, 1, 1)

		#plt.plot(np.arange(len(all_predictions)), all_predictions)

		#plt.subplot(2, 1, 2)

		#plt.plot(np.arange(len(all_groundtruth)), all_groundtruth)

		#plt.savefig(str(test_folder) + '_alex.png')









