import torch
import torch.nn as nn
import torch.optim as optim

# for the CNN
import numpy as np
import matplotlib
matplotlib.use('Agg')
import librosa
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
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomPyTorchModel(nn.Module):
    def __init__(self, num_classes=10):  # Adjust num_classes as per your requirement
        super(CustomPyTorchModel, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0)  # Input shape (1, 225, 225)
        self.bn1 = nn.BatchNorm2d(96)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(384)
        
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Dynamically calculate the flattened size
        self.flattened_size = self._get_conv_output((225, 225))
        
       # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 4096)
        self.fc_bn1 = nn.BatchNorm1d(4096)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_bn2 = nn.BatchNorm1d(4096)
        
        self.fc3 = nn.Linear(4096, 1000)
        self.fc_bn3 = nn.BatchNorm1d(1000)
        
        self.fc_final = nn.Linear(1000, num_classes)  # Correctly updated for binary classification
	
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, 1, *shape))
            output_feat = self._forward_features(input)
            n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), 0.5)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), 0.5)
        x = F.dropout(F.relu(self.fc_bn3(self.fc3(x))), 0.5)
        x = self.fc_final(x)  # Output from the final fully connected layer
        return x

n_fft = 980
hop_length = 490
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
from os.path import expanduser
def train_alex(data_folder,label_folder,user_folders,model_output_folder):
	# home = expanduser("~")
	# data_folder = home + '/data/deBarbaroCry/kyunghun-10min-data/'
	# label_folder = home + '/data/deBarbaroCry/kyunghun-10min-label/'
	# user_folders = ['P38,'P31','P32','P30']
	episodes = []
	all_data = []
	all_labels = []
	import os
	# user_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
	for user_folder in user_folders:
		if user_folder != test_folder:
			user_episodes = [file for file in os.listdir(data_folder + user_folder) if file.endswith('.wav')]
			for user_episode in user_episodes:
					episodes.append(user_folder + '/' + user_episode[:-4])
	# # user_folders = ['P38,'P31','P32','P30']
	
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
		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
		S = librosa.power_to_db(S, ref=np.max) + 80
		image_windows = []
		for item in windows:
			image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])
		all_labels.extend(labels)
		all_data.extend(image_windows)
	all_data = np.asarray(all_data)
	print(all_data.shape) #(number of windows, n_mels, 5 * sr / hop_length)
	print(Counter(all_labels))
	
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
	# rus = RandomUnderSampler(random_state=42)
	# x_train, y_train = rus.fit_resample(x_train, y_train)
	# print(Counter(y_train))
	# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	# Use SMOTE to oversample the minority class
	smote = SMOTE(random_state=42)
	x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
	x_train,y_train = x_train_resampled,y_train_resampled
	
	
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
	
	num_epochs = 50
	import torch.nn.functional as F
	from torch.utils.data import TensorDataset, DataLoader
	# Convert your training data to PyTorch tensors
	# x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
	
	# Then use this DataLoader in your training loop
	# Assuming x_train is your input data and y_train is one-hot encoded
	y_train_indices = np.argmax(y_train, axis=1)  # Convert one-hot to class indices if needed

	x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)  # Reshape to [N, C, H, W]
	y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)

	# Create a dataset and dataloader for training
	train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
	train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  # Adjust batch size as necessary

	model = CustomPyTorchModel(num_classes=2)
	criterion = torch.nn.CrossEntropyLoss()  # Suitable for classification
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Learning rate can be adjusted

	num_epochs = 10  # Number of epochs to train for

	for epoch in range(num_epochs):
		model.train()  # Set the model to training mode
		running_loss = 0.0

		for inputs, labels in train_loader:
			optimizer.zero_grad()  # Zero the parameter gradients

			# Forward pass
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			# Backward pass and optimize
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		# Print average loss for the epoch
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
	
	model_output_folder = '.trained/'
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
	# model.save(model_output_folder + 'pics_alex_noflip_torch_distress.h5')
	torch.save(model.state_dict(), model_output_folder + 'pics_alex_noflip_torch_distress.h5')
	
	# When load this model:
	# Define the model architecture (should be the same as the saved model)
	# model = TheModelClass(*args, **kwargs)
	# model = CustomPyTorchModel(num_classes=2)

	# Load the state dict back into the model
	# model.load_state_dict(torch.load(model_output_folder + 'pics_alex_noflip_torch_distress.h5'))

	# Set the model to evaluation mode if you are not planning to train it further
	# model.eval()

	# model.save(model_output_path)
	#model = load_model('pics_alex_' + session_num + '_distress.h5')
	
