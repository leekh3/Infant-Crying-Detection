import librosa
import numpy as np
import glob
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa
import tensorflow as tf
# Convert VGGish model to Keras format

# find all folders
home = expanduser("~")
inFolders = glob.glob(home + "/data/LENA/*/AN1/segmented_2min/")
inFolder = inFolders[0]

# find the list of wav files.
inFiles = glob.glob(inFolder + '/*.wav')
inFiles.sort()

# Load audio file
for audio_file in inFiles:
    import tensorflow as tf
    import numpy as np
    import librosa

    # Download and extract the pre-trained sound classification model
    model_url = 'https://storage.googleapis.com/audioset/yamnet.h5'
    model_path = 'yamnet.h5'
    model_path = tf.keras.utils.get_file(model_path, model_url)

    # Download the class map
    class_map_url = 'https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv'
    class_map_path = 'yamnet_class_map.csv'
    class_map_path = tf.keras.utils.get_file(class_map_path, class_map_url)

    # Load the pre-trained sound classification model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load the class map
    class_map = {}
    with open(class_map_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            class_map[int(parts[0])] = parts[2]

    # Define the classification thresholds
    voice_threshold = 0.5

    # Define the audio file to classify
    # audio_file = 'path/to/audio/file.wav'

    # Load the audio file and extract its features
    audio_data, sampling_rate = librosa.load(audio_file, sr=16000, mono=True, duration=10)
    audio_features = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_fft=1024, hop_length=512,
                                                    n_mels=64)
    # Reshape the audio features to match the input shape of the model
    audio_features = np.reshape(audio_features, (1, audio_features.shape[0], audio_features.shape[1], 1))

    # Use the pre-trained model to predict the classes of the audio file
    predictions = model.predict(audio_features)
    # Get the index of the top-scoring class
    top_class_index = np.argmax(predictions)
    # Get the name of the top-scoring class
    top_class_name = class_map[top_class_index]
    # Get the probability of the top-scoring class
    top_class_prob = predictions[0, top_class_index]
    # Classify the audio based on the classification threshold
    if top_class_name == 'Speech' and top_class_prob > voice_threshold:
        print('Human voice detected!')
    else:
        print('No human voice detected.')