# Import necessary libraries
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the pre-trained model
pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the pre-trained model layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Add new layers to the pre-trained model
inputs = Input(shape=(None, None, 1))
x = Conv2D(1, (1, 1), padding='same')(inputs)
x = tf.image.grayscale_to_rgb(x)
x = pretrained_model(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
# outputs = x

# Create the model
model = Model(inputs, outputs)

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Load the audio file
# audio_file = 'paper/test.wav'
audio_file = '/Users/leek13/data/deBarbaroCry/P06/P06_4_notcry.wav'
signal, sr = librosa.load(audio_file, sr=22050, duration=5, res_type='kaiser_fast')

#########################################

# Reshape the signal for the model input
# signal = np.resize(signal,(len(signal),1))
spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128);print(spec.shape)
# spec = np.resize(spec,(128,128))
# spec = spec.reshape(128,128)
spec = spec[:128,:128]
spec = librosa.power_to_db(spec, ref=np.max);print(spec.shape)
spec = np.expand_dims(spec, axis=-1);print(spec.shape)
spec = np.expand_dims(spec, axis=0);print(spec.shape)

# Predict the probability of crying
prob_crying = model.predict(spec)
print(prob_crying)
# Classify the audio as crying or non-crying based on the probability threshold
threshold = 0.5
if prob_crying > threshold:
    print('The audio contains crying.')
else:
    print('The audio does not contain crying.')
