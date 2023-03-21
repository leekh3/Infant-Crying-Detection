import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the trained model
model = load_model('infant_crying_model.h5')

# Data preprocessing
data_gen = ImageDataGenerator(rescale=1./255)

# Load test data
test_data = data_gen.flow_from_directory('test_dir', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Make predictions on test data
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Print a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:")
print(confusion_matrix(test_data.classes, predicted_classes))

# Print a classification report
print("Classification Report:")
print(classification_report(test_data.classes, predicted_classes, target_names=test_data.class_indices.keys()))
