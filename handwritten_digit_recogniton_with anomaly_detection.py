import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your data (replace with your actual data loading code)
data = np.load("handwritten_digits.npz")
x_data = data["x"]
y_data = data["y"]

# Preprocess your data (replace with your actual preprocessing steps)
x_data = x_data / 255.0
y_data = to_categorical(y_data - 1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define your neural network model (replace with your actual model architecture)
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(9, activation="softmax"),
])

# Compile your model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train your model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate your model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")

# Predictions and anomaly detection
predictions = model.predict(x_test)
confidence_threshold = 0.4  # Adjust this threshold as needed

anomalies = []
for i in range(len(x_test)):
    confidence = np.max(predictions[i])
    if confidence < confidence_threshold:
        anomalies.append(i)

print(f"Detected {len(anomalies)} anomalies based on prediction confidence.")

# Plot anomalies (all at once)
num_anomalies_to_display = len(anomalies)
if num_anomalies_to_display > 0:
    fig, axes = plt.subplots(1, num_anomalies_to_display, figsize=(num_anomalies_to_display * 3, 3))
    for j, anomaly_idx in enumerate(anomalies):
        anomaly_image = x_test[anomaly_idx]
        anomaly_label = np.argmax(y_test[anomaly_idx])
        predicted_label = np.argmax(predictions[anomaly_idx])
        axes[j].imshow(anomaly_image, cmap="gray")
        axes[j].set_title(f"Anomaly {j+1}\nT.L: {anomaly_label + 1}, P.L: {predicted_label + 1}")
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No anomalies detected.")
