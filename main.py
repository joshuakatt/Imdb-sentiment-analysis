# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Pre-process data
train_data = pad_sequences(train_data, maxlen=350, padding='post')
test_data = pad_sequences(test_data, maxlen=350, padding='post')

# Model definition
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=350),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    train_data, train_labels,
    epochs=9,
    batch_size=512,
    validation_split=0.2
)

# Model evaluation
test_loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy : {accuracy*100:.2f}%")