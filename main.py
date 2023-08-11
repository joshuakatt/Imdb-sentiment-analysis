# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, GRU
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=30000)

# Pre-process data
train_data = pad_sequences(train_data, maxlen=350, padding='post')
test_data = pad_sequences(test_data, maxlen=350, padding='post')

# Pre-processing GloVe embeddings to be used in the dataset.

embedding_index = {}

with open('GloVe-embedding/glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word]=coeffs

# Creating the embedding matric specifically for the Imdb dataset

word_index = imdb.get_word_index()
embedding_dim =300
# populating a zero matrix
embedding_matrix = np.zeros((30000, embedding_dim))

# initializing values

for word, i in word_index.items():
    if i < 30000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Model definition
model = Sequential([
    Embedding(input_dim=30000, output_dim=300, input_length=350, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
              trainable=True),
    GRU(32, dropout = 0.3, recurrent_dropout = 0.3, return_sequences= False),
    Dense(16, activation='relu'),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

# Model compilation
adam_optimizer = Adam(lr=0.01)

# Compile the model
model.compile(
    optimizer=adam_optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    train_data, train_labels,
    epochs=10,
    batch_size=512,
    validation_split=0.2
)

# Model evaluation
test_loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy : {accuracy*100:.2f}%"). 