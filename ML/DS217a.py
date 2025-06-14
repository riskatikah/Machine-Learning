import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love deep learning",
    "deep learning is amazing"
]

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)
max_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

X, Y = sequences[:, :-1], sequences[:, 1:]

# Build RNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=10, input_length=max_len - 1),
    tf.keras.layers.LSTM(64, return_sequences=True),  
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, verbose=1)

# Predict Next Word
def predict_next_word(text):
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = pad_sequences([tokens], maxlen=max_len - 1, padding='post')
    prediction = model.predict(tokens)[0, -1]
    
# distribution probability
    for word, index in tokenizer.word_index.items():
        print(f"{word}: {prediction[index - 1]:.4f}")
    
    predicted_token_id = np.argmax(prediction)

    return tokenizer.index_word.get(predicted_token_id, "[UNK]")

# Test
input_text = "deep"
predicted_word = predict_next_word(input_text)
print(f"Predicted next word: {predicted_word}")
print("Tokenizer index-word mapping:", tokenizer.index_word)

