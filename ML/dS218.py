import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np

# data
texts = [
    "I love this product", 
    "This is the worst experience", 
    "Absolutely fantastic!", 
    "I hate it", 
    "Not bad but not great", 
    "Could be better", 
    "Best purchase ever!"
]
labels = np.array([1, 0, 1, 0, 1, 0, 1])  # 1 = Positive, 0 = Negative sentiments

# Tokenizer process
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="tf")

# create model using BERT
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

optimizer = Adam(learning_rate=5e-5)
loss = SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(
    x=encoded_inputs["input_ids"], 
    y=labels, 
    epochs=3,
    batch_size=2, 
    verbose=1
)

# Test
new_texts = ["I absolutely love it", "This is terrible"]
new_inputs = tokenizer(new_texts, padding=True, truncation=True, max_length=32, return_tensors="tf")

predictions = model.predict(new_inputs["input_ids"]).logits
predicted_labels = np.argmax(predictions, axis=1)

for text, label in zip(new_texts, predicted_labels):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"{text} -> {sentiment}")
