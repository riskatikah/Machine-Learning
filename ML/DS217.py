import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlxtend.frequent_patterns import apriori, association_rules

# data
data = {
    'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'transaction_id': [101, 101, 102, 201, 202, 301, 301, 302, 303],
    'product': ['bread', 'milk', 'butter', 'milk', 'bread', 'butter', 'milk', 'bread', 'jam']
}
df = pd.DataFrame(data)

# association
def generate_product_associations(customer_id):
    customer_data = df[df['customer_id'] == customer_id]
    basket = customer_data.groupby(['transaction_id', 'product']).size().unstack(fill_value=0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# RNN
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['product'])
df['product_id'] = df['product'].apply(lambda x: tokenizer.word_index.get(x, 0))
transactions = df.groupby('customer_id')['product_id'].apply(list)

max_len = max(transactions.apply(len))
sequences = pad_sequences(transactions, maxlen=max_len, padding='post')
X, Y = sequences[:, :-1], sequences[:, 1:]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_len-1),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, verbose=1)

# prediction
def predict_next_purchase(customer_id):
    if customer_id not in transactions:
        return None
    
    sequence = transactions[customer_id]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='post')
    prediction = model.predict(sequence)
    predicted_token_id = np.argmax(prediction[0, -1])
    return tokenizer.index_word.get(predicted_token_id, "[UNK]")

# test
customer_id = 3
print("Product Associations:")
print(generate_product_associations(customer_id))
print("Predicted Next Purchase:", predict_next_purchase(customer_id))