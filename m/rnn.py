import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
texts = [
    "I love this movie",
    "This film is very good",
    "I hate this movie",
    "This film is very bad"
]

labels = [1, 1, 0, 0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=5)

X = np.array(padded)
y = np.array(labels)
model = Sequential()
model.add(Embedding(input_dim=50, output_dim=8, input_length=5))
model.add(SimpleRNN(16))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20)
test_text = ["I love this film"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=5)

prediction = model.predict(test_pad)

print("Prediction:", prediction)