import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


df = pd.read_csv('train.csv')

x = df['sms']
y = df['label']

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

max_features = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x)
x_integer = tokenizer.texts_to_sequences(x)
x = pad_sequences(x_integer, maxlen = max_len)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=max_len),
    layers.LSTM(128, activation = 'tanh'),  # SimpleRNN or GRU
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.2)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------____-

model_loss, model_acc = model.evaluate(x_test, y_test)

print(f'model loss:{model_loss}')
print((f'model_accuracy:{model_acc}'))
