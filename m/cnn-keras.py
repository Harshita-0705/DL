import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

data = []
labels = []

folders = ["circles", "squares"]

for i, folder in enumerate(folders):
    path = "dataset/" + folder
    for img in os.listdir(path):
        img = cv2.imread(path + "/" + img)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data.append(img)
        labels.append(i)

data = np.array(data)/255.0
labels = np.array(labels)

data = data.reshape(-1,64,64,1)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True
)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test,y_test)
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.show()