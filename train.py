import wave
import numpy as np
import tensorflow as tf
from keras import layers,utils
import os

signalLenght = 16000
yesFilePath = "./yes"
noFilePath = "./no"

def extractArrayFromFileName(path):
    wav_obj = wave.open(path, 'rb')

    # extract the sound
    n_samples = wav_obj.getnframes()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    signal_pading = signalLenght - len(signal_array)
    if signal_pading>0: # add values to get to to signal lenght
        signal_array = np.pad(signal_array,signal_pading,constant_values=0)
    return signal_array


# transform the data

values = []
label = []
for path in [yesFilePath,noFilePath]:
    for file in (os.listdir(path)):
        signal = (extractArrayFromFileName(path+"/"+file))
        if len(signal)==signalLenght: #exclude samples that are too long
            values.append(signal)
            label.append(path != yesFilePath)

values = np.array(values)
label = np.array(label)
label = utils.to_categorical(label, num_classes=2)

# # Step 2: Build the 1D Neural Network
model = tf.keras.Sequential([
    layers.Input(shape=(signalLenght,1)),  # num_features is the size of your feature representation
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 output classes (yes and no)
])

# Step 3: Compile and Train the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(values, label, epochs=3)#, validation_data=(Y_values[33:40], N_values[33:40]))

# Step 4: Save the weights
model.save_weights('my_cnn_weights.h5')
