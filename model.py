import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Cropping2D

def generate_mirror_data(image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped

def load_image(p):
    bgr = cv2.imread(str(p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def load_data(log_dir, camera='center'):
    df_log = pd.read_csv(log_dir / 'driving_log.csv')
    
    plt.figure(figsize=(20, 15))
    plt.plot(df_log['steering'])
    plt.savefig('writeup-images/{log_dir.stem}-steering-before.jpg'.format(**locals()))

    # Smoothing
    smoothed = df_log['steering'].shift(-5).rolling(10).mean().fillna(method='bfill')
    plt.figure(figsize=(20, 15))
    plt.plot(smoothed)
    plt.savefig('writeup-images/{log_dir.stem}-steering-after.jpg'.format(**locals()))
    
    if camera == 'left':
        smoothed += 0.2
    elif camera == 'right':
        smoothed -= 0.2

    images = [load_image(log_dir / p.strip()) for p in df_log[camera]]
    measurements = list(smoothed)

    return images, measurements

def generate_model(dropout_rate=0.2):
    # Input planes
    model = Sequential()
    
    # Normalized input planes
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    
    # Convolutional feature map
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional feature map
    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional feature map
    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional feature map
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional feature map
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Dropout(dropout_rate))

    # Flatten
    model.add(Flatten())
    
    # Fully-connected layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Fully-connected layer
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Fully-connected layer
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    
    return model

if __name__ == '__main__':
    #%% Load data
    data_dir = Path('./data')
    log_dirs = [
        data_dir / 'track1' / 'default',
        data_dir / 'track1' / 'ccw-lap1',
        data_dir / 'track1' / 'cw-lap1',
        data_dir / 'track1' / 'recovery',
        data_dir / 'track1' / 'curve',
    ]
    
    all_images = []
    all_measurements = []
    
    for log_dir in log_dirs:
        for camera in ['center', 'left', 'right']:
            images, measurements = load_data(log_dir, camera)
            all_images += images
            all_measurements += measurements
    
    #%% Data Augmentation
    for i in range(len(all_images)):
        image_flipped, measurements_flipped = generate_mirror_data(all_images[i], all_measurements[i])
        all_images.append(image_flipped)
        all_measurements.append(measurements_flipped)
    
    #%% Setup model
    X_train = np.array(all_images)
    y_train = np.array(all_measurements)

    model = generate_model()
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)
    
    #%% Save model
    model.save('model.h5')

    #%% Visualize learning history
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('writeup-images/learning-history.jpg')
