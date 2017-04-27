
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, Lambda, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

nb_epoch = 50
learning_rate = 0.0001
side_correction = 0.5 # adjusted steering measurements for the side camera images
curve_multiplier = 1.5 # for steep curve

data_path = './data'
lines = []
with open(data_path+'/driving_log_1.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)



def generator(samples, batch_size=36, flip=False, side=False):
    """
    flip: flipping images
    side: using left and right images for recover back.
    """

    num_samples = len(samples)
    print('@batch_size=', batch_size, flip, side, num_samples, flush=True)
    if side:
        if flip:
            batch_size = batch_size // 6
        else:
            batch_size = batch_size // 3
    elif flip:
        batch_size = batch_size // 2

    while 1:
        sklearn.utils.shuffle(samples)
        sample_count = 0
        print('new epoch', batch_size, flush=True)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_path+'/IMG/'+batch_sample[0].split('/')[-1]
                center_image = plt.imread(name)
                #center_iamge = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3]) * curve_multiplier 
                images.append(center_image)
                angles.append(center_angle)

                if side:
                    # create adjusted steering measurements for the side camera images
                    correction = side_correction 
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction

                    # read in images from center, left and right cameras
                    name = data_path+'/IMG/'+batch_sample[1].split('/')[-1]
                    img_left = cv2.imread(name) 
                    name = data_path+'/IMG/'+batch_sample[1].split('/')[-1]
                    img_right = cv2.imread(name)

                    # add images and angles to data set
                    images.extend([img_left, img_right])
                    angles.extend([steering_left, steering_right])

                    if flip:
                        images.append(cv2.flip(img_left, 1))
                        angles.append(steering_left * -1.0)

                        images.append(cv2.flip(img_right, 1))
                        angles.append(steering_right * -1.0)

                        images.append(cv2.flip(center_image, 1))
                        angles.append(center_angle * -1.0)

                elif flip:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            sample_count += len(X_train)
            print('batch_count=', len(X_train), 'total_size=', sample_count, flush=True)
            yield sklearn.utils.shuffle(X_train, y_train)



def net_model():
    model = Sequential()

    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    #"""
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    #"""
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


def model_run(model, side=False, flip=False):
    train_generator = generator(train_samples, batch_size=6*500, flip=flip, side=side)
    validation_generator = generator(validation_samples, batch_size=6*300) #36)

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    
    samples_per_epoch = len(train_samples)
    if side:
        if flip:
            samples_per_epoch = len(train_samples) * 6
        else:
            samples_per_epoch = len(train_samples) * 3
    elif flip:
        samples_per_epoch = len(train_samples) * 2
    print('original training samples=%d -> augmented samples_per_epoch=%d' 
        %(len(train_samples), samples_per_epoch))
    print('original validation samples=%d' %(len(validation_samples)))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history_object = model.fit_generator(train_generator, 
                    samples_per_epoch=samples_per_epoch,
                    validation_data = validation_generator, 
                    nb_val_samples=len(validation_samples),
                    nb_epoch=nb_epoch, verbose=2,
                    callbacks=[early_stopping])
    model.save('model.h5')
    return history_object

history_object = model_run(net_model(), side=True, flip=True)
print(history_object.history.keys())

plt.subplot(1,2, 1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squeared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.subplot(1,2, 2)
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.legend(['training accu', 'validation accu'], loc='upper right')
plt.show()






# test
# launch simulator. python drive.py model.h5

# 2,3 laps of center, 1 of recovery, 1 of curving

