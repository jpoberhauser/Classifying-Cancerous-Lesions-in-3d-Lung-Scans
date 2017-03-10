import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
################
################
DATA_DIR = utils.get_data_dir()

patients = os.listdir(DATA_DIR + 'sample_images/')
labels_df = pd.read_csv(DATA_DIR + 'stage1_labels.csv')

labeled_patients = list(set(patients) & set(labels_df["id"]))
labels_df = labels_df[labels_df['id'].isin(patients)]
################
################
import preprocessing
reload(preprocessing)
from preprocessing import preprocess
from multiprocessing import Pool

################
################

data_pre = '/data/kaggle/data/stage1-preprocessed-medium/'
patients_npy = os.listdir(data_pre)
main_dir = '/data/kaggle/data/'
labels_df = pd.read_csv(main_dir + 'stage1_labels.csv')
labeled_patients = list(set(labels_df["id"]))
#Make sure ordering is correct
#Make sure ordering is correct
cancer_labels = [int(labels_df.cancer[labels_df.id == i]) for i in labeled_patients]
train_patients = [list(labels_df.id[labels_df.id == i])[0] for i in labeled_patients]
cancer_labels = cancer_labels[0:1177] + cancer_labels[1178:]
train_patients.remove(train_patients[1177])

#now read in pre-processed and make big matrix
single_patient = np.load('{}{}.npy'.format(data_pre, train_patients[0]))
dim1, dim2, dim3 =  single_patient.shape[0], single_patient.shape[1], single_patient.shape[2]
arr = np.empty(shape=(len(train_patients), dim1, dim2, dim3))
print len(cancer_labels), len(train_patients)


for p in [x for x in range(len(train_patients)) if x != 1177]:
    arr[p,:,:] = np.load(data_pre + train_patients[p] + ".npy")
#split into train/validation
train_percentage = .8
splitter = int(round(len(arr)*train_percentage,0))
#cancer_labels = np_utils.to_categorical(cancer_labels, 2)
train_inputs, train_labels = np.expand_dims(np.array(arr[0:splitter]),1), np.array(cancer_labels[0:splitter])
validation_inputs, validation_labels = np.expand_dims(np.array(arr[splitter:]),1), np.array(cancer_labels[splitter:])

################################################
################################################
#y_test = np_utils.to_categorical(y_test, 2)
arr.shape

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution3D, MaxPooling3D, Flatten, Dense, Dropout, Input
from keras import backend as K
K.set_image_dim_ordering('th')


################################################
################################################
nb_classes= 2
data_pre = '/data/kaggle/data/stage1-preprocessed/'
patients_npy = os.listdir(data_pre)
main_dir = '/data/kaggle/data/'
labels_df = pd.read_csv(main_dir + 'stage1_labels.csv')
labeled_patients = list(set(labels_df["id"]))

#Make sure ordering is correct
cancer_labels = [int(labels_df.cancer[labels_df.id == i]) for i in labeled_patients]
train_patients = [list(labels_df.id[labels_df.id == i])[0] for i in labeled_patients]
cancer_labels = cancer_labels[0:1177] + cancer_labels[1178:]
train_patients.remove(train_patients[1177])

#now read in pre-processed and make big matrix
single_patient = np.load('{}{}.npy'.format(data_pre, train_patients[0]))
dim1, dim2, dim3 =  single_patient.shape[0], single_patient.shape[1], single_patient.shape[2]
arr = np.empty(shape=(len(train_patients), dim1, dim2, dim3))
print len(cancer_labels), len(train_patients)

for p in [x for x in range(len(train_patients)) if x != 1177]:
    arr[p,:,:] = np.load(data_pre + train_patients[p] + ".npy")
#split into train/validation
train_percentage = .8
splitter = int(round(len(arr)*train_percentage,0))
#cancer_labels = np_utils.to_categorical(cancer_labels, 2)
X_train, train_labels = np.array(arr[0:splitter]), np.array(cancer_labels[0:splitter])
Y_train = np_utils.to_categorical(train_labels, 2)

X_test, validation_labels = np.array(arr[splitter:]), np.array(cancer_labels[splitter:])
Y_test  = np_utils.to_categorical(validation_labels, 2)
################################################
################################################
rgb_eq, dim1, dim2 = X_train.shape[1:]

from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten

model = Sequential()
model.add(Convolution2D(rgb_eq, 3, 3, border_mode='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(rgb_eq, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))



model = Sequential()
model.add(Convolution2D(rgb_eq, 3, 3, border_mode='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(rgb_eq, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

nb_classes= 2
model.add(Convolution2D(rgb_eq*2, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(rgb_eq*2, 3, 3))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2312))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
metrics=['accuracy'])


model.fit(X_train, Y_train,
              batch_size=30,
              nb_epoch=5,
              validation_data=(X_test, Y_test),
shuffle=True)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
preds = model.predict(X_test)
