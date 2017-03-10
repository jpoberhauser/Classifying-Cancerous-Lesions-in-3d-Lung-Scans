import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils

DATA_DIR = utils.get_data_dir()

patients = os.listdir(DATA_DIR + 'sample_images/')
labels_df = pd.read_csv(DATA_DIR + 'stage1_labels.csv')

labeled_patients = list(set(patients) & set(labels_df["id"]))
labels_df = labels_df[labels_df['id'].isin(patients)]

#################
#################
#################

import preprocessing
reload(preprocessing)
from preprocessing import preprocess
from multiprocessing import Pool



from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution3D, MaxPooling3D, Flatten, Dense, Dropout, Input
from keras import backend as K

#################
#################
#################
#################

nb_classes= 2
from keras.utils import np_utils
data_pre = '/data/kaggle/data/stage1-preprocessed-medium/'
patients_npy = os.listdir(data_pre)
main_dir = '/data/kaggle/data/'
labels_df = pd.read_csv(main_dir + 'stage1_labels.csv')
labeled_patients = list(set(labels_df["id"]))
#Make sure ordering is correct
#Make sure ordering is correct
cancer_labels = [int(labels_df.cancer[labels_df.id == i]) for i in labeled_patients]
train_patients = [list(labels_df.id[labels_df.id == i])[0] for i in labeled_patients]
#cancer_labels = cancer_labels[0:1177] + cancer_labels[1178:]
#train_patients.remove(train_patients[1177])


#now read in pre-processed and make big matrix
single_patient = np.load('{}{}.npy'.format(data_pre, train_patients[0]))
dim1, dim2, dim3 =  single_patient.shape[0], single_patient.shape[1], single_patient.shape[2]
arr = np.empty(shape=(len(train_patients), dim1, dim2, dim3))
print len(cancer_labels), len(train_patients)


for p in [x for x in range(len(train_patients))]:
    arr[p,:,:] = np.load(data_pre + train_patients[p] + ".npy")


#split into train/validation
train_percentage = .8
splitter = int(round(len(arr)*train_percentage,0))
#cancer_labels = np_utils.to_categorical(cancer_labels, 2)
train_inputs, train_labels = np.expand_dims(np.array(arr[0:splitter]),1), np.array(cancer_labels[0:splitter])
validation_inputs, validation_labels = np.expand_dims(np.array(arr[splitter:]),1), np.array(cancer_labels[splitter:])



from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization



from theano.sandbox import cuda
cuda.use('gpu0')

print("TRAINING")

K.set_image_dim_ordering('th')
basic_conv3d = Sequential([
        Convolution3D(input_shape=(1, dim1, dim2, dim3), nb_filter=32,
                      kernel_dim1=7, kernel_dim2=7, kernel_dim3=7, activation='relu'),
        Dropout(0.2),    Convolution3D(nb_filter=32,
                                       kernel_dim1=7, kernel_dim2=7, kernel_dim3=7, activation='relu'),
        Dropout(0.2),    Convolution3D(nb_filter=32,
                                       kernel_dim1=7, kernel_dim2=7, kernel_dim3=7, activation='relu'),
        Dropout(0.2),    BatchNormalization(),    MaxPooling3D(pool_size=(2, 2, 2)),
        Convolution3D(nb_filter=16,
                      kernel_dim1=5, kernel_dim2=5, kernel_dim3=5, activation='relu'),
        Dropout(0.2),    Convolution3D(nb_filter=16,
                                       kernel_dim1=5, kernel_dim2=5, kernel_dim3=5, activation='relu'),
        Dropout(0.2),    Convolution3D(nb_filter=16,
                                       kernel_dim1=5, kernel_dim2=5, kernel_dim3=5, activation='relu'),
        Dropout(0.2),    BatchNormalization(),    MaxPooling3D(pool_size=(2, 2, 2)),
        Convolution3D(nb_filter=8,
                      kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu'),
        Dropout(0.2),
        Convolution3D(nb_filter=8,
                      kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu'),
        Dropout(0.2),    Convolution3D(nb_filter=8,
                                       kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu'),
        Dropout(0.2),    BatchNormalization(),    MaxPooling3D(pool_size=(2, 2, 2)),
        #AveragePooling3D(pool_size=(3,3,3)),
        Flatten(),
        Dense(64, init='normal', activation='relu'),
        Dense(1, init='normal', activation='sigmoid')])

basic_conv3d.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
basic_conv3d.optimizer.lr.set_value(0.01)
basic_conv3d.fit(train_inputs, train_labels,
                 validation_data=(validation_inputs, validation_labels),
                 nb_epoch=5, batch_size=1)



print("predictions beginning")


tmp = [list(labels_df.id[labels_df.id == i])[0] for i in labeled_patients]
all_patients = [i.split('.')[0] for i in patients_npy]
#test_patients = [i for i in all_patients if i not in train_patients + [tmp[1177]]]
test_patients = [i for i in all_patients]
test_arr = np.empty(shape=(len(test_patients), dim1, dim2, dim3))

for patient_indx in range(len(test_patients)):
    test_arr[patient_indx,:,:] = np.load('{}{}.npy'.format(data_pre, test_patients[patient_indx]))

print("writing predictions")

test_inputs = np.expand_dims(np.array(test_arr),1)

preds = basic_conv3d.predict(test_inputs, batch_size=2)



out_tuples = [[test_patients[i], preds[i][0]] for i in range(len(test_patients))] 

out_df = pd.DataFrame.from_records(out_tuples)

out_df.columns = ['id', 'cancer']



out_df.to_csv('/data/kaggle/03-06-2017_preds.csv')

print("finished successfully")
