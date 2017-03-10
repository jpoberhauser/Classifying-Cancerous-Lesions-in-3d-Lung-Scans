from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.utils import np_utils
from keras.layers.core import Activation, Dense, Dropout, Flatten


def run_model(X_train, y_train):
    """
    X_train: sample_images3D
    y_train: list(labels_df["cancer"])
    """
    batch_size = 32
    nb_classes = 2
    nb_epoch = 200

    # take look at ImageDataGenerator if there is 3D support
    # data_augmentation = True

    slices, rows, cols = 37, 37, 37 # images shape. Need to all be equal!
    img_channels = 1

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    print Y_train
    # Y_test = np_utils.to_categorical(y_test, nb_classes)

    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # initialize model (Sequential?)
    model = Sequential()
    # how many convolution filters for an input we don't know...
    nb_filter = 30
    model.add(Convolution3D(nb_filter, 3, 3, 3, border_mode='same',
                            input_shape=(3, slices, rows, cols)))

    model.add(Activation('relu'))
    model.add(Convolution3D(nb_filter, 3, 3, 3))
    model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    # model.add(Convolution3D(nb_filter*2, 3, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution3D(64, 3, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Dropout(0.25))

    # no idea how 512 is chosen
    model.add(Flatten())
    # model.add(Dense(512))

    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation="sigmoid"))
    model.add(Activation('softmax'))

    # Let's train the model using adam
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # print model
    model.summary()

    # fit model
    model.fit(X_train, Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        shuffle=True)
