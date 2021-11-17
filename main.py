
import numpy as np
import pandas as pd
import pydicom as di
import matplotlib.pyplot as plt
import keras
import os
import random
from tqdm import tqdm
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import metrics

root_path = "d://xray/"
train_path = "d://xray/train800/"
test_path = "d://xray/test800/"
batch_size = 2
dim=(800,800)
valRatio = 0.2

# img = di.read_file(train_path + '8da36a523c6b7ccc9f3589a4d129c96d' + '.dicom', force=True)
# img = img.pixel_array

# get the original images' size and store them in a csv file
# retrieve the files existing in specific folder
# train_files = [fileName for fileName in os.listdir("d://xray/train/") if fileName.endswith(".dicom")]
# test_files = [fileName for fileName in os.listdir("d://xray/test/") if fileName.endswith(".dicom")]
# train_files = [_.split('.')[0]   for _ in train_files]      # remove the extension file names
# test_files = [_.split('.')[0]   for _ in test_files]
# train_sizes = pd.DataFrame({'image_id': train_files})
# test_sizes = pd.DataFrame({'image_id': test_files})
# # save file size in dataframe
# for dicom in tqdm(train_files):
#     img = di.read_file(train_path + dicom + '.dicom', force=True)
#     train_sizes.loc[train_sizes.image_id==dicom, 'x_size'] = img.pixel_array.shape[0]
#     train_sizes.loc[train_sizes.image_id == dicom, 'y_size'] = img.pixel_array.shape[1]
#
# for dicom in tqdm(test_files):
#     img = di.read_file(test_path + dicom + '.dicom', force=True)
#     test_sizes.loc[test_sizes.image_id==dicom, 'x_size'] = img.pixel_array.shape[0]
#     test_sizes.loc[test_sizes.image_id == dicom, 'y_size'] = img.pixel_array.shape[1]
#
# train_sizes.to_csv("d://xray/train_fileSizes.csv")
# test_sizes.to_csv("d://xray/test_fileSizes.csv")


# custom DataGenerator because the data set is too huge to load all to memory
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels.iloc[:, 1:]
        self.labels.iloc[:, 1:] = self.labels.iloc[:, 1:].astype(np.float16)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, f, cl0, bb0, cl1, bb1, cl2, bb2, cl3, bb3, cl4, bb4, cl5, bb5, cl6, bb6,\
            cl7, bb7, cl8, bb8, cl9, bb9 = self.__data_generation(list_IDs_temp)
        return [X], [f, cl0, bb0, cl1, bb1, cl2, bb2, cl3, bb3, cl4, bb4, cl5, bb5, cl6, bb6, cl7, bb7, cl8, bb8, cl9, bb9]
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        f = np.empty((self.batch_size, 1), dtype=np.float16)
        cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9 = \
            (np.empty((self.batch_size, 16), dtype=np.float16) for _ in range(10))
        bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9 = \
            (np.empty((self.batch_size, 4), dtype=np.float16) for _ in range(10))
        # Generate data
        for i, image_id in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(train_path + image_id + '.npy')[:,:, np.newaxis]
            # Store class
            df_temp = self.labels.loc[df.image_id == image_id]
            f[i] = np.array(df_temp.iloc[0, 1:2].values[0])
            cl0[i] = np.array(df_temp.iloc[0, 2:18].tolist())
            cl1[i] = np.array(df_temp.iloc[0, 22:38].tolist())
            cl2[i] = np.array(df_temp.iloc[0, 42:58].tolist())
            cl3[i] = np.array(df_temp.iloc[0, 62:78].tolist())
            cl4[i] = np.array(df_temp.iloc[0, 82:98].tolist())
            cl5[i] = np.array(df_temp.iloc[0, 102:118].tolist())
            cl6[i] = np.array(df_temp.iloc[0, 122:138].tolist())
            cl7[i] = np.array(df_temp.iloc[0, 142:158].tolist())
            cl8[i] = np.array(df_temp.iloc[0, 162:178].tolist())
            cl9[i] = np.array(df_temp.iloc[0, 182:198].tolist())
            bb0[i] = np.array(df_temp.iloc[0, 18:22].tolist())
            bb1[i] = np.array(df_temp.iloc[0, 38:42].tolist())
            bb2[i] = np.array(df_temp.iloc[0, 58:62].tolist())
            bb3[i] = np.array(df_temp.iloc[0, 78:82].tolist())
            bb4[i] = np.array(df_temp.iloc[0, 98:102].tolist())
            bb5[i] = np.array(df_temp.iloc[0, 118:122].tolist())
            bb6[i] = np.array(df_temp.iloc[0, 138:142].tolist())
            bb7[i] = np.array(df_temp.iloc[0, 158:162].tolist())
            bb8[i] = np.array(df_temp.iloc[0, 178:182].tolist())
            bb9[i] = np.array(df_temp.iloc[0, 198:].tolist())
            # for j in range(10):
            #     y[i][2*(j+1)-1] = df_temp.iloc[0, j * 20 + 2 : j * 20 + 18].tolist()
            #     y[i][2*(j+1)][0:4] = df_temp.iloc[0, j * 20 + 18 : j * 20 + 22].tolist()
            # y = np.array(y)
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, f, cl0, bb0, cl1, bb1, cl2, bb2, cl3, bb3, cl4, bb4, cl5, bb5, cl6, bb6, cl7, bb7, cl8, bb8, cl9, bb9


def BaseLineModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((4, 4))(X_input)
    X = Conv2D(32, (9, 9), strides = (4, 4), name = 'conv0')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(128, (3, 3), strides = (2, 2), name = 'conv2')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(256, (3, 3), strides = (2, 2), name = 'conv3')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(256, (3, 3), strides = (2, 2), name = 'conv4')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc0')(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    f = Dense(1, activation='sigmoid', name='found')(X)
    cl0 = Dense(16, activation='softmax', name='cl0')(X)
    bb0 = Dense(4, activation='relu', name='bb0')(X)
    cl1 = Dense(16, activation='softmax', name='cl1')(X)
    bb1 = Dense(4, activation='relu', name='bb1')(X)
    cl2 = Dense(16, activation='softmax', name='cl2')(X)
    bb2 = Dense(4, activation='relu', name='bb2')(X)
    cl3 = Dense(16, activation='softmax', name='cl3')(X)
    bb3 = Dense(4, activation='relu', name='bb3')(X)
    cl4 = Dense(16, activation='softmax', name='cl4')(X)
    bb4 = Dense(4, activation='relu', name='bb4')(X)
    cl5 = Dense(16, activation='softmax', name='cl5')(X)
    bb5 = Dense(4, activation='relu', name='bb5')(X)
    cl6 = Dense(16, activation='softmax', name='cl6')(X)
    bb6 = Dense(4, activation='relu', name='bb6')(X)
    cl7 = Dense(16, activation='softmax', name='cl7')(X)
    bb7 = Dense(4, activation='relu', name='bb7')(X)
    cl8 = Dense(16, activation='softmax', name='cl8')(X)
    bb8 = Dense(4, activation='relu', name='bb8')(X)
    cl9 = Dense(16, activation='softmax', name='cl9')(X)
    bb9 = Dense(4, activation='relu', name='bb9')(X)
    model = Model(inputs = X_input, outputs = [f, cl0, bb0, cl1, bb1, cl2, bb2, cl3, bb3,\
                                               cl4, bb4, cl5, bb5, cl6, bb6, cl7, bb7,\
                                               cl8, bb8, cl9, bb9])
    return model


def BaseLineModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((4, 4))(X_input)
    X = Conv2D(32, (9, 9), strides = (4, 4), name = 'conv0')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(64, (3, 3), strides = (2, 2), name = 'conv2')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(128, (3, 3), strides = (2, 2), name = 'conv3')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(128, (3, 3), strides = (2, 2), name = 'conv4')(X)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    #X = Dense(256, activation='relu', name='fc0')(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    f = Dense(1, activation='sigmoid', name='found')(X)
    cl0 = Dense(16, activation='softmax', name='cl0')(X)
    bb0 = Dense(4, activation='relu', name='bb0')(X)
    cl1 = Dense(16, activation='softmax', name='cl1')(X)
    bb1 = Dense(4, activation='relu', name='bb1')(X)
    cl2 = Dense(16, activation='softmax', name='cl2')(X)
    bb2 = Dense(4, activation='relu', name='bb2')(X)
    cl3 = Dense(16, activation='softmax', name='cl3')(X)
    bb3 = Dense(4, activation='relu', name='bb3')(X)
    cl4 = Dense(16, activation='softmax', name='cl4')(X)
    bb4 = Dense(4, activation='relu', name='bb4')(X)
    cl5 = Dense(16, activation='softmax', name='cl5')(X)
    bb5 = Dense(4, activation='relu', name='bb5')(X)
    cl6 = Dense(16, activation='softmax', name='cl6')(X)
    bb6 = Dense(4, activation='relu', name='bb6')(X)
    cl7 = Dense(16, activation='softmax', name='cl7')(X)
    bb7 = Dense(4, activation='relu', name='bb7')(X)
    cl8 = Dense(16, activation='softmax', name='cl8')(X)
    bb8 = Dense(4, activation='relu', name='bb8')(X)
    cl9 = Dense(16, activation='softmax', name='cl9')(X)
    bb9 = Dense(4, activation='relu', name='bb9')(X)
    model = Model(inputs = X_input, outputs = [f, cl0, bb0, cl1, bb1, cl2, bb2, cl3, bb3,\
                                               cl4, bb4, cl5, bb5, cl6, bb6, cl7, bb7,\
                                               cl8, bb8, cl9, bb9])
    return model

df = pd.read_csv(root_path+'train.csv')
idx = df.index.tolist()
random.shuffle(idx)
val_idx = idx[:int(len(idx)*valRatio)]
train_idx = idx[int(len(idx)*valRatio):]
X_train = df.loc[train_idx, 'image_id'].tolist()
X_validation = df.loc[val_idx, 'image_id'].tolist()
y_train = df.loc[train_idx, :]
y_validation = df.loc[val_idx, :]

training_generator = DataGenerator(X_train, y_train, batch_size, dim)
validation_generator = DataGenerator(X_validation, y_validation, batch_size, dim)
baselineModel = BaseLineModel((800, 800, 1))
baselineModel.compile(loss={'found': 'binary_crossentropy',
    'cl0': 'categorical_crossentropy', 'bb0': 'mean_squared_error',
    'cl1': 'categorical_crossentropy', 'bb1': 'mean_squared_error',
    'cl2': 'categorical_crossentropy', 'bb2': 'mean_squared_error',
    'cl3': 'categorical_crossentropy', 'bb3': 'mean_squared_error',
    'cl4': 'categorical_crossentropy', 'bb4': 'mean_squared_error',
    'cl5': 'categorical_crossentropy', 'bb5': 'mean_squared_error',
    'cl6': 'categorical_crossentropy', 'bb6': 'mean_squared_error',
    'cl7': 'categorical_crossentropy', 'bb7': 'mean_squared_error',
    'cl8': 'categorical_crossentropy', 'bb8': 'mean_squared_error',
    'cl9': 'categorical_crossentropy', 'bb9': 'mean_squared_error'},
    metrics={'found': metrics.BinaryAccuracy(),
    'cl0': metrics.CategoricalAccuracy(), 'bb0': metrics.MeanSquaredError(),
    'cl1': metrics.CategoricalAccuracy(), 'bb1': metrics.MeanSquaredError(),
    'cl2': metrics.CategoricalAccuracy(), 'bb2': metrics.MeanSquaredError(),
    'cl3': metrics.CategoricalAccuracy(), 'bb3': metrics.MeanSquaredError(),
    'cl4': metrics.CategoricalAccuracy(), 'bb4': metrics.MeanSquaredError(),
    'cl5': metrics.CategoricalAccuracy(), 'bb5': metrics.MeanSquaredError(),
    'cl6': metrics.CategoricalAccuracy(), 'bb6': metrics.MeanSquaredError(),
    'cl7': metrics.CategoricalAccuracy(), 'bb7': metrics.MeanSquaredError(),
    'cl8': metrics.CategoricalAccuracy(), 'bb8': metrics.MeanSquaredError(),
    'cl9': metrics.CategoricalAccuracy(), 'bb9': metrics.MeanSquaredError()},
                      optimizer='adam')
baselineModel.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs = 10, shuffle = True,
                            max_queue_size=10, workers = 2)

baselineModel.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=False,
                            epochs = 10, shuffle = True)

