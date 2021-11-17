class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=16, shuffle=True, train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels.iloc[:, 1:]
        # self.labels.iloc[:, 1:] = self.labels.iloc[:, 1:].astype(np.float16)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train = train
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return [X]
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        f = np.empty((self.batch_size, 1), dtype=np.float16)
        cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9 = \
            (np.empty((self.batch_size, 16), dtype=np.float16) for _ in range(10))
        bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9 = \
            (np.empty((self.batch_size, 4), dtype=np.float16) for _ in range(10))
        # Generate data
        for i, image_id in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(test_path + image_id + '.npy')[:,:, np.newaxis]
            # Store class
            # df_temp = self.labels.loc[df.image_id == image_id]
            # f[i] = np.array(df_temp.iloc[0, 1:2].values[0])
            # cl0[i] = np.array(df_temp.iloc[0, 2:18].tolist())
            # cl1[i] = np.array(df_temp.iloc[0, 22:38].tolist())
            # cl2[i] = np.array(df_temp.iloc[0, 42:58].tolist())
            # cl3[i] = np.array(df_temp.iloc[0, 62:78].tolist())
            # cl4[i] = np.array(df_temp.iloc[0, 82:98].tolist())
            # cl5[i] = np.array(df_temp.iloc[0, 102:118].tolist())
            # cl6[i] = np.array(df_temp.iloc[0, 122:138].tolist())
            # cl7[i] = np.array(df_temp.iloc[0, 142:158].tolist())
            # cl8[i] = np.array(df_temp.iloc[0, 162:178].tolist())
            # cl9[i] = np.array(df_temp.iloc[0, 182:198].tolist())
            # bb0[i] = np.array(df_temp.iloc[0, 18:22].tolist())
            # bb1[i] = np.array(df_temp.iloc[0, 38:42].tolist())
            # bb2[i] = np.array(df_temp.iloc[0, 58:62].tolist())
            # bb3[i] = np.array(df_temp.iloc[0, 78:82].tolist())
            # bb4[i] = np.array(df_temp.iloc[0, 98:102].tolist())
            # bb5[i] = np.array(df_temp.iloc[0, 118:122].tolist())
            # bb6[i] = np.array(df_temp.iloc[0, 138:142].tolist())
            # bb7[i] = np.array(df_temp.iloc[0, 158:162].tolist())
            # bb8[i] = np.array(df_temp.iloc[0, 178:182].tolist())
            # bb9[i] = np.array(df_temp.iloc[0, 198:].tolist())
        return X

