import numpy as np
import pandas as pd
import pydicom as di
import matplotlib.pyplot as plt
import keras
import os
from tqdm import tqdm
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from skimage.transform import resize
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


root_path = "d://xray/"
train_path = "d://xray/train800/"
test_path = "d://xray/test800/"

# output new file which keeps the original image sizes
def initSizes(path, ext, fileName):
    import os
    from tqdm import tqdm
    files_lst = [fileName for fileName in os.listdir(path) if fileName.endswith(ext)]
    files_lst = [_.split('.')[0] for _ in files_lst]         # remove the extension file names
    df = pd.DataFrame({'image_id': files_lst})
    for file in tqdm(files_lst):
        img = di.read_file(path + file + ext, force=True).pixel_array
        df.loc[df.image_id == file, 'y_size'] = img.shape[0]
        df.loc[df.image_id == file, 'x_size'] = img.shape[1]
    df.to_csv(fileName)

initSizes(train_path, '.dicom', 'd://xray/train_fileSizes.csv')
initSizes(test_path, '.dicom', 'd://xray/test_fileSizes.csv')

# calculate and add bounding boxes
def addBBox(path, file):
    df = pd.read_csv(path + file)
    df['w'] = df.x_max - df.x_min
    df['h'] = df.y_max - df.y_min
    df['bb_x'] = df.x_min + (0.5 * df.w).round(decimals=0)
    df['bb_y'] = df.y_min + (0.5 * df.h).round(decimals=0)
    df.to_csv(path+file)

addBBox(root_path, 'train.csv')


# normalize boxes
def normBoxes(path, sizefile, trainfile):
    tdf = pd.read_csv(path + trainfile)
    sdf = pd.read_csv(path + sizefile)
    for img in tqdm(sdf.image_id):
        x_size = sdf.loc[sdf.image_id == img, ['x_size']].astype(float)
        y_size = sdf.loc[sdf.image_id == img, ['y_size']].astype(float)
        tdf.loc[tdf.image_id == img, ['x_min', 'x_max', 'bb_x', 'w']] = \
            tdf.loc[tdf.image_id == img, ['x_min', 'x_max', 'bb_x', 'w']] \
            / x_size.values[0,0]
        tdf.loc[tdf.image_id == img, ['y_min', 'y_max', 'bb_y', 'h']] = \
            tdf.loc[tdf.image_id == img, ['y_min', 'y_max', 'bb_y', 'h']] \
            / y_size.values[0,0]
    tdf.to_csv(path + trainfile)

normBoxes(root_path, "train_fileSizes.csv", "train.csv")

# downsize images
def downsize(inpath, outpath, outshape, ext):
    import os
    from tqdm import tqdm
    files_lst = [fileName for fileName in os.listdir(inpath) if fileName.endswith(ext)]
    files_lst = [_.split('.')[0] for _ in files_lst]         # remove the extension file names
    for file in tqdm(files_lst):
        img = di.read_file(inpath + file + ext, force=True).pixel_array
        img1 = resize(img, outshape, anti_aliasing=True)
        np.save(outpath+file+'.npy', img1)

downsize('G://ML/Kaggle/Vision/xray/data/train/', 'D://xray/train800/', (800, 800), '.dicom')
downsize('G://ML/Kaggle/Vision/xray/data/test/', 'D://xray/test800/', (800, 800), '.dicom')

# prepare training data which each row for one image
# combine multiple records with same image_id to one row
def combineRows(path, file):
    df = pd.read_csv(path+file)
    df = df[['image_id', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max']]
    # add found feature which indicates if find any problem
    df['found'] = [0 if _==14 else 1 for _ in df['class_id']]
    # add 1 to all class_id, the 0 is preserved for standing null
    # after dummies, any classx_0==1 means no data after that
    df['class_id'] = df['class_id']+1
    # average multiple records with same image and class_id
    df = df.groupby(by=['image_id', 'class_id'], as_index=False).mean()
    df = df.assign(cid=df.groupby(['image_id']).cumcount()).set_index(\
        ['image_id', 'cid']).unstack(-1).sort_index(1, 1)
    df.columns = [f'{x}{y}' for x, y in df.columns]
    df = df.reset_index()
    # convert float in column names to integer
    cols = [c for c in df.columns if c[:8]=='class_id']
    df[cols] = df[cols].fillna(0)
    df[cols] = df[cols].astype(int)
    # dummy class_id feature
    for i in range(10):
        df = pd.concat([df, pd.get_dummies(df['class_id'+str(i)], prefix='class'+str(i))], axis=1)
        df.drop(['class_id'+str(i)], axis=1, inplace=True)
    existing_cols = [c for c in df.columns if c[:5]=='class']
    all_class_cols = ['class'+str(j)+'_'+str(i)  for i in range(16) for j in range(10)]
    for col in list(set(all_class_cols)-set(existing_cols)):
        df[col] = 0
    # fillna and change to float16
    df = df.fillna(0)
    df.loc[:, df.columns[1:]] = df.loc[:, df.columns[1:]].astype(np.float16)
    # drop found1-found9
    cols = ['found'+str(i) for i in range(1,10)]
    df.drop(cols, axis=1, inplace=True)
    # rearrange the order of columns
    cols = df.columns[:2]
    for i in range(10):
        for j in range(16):
            cols.append('class'+str(i)+'_'+str(j))
        cols += ['x_min' + str(i), 'y_min' + str(i), 'x_max' + str(i), 'y_max' + str(i)]
    df = df[cols]
    #df = df.iloc[:, 1:]  # drop the columns of previous index
    df.to_csv(path+file)

combineRows(root_path, 'train.csv')

