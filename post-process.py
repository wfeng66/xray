import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv(root_path+'train.csv')
cols = train_df.columns.tolist()[1:]
found = np.squeeze(pred[0][:3000]).tolist()
cl, bb, po = [np.nan for _ in range(int((len(pred)-1)/2))],\
             [np.nan for _ in range(int((len(pred)-1)/2))], [np.nan for _ in range(int((len(pred)-1)/2))]

pred_df = pd.DataFrame({
    'image_id': pred_files_lst,
    'found': found,
})

for i in range(int((len(pred)-1)/2)):
    # reduce 1 for each class because they were added by 1 before training
    cl[i] = pd.DataFrame(np.argmax(pred[(i+1)*2-1][:3000], axis=1)-1, columns=['cl'+str(i)])
    po[i] = pd.DataFrame(np.max(pred[(i+1)*2-1][:3000], axis=1), columns=['po'+str(i)])
    bb[i] = pd.DataFrame(pred[(i+1)*2][:3000], columns=cols[i*20+18:i*20+22])
    # bb[i] = pred[(i + 1) * 2][:3000]
    pred_df = pd.concat([pred_df, cl[i], po[i], bb[i]], axis=1)

subm_df = pd.DataFrame({
    'image_id': pred_df.image_id,
    'PredictionString': [np.nan for _ in range(len(pred_df))]
})
sdf = pd.read_csv(root_path+'test_fileSizes.csv')
for idx, row in pred_df.iterrows():
    predStr = ''
    for i in range(int((len(pred_df.columns)-2)/6)):
        if row['cl'+str(i)] < 0 or float(row['po'+str(i)]) < 0.5:
            pass
        else:
            predStr = ' '.join([predStr, str(row['cl' + str(i)]), str(row['po' + str(i)])])
            if row['cl'+str(i)] == 14:
                predStr = ' '.join([predStr, '0', '0', '1', '1'])
            else:
                predStr = ' '.join([predStr, str(int(row['x_min'+str(i)]*sdf.loc[sdf.image_id==row['image_id'], 'x_size'])), \
                                    str(int(row['y_min' + str(i)] * sdf.loc[sdf.image_id == row['image_id'], 'y_size'])),
                                    str(int(row['x_max' + str(i)] * sdf.loc[sdf.image_id == row['image_id'], 'x_size'])),
                                    str(int(row['y_max' + str(i)] * sdf.loc[sdf.image_id == row['image_id'], 'y_size'])),
                                    ])
    subm_df.loc[subm_df.image_id == row['image_id'], 'PredictionString'] = predStr







cl = [np.nan for _ in range(10)]
for i in range(10):
    cl[i] = i**i