import pandas as pd
import os
import os.path
from shutil import copy

df_train = pd.read_csv('train_label.csv')
data_dir_2020='/home/data/ISIC/ISIC2020/'
df_train['source_path'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir_2020, 'jpeg/train', f'{x}.jpg')) # jpg path
df_train['target_path'] = df_train['image_name'].apply(lambda x: os.path.join('/home/tmk/project/mel_classfication/jpeg/train/')) # jpg path
for index, row in df_train.iterrows():
    copy(row['source_path'],row['target_path'])