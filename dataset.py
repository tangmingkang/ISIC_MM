import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

num=0

class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform
        self.num=0

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread(row.filepath) # 默认读出的是BGR模式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # m*n*3
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32) # 512*512*3
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1) # 3*512*512

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float()) # (3*512*512,14(site:10))
        else:
            data = torch.tensor(image).float()
        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()




def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val


def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site') # 获取site one hot编码  dummy_na=True：空值单独编码
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train.loc[df_train['patient_id'] == 0, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    
    df_train['image_size'] = np.log(train_sizes)

    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features



def get_df(kernel_type, out_dim, data_dir_2020,data_dir_2019,data_dir_2018, use_meta):

    # 四个数据集都不包含重复图片，2018train在2019中完整出现过，2018val是新的数据集
    df_train = pd.read_csv(os.path.join(data_dir_2020, 'train.csv')) # label
    df_fold = pd.read_csv('fold.csv')
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir_2020, 'jpeg/train', f'{x}.jpg')) # jpg path
    df_train['image'] = df_train['image_name']
    del df_train['image_name']

    df_train2 = pd.read_csv(os.path.join(data_dir_2019, 'ISIC_2019_Training_GroundTruth.csv')) # label
    df_train2_meta = pd.read_csv(os.path.join(data_dir_2019, 'ISIC_2019_Training_Metadata.csv')) # meta
    df_train2['filepath'] = df_train2['image'].apply(lambda x: os.path.join(data_dir_2019, 'ISIC_2019_Training_Input', f'{x}.jpg')) # jpg path
    # df_train2 的标签是one hot编码  转换为正常编码
    def f(x):
        for c in df_train2.columns:
            if x[c]==1.0:
                return c
    df_train2['diagnosis']=df_train2.apply(f,axis=1)
    df_train2=df_train2[['image','filepath','diagnosis']]
    df_train2=pd.merge(df_train2, df_train2_meta, on=['image'], how='left')
    df_train2['anatom_site_general_challenge']=df_train2['anatom_site_general']
    del df_train2['anatom_site_general']

    df_train3 = pd.read_csv(os.path.join(data_dir_2018, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')) # label
    df_train3['filepath'] = df_train3['image'].apply(lambda x: os.path.join(data_dir_2018, 'ISIC2018_Task3_Training_Input', f'{x}.jpg')) # jpg path
    df_train3_val = pd.read_csv(os.path.join(data_dir_2018, 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')) # label
    df_train3_val['filepath'] = df_train3_val['image'].apply(lambda x: os.path.join(data_dir_2018, 'ISIC2018_Task3_Validation_Input', f'{x}.jpg')) # jpg path
    
    # df_train=df_train['image']
    # df_train2=df_train2['image']
    # df_train=pd.concat([df_train,df_train2]).unique()
    # df_train3=df_train3['image'].unique()
    # df_train3_val=df_train3_val['image'].unique()

    # df_train['fold']=np.random.randint(15,size=len(df_train))
    # df_train2['fold']=np.random.randint(15,size=len(df_train2))
    if 'fold+' in kernel_type:
        foldmap = {
            8:0, 5:0, 11:0,
            7:1, 0:1, 6:1,
            10:2, 12:2, 13:2,
            9:3, 1:3, 3:3,
            14:4, 2:4, 4:4,
        }
    elif 'fold++' in kernel_type:
        foldmap = {i: i % 5 for i in range(15)}
    else:
        foldmap = {
            2:0, 4:0, 5:0,
            1:1, 10:1, 13:1,
            0:2, 9:2, 12:2,
            3:3, 8:3, 11:3,
            6:4, 7:4, 14:4,
        }
    # df_train['fold'] = df_train['fold'].map(foldmap)
    # df_train2['fold']=df_train2['fold'].map(foldmap)

    df_train['is_ext'] = 0
    df_train2['is_ext'] = 1

    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    if out_dim == 9:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()

    # concat train data
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # test data
    df_test = pd.read_csv(os.path.join(data_dir_2020, 'test.csv')) # label
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir_2020, 'jpeg/test', f'{x}.jpg')) # jpg path
    df_test['image'] = df_test['image_name']
    del df_test['image_name']

    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    # class mapping
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    df_train['fold']=df_fold['fold']
    df_train['fold'] = df_train['fold'].map(foldmap)
    mel_idx = diagnosis2idx['melanoma']
    return df_train, df_test, meta_features, n_meta_features, mel_idx # MM的id
