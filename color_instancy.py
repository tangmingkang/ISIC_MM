import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)
    
    return img.astype(img_dtype) 

ISIC2020_img_train_paths = glob("/home/data/ISIC2020/jpeg/train/*.jpg")
ISIC2020_img_test_paths = glob("/home/data/ISIC2020/jpeg/test/*.jpg")

_n_samples = 8
img_list=[]
img_cc_list=[]
for path in ISIC2020_img_train_paths[0:_n_samples]:
    _img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    img_cc = shade_of_gray_cc (img)  
    img_list.append(img)
    img_cc_list.append(img_cc)
fig=plt.figure()
for i in range(len(img_list)):
    ax=fig.add_subplot(4,4,2*i+1)
    ax.imshow(img_list[i])
    ax1=fig.add_subplot(4,4,2*i+2)
    ax1.imshow(img_cc_list[i])

plt.savefig('test.png')
# def apply_cc (img_paths, output_folder_path, resize=None):
    
#     if not os.path.isdir(output_folder_path):
#         os.mkdir(output_folder_path)    

#     with tqdm(total=len(img_paths), ascii=True, ncols=100) as t:
        
#         for img_path in img_paths:
#             img_name = img_path.split('/')[-1]
#             img_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
#             if resize is not None:
#                 img_ = cv2.resize(img_, resize, cv2.INTER_AREA)
#             np_img = shade_of_gray_cc (img_)            
#             cv2.imwrite(os.path.join(output_folder_path, img_name.split('.')[0] + '.jpg'), np_img)
#             t.update()