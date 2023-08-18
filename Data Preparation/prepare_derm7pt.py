import os
import pandas as pd
import shutil

src_csv_dir = '/home/eliezer/Área de Trabalho/Drive/Datasets/release_v0/meta'
src_imgs_dir = '/home/eliezer/Área de Trabalho/Drive/Datasets/release_v0/images'
dst_dir = '../Derm7pt'

df = pd.read_csv(os.path.join(src_csv_dir, 'meta.csv'))

atypical_nevus_df = df.loc[df['diagnosis'] == 'clark nevus']
melanoma_df = df.loc[df['diagnosis'].astype(str).str.startswith('melanoma')]

notmelanoma_dir = os.path.join(dst_dir, 'notmelanoma')
if not os.path.exists(notmelanoma_dir):
    os.makedirs(notmelanoma_dir)

for index, row in atypical_nevus_df.iterrows():
    src_img_dir = os.path.join(src_imgs_dir, row['clinic'])
    img_name = src_img_dir.split('/')[-1]
    print('Copiando a imagem {}...'.format(img_name))
    dst_img_dir = os.path.join(notmelanoma_dir, img_name)
    shutil.copyfile(src_img_dir, dst_img_dir)    

melanoma_dir = os.path.join(dst_dir, 'melanoma')
if not os.path.exists(melanoma_dir):
    os.makedirs(melanoma_dir)

for index, row in melanoma_df.iterrows():
    src_img_dir = os.path.join(src_imgs_dir, row['clinic'])
    img_name = src_img_dir.split('/')[-1]
    print('Copiando a imagem {}...'.format(img_name))
    dst_img_dir = os.path.join(melanoma_dir, img_name)
    shutil.copyfile(src_img_dir, dst_img_dir)

