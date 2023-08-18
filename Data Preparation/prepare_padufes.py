import cv2
import os
import pandas as pd
from PIL import Image

csv_dir = '/home/eliezer/Downloads/zr7vgbcyr2-1'
imgs_dir = '/home/eliezer/Downloads/zr7vgbcyr2-1/images'
dst_dir = '../PAD-UFES'

df = pd.read_csv(os.path.join(csv_dir, 'metadata.csv'))
df_biopsed = df.loc[df['biopsed'] == True] 
df_notmelanoma = df_biopsed.loc[df_biopsed['diagnostic'] == 'NEV']
df_melanoma = df_biopsed.loc[df_biopsed['diagnostic'] == 'MEL']

notmelanoma_dir = os.path.join(dst_dir, 'notmelanoma')
if not os.path.exists(notmelanoma_dir):
    os.makedirs(notmelanoma_dir)

for index, row in df_notmelanoma.iterrows():
    img_name = row['img_id'].split('.')[0]
    print('Convertendo para .jpg e copiando a imagem {}...'.format(img_name))
    img_tmp = cv2.imread(os.path.join(imgs_dir, img_name + '.png'))
    img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img, 'RGB')
    img_pil.save(os.path.join(notmelanoma_dir, img_name + '.jpg'))

melanoma_dir = os.path.join(dst_dir, 'melanoma')
if not os.path.exists(melanoma_dir):
    os.makedirs(melanoma_dir)

for index, row in df_melanoma.iterrows():
    img_name = row['img_id'].split('.')[0]
    print('Convertendo para .jpg e copiando a imagem {}...'.format(img_name))
    img_tmp = cv2.imread(os.path.join(imgs_dir, img_name + '.png'))
    img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img, 'RGB')
    img_pil.save(os.path.join(melanoma_dir, img_name + '.jpg'))

