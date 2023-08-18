from os.path import dirname

import cv2
import numpy as np
from PIL import Image

from pathutils import mkd

def is_gray(img_bgr):
    if len(img_bgr.shape) < 3: return True
    if img_bgr.shape[2]  == 1: return True
    b,g,r = img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def get_img(img_path):
    img_bgr = cv2.imread(img_path)
    if is_gray(img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_img(img, img_path):
    mode = 'L' if len(img.shape) < 3 else 'RGB'
    img_pil = Image.fromarray(np.uint8(img), mode)
    mkd(dirname(img_path))
    img_pil.save(img_path)

    
