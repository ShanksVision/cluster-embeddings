# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:57:36 2020

@author: shankarj
"""
import glob2
from PIL import Image
import shutil
from pathlib import Path

min_size = 112
new_size = 224, 224
folder = 'Data/raw/zip/*/*'
new_folder = 'Data/processed/zip'

for file in glob2.glob(folder):
    im = Image.open(file)
    w, h = im.size
    if w < min_size or h < min_size:
        continue
    new_image = im.resize(new_size)
    file_path = Path(file)   
    shutil.os.makedirs(f'{new_folder}/{file_path.parent.name}', exist_ok=True)
    new_image.save(f'{new_folder}/{file_path.parent.name}/{file_path.stem}.png', "PNG")
    