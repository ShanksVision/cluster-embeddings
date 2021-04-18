# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:01:21 2020

@author: shankarj
"""
import glob2
import shutil

for i in range(1, 31):
    folder = f'Data/tutorial/{i}'
    #shutil.os.makedirs(folder, exist_ok=True)
    for file in glob2.glob(f'Data/tutorial/{i}_*'):        
        shutil.copy(file, folder)
