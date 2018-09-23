import os
import shutil
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import ipdb


classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

data = pd.read_csv('../ISIC2018_Task3_Training_GroundTruth.csv', header=0)

root = '../data/'

#ipdb.set_trace()

data = shuffle(data)

total = data.shape[0]
train_number = int(total*0.9)

for i in range(data.shape[0]):
    img = data.iloc[i,0]
    label = data.iloc[i,1:].values.argmax()
    label = classes[label]   
    if i <= train_number: # in training set
        root = '../data/train/'
    else:
        root = '../data/val/'
    if not os.path.exists(root + label):
        os.makedirs(root + label)
    shutil.copy('../data/ISIC2018_Task3_Training_Input/'+img+'.jpg', root + label + '/' + img + '.jpg')


