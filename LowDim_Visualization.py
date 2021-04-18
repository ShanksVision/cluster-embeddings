# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:26:20 2020

@author: shankarj
"""

import torch as pt
import torchvision as tv
import sklearn
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary as summary
from PIL import Image

gpu = pt.device("cuda:0")
pre_proc = tv.transforms.Compose([tv.transforms.ToTensor()])
data = tv.datasets.ImageFolder('Data/processed/tutorial', transform=pre_proc)
model = tv.models.wide_resnet50_2(pretrained=True, progress=False)

data_loader = pt.utils.data.DataLoader(data, batch_size=len(data), shuffle=False,
                                       pin_memory=True)
class_names = data_loader.dataset.class_to_idx
model.fc = pt.nn.Identity()
# dummy_tensor = pt.ones((1, 3, 224, 224))
# output_tensor = model(dummy_tensor)

model.train(False)
model.to(gpu)

with pt.no_grad():
    for i, (data,labels) in enumerate(data_loader):
        data, labels = data.to(gpu), labels.to(gpu)
        output = model(data)        
        #Copy the data from gpu memory to cpu before passing to numpy
        features = output.cpu().numpy()
        labels = labels.cpu().numpy()

tSNE = manifold.TSNE(perplexity=40, n_components=2)
Z = tSNE.fit_transform(features)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):

    # compute the distribution range4
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range

tx = scale_to_01_range(Z[:, 0])
ty = scale_to_01_range(Z[:, 1])

# initialize a matplotlib plot
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0.1, 0.9, len(class_names))]
ax.set_prop_cycle('color', colors)

# for every class, we'll add a scatter plot separately
for label in class_names:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == class_names[label]]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    # color = np.array(class_names[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    a = np.random.uniform(0.2, 1.0)
    ax.scatter(current_tx, current_ty, label=label, alpha=a)
    #ax.scatter(current_tx, current_ty, label=label, color='blue')
    ax.annotate(label, (np.random.choice(current_tx), np.random.choice(current_ty)))

# build a legend using the labels we set previously
ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

# finally, show the plot
plt.tight_layout(pad = 1.05)
plt.show()


#show outlier data
Z_scaled = np.stack((tx, ty), axis=1)
knn = sklearn.neighbors.KNeighborsClassifier(len(class_names))
knn.fit(Z_scaled, labels)
labels_pred = knn.predict(Z_scaled)

def plot_loss_wrong_preds(x, y, yhat, labels, mis_idx, Z=None):   
    #Show some misclassified samples   
    size = min(len(mis_idx), 16)
    if size < 16:
        wrong_preds = mis_idx
    else:
        wrong_preds = np.random.choice(np.array(mis_idx), size=size) 
    ax = []
    fig=plt.figure(figsize=(12, 12))
    columns = 4
    rows = 4
    for i, j in enumerate(wrong_preds):
        if(type(x) is np.ndarray):
            img = x[j]
        else:
            img = plt.imread(x[j])
        ax.append(fig.add_subplot(rows, columns, i+1))
        #ax[-1].set_title(f'true: {labels[y[j]]}, pred: {labels[yhat[j]]}')   
        ax[-1].set_title(f'true: {labels[y[j]]}')                          
        plt.imshow(img)
    plt.tight_layout(pad=1.2)    
    fig.suptitle('Wrong predictions', y = 0.001)
    plt.show()

X = [x[0] for x in data_loader.dataset.imgs]

#get some wrong predictions
# mis_idx = np.where(labels != labels_pred)[0]        
# mis_idx = np.arange(22*24, 23*24)
# mis_idx = np.append(mis_idx, np.arange(28*24, 30*24))
#mis_idx = np.intersect1d(np.where(Z_scaled[:,0] > 0.55)[0], np.where(labels == 0)[0])
_, mis_idx = np.unique(labels, return_index=True)
plot_loss_wrong_preds(X, labels, labels_pred, data_loader.dataset.classes, 
                      mis_idx, Z_scaled)  
