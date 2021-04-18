# -*- coding: utf-8 -*-
"""
Created on Oct 26 13:26:20 2020

@author: shankarj
"""

import torch as pt
import torchvision as tv
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import argparse
import glob2;

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple implementation of tsne and clustering')
    
    parser.add_argument('--path', type=str, 
                        help='path where images are stored, needs atleast 1 subfolder')
    parser.add_argument('--classes', type=int,
                        help='Number of expected classes')
    
    parser = parser.parse_args(args)
    
    if parser.path is None:
        raise ValueError('Must provide --path where images are stored')
        return -1
    
    if parser.classes is None:
        raise ValueError('Must provide --classes which is expected number of classes')
        return -1
    
    try:
        #Run dim reduction
        Z, class_dict, orig_labels = reduce_dimension(parser.path, parser.classes)
        plot_data(parser.path, class_dict, orig_labels, Z, 'original')
        
        #Run clustering
        pred_labels = label_data(Z, parser.classes)         
        class_names = {}
        for i in range(parser.classes):
            class_names[f'class{i}'] = i
        plot_data(parser.path, class_names, pred_labels, Z, 'predicted')
        
        #Run data saving
        imgs = glob2.glob(rf'{parser.path}\*\*')
        class_labels = [f'class{i}' for i in pred_labels]
        df = pd.DataFrame(list(zip(imgs, class_labels)), columns=['image', 'class_label'])
        df.to_csv(rf'{parser.path}\pred.csv', header=False, index=False)
        
        return 1
    except Exception as e:
        print("Error in script : ", str(e))
        return 0
        
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):

    # compute the distribution range4
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range    
    
def reduce_dimension(data_path, classes):
    raw_path = rf'{data_path}'
    gpu = pt.device("cuda:1")
    pre_proc = tv.transforms.Compose([tv.transforms.Lambda(lambda x: x.convert('RGB')),
                                      tv.transforms.Resize((224, 224)),                                      
                                      tv.transforms.ToTensor(),
                                      tv.transforms.Normalize(0, 1)])
    img_files = tv.datasets.ImageFolder(raw_path, transform=pre_proc)
    model = tv.models.wide_resnet50_2(pretrained=True, progress=False)
    
    data_loader = pt.utils.data.DataLoader(img_files, batch_size=len(img_files), shuffle=False,
                                           pin_memory=True)
    class_names = data_loader.dataset.class_to_idx
    model.fc = pt.nn.Identity()
    # dummy_tensor = pt.ones((1, 3, 224, 224))
    # output_tensor = model(dummy_tensor)
    
    model.train(False)
    model.to(gpu)
	
    #features = np.zeros((len(img_files),1,2048))
    #labels = np.zeros(len(img_files))       
    
    with pt.no_grad():
        for i, (data,targets) in enumerate(data_loader):
            data, targets = data.to(gpu), targets.to(gpu)        
            output = model(data)        
            #Copy the data from gpu memory to cpu before passing to numpy
            #features[i,:] = output.cpu().numpy()
            #labels[i] = targets.cpu().numpy()
            features = output.cpu().numpy()
            labels = targets.cpu().numpy()
    
   
    #features = features.reshape(len(img_files), 2048)
    tSNE = manifold.TSNE(perplexity=40, n_components=2)
    Z = tSNE.fit_transform(features)   

    return Z, class_names, labels   

#Cluster the data and write results to csv
def label_data(Z_vector, class_count):  
   
    Z_unscaled = np.stack((Z_vector[:, 0], Z_vector[:, 1]), axis=1)
    from sklearn.cluster import KMeans, SpectralClustering
    kmeans = KMeans(n_clusters=class_count, random_state=0).fit(Z_unscaled)
    #kmeans = SpectralClustering(n_clusters=class_count, assign_labels='discretize', 
                                #random_state=0).fit(Z_unscaled)
    labels_pred = kmeans.labels_    

    return labels_pred    

def plot_data(data_path, class_names, labels, Z, plot_name):
    raw_path = rf'{data_path}'
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
        if len(current_tx) != 0 or len(current_ty) != 0:
            ax.annotate(label, (np.random.choice(current_tx), np.random.choice(current_ty)))
    
    # build a legend using the labels we set previously
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
    
    # finally, show the plot
    plt.tight_layout(pad = 1.05)
    plt.savefig(f'{raw_path}\\{plot_name}.png')    
    
    #plt.show()
    
if __name__ == "__main__":
    ret = main()
    if ret <= 0:
        print("Program exited with error code", ret)
    else:
        print("Program exited with success code", ret)
	
	
    
    
    
    
    