import numpy as np 
import glob
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
#import cv2
import random

class CustomDataset(Dataset):
    def __init__(self, sequences1, labels):
        self.sequences1 = sequences1
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sequence1 = self.sequences1[index]
        label = self.labels[index]
        
        return sequence1, label

class give_batch():
    def __init__(self):
        self.check_and_load()
    
    def adj_to_edge_index_np(self,adj_matrix):
        adj_matrix = np.squeeze(adj_matrix)
        rows, cols = np.nonzero(adj_matrix)
        edge_index = np.vstack([rows, cols])
        edge_index = np.expand_dims(edge_index, axis=0)
        return edge_index
    
    def sample_balanced_classes(self,features, labels, num_samples_per_class):
        unique_labels = np.unique(labels)
        sampled_features = []
        sampled_labels = []
    
        for label in unique_labels:
            # Find indices of all instances of this class
            indices = np.where(labels == label)[0]
            
            # If there are not enough samples for this class, take them all
            if len(indices) < num_samples_per_class:
                sampled_indices = indices
            else:
                # Randomly sample the desired number of indices for this class
                sampled_indices = np.random.choice(indices, num_samples_per_class, replace=False)
            
            # Append the sampled features and labels
            sampled_features.append(features[sampled_indices])
            sampled_labels.append(labels[sampled_indices])
    
        # Concatenate all sampled features and labels
        sampled_features = np.concatenate(sampled_features, axis=0)
        sampled_labels = np.concatenate(sampled_labels, axis=0)
    
        return np.array(sampled_features), np.array(sampled_labels)
    
    def check_and_load(self):
        self.train_edge=np.load('./data_4/train_edge.npy')
        self.train_feature=np.load('./data_4/train_feature.npy')
        self.train_y=np.load('./data_4/train_label.npy')
        
        self.train_edge = self.adj_to_edge_index_np(self.train_edge)
        self.train_edge = np.tile(self.train_edge, (5000, 1, 1))
        
        self.train_feature,self.train_y = self.sample_balanced_classes(self.train_feature,self.train_y,500)
        
        #index_train = list(range(len(self.train_feature)))
        #random.shuffle(index_train)
        #self.train_feature = [self.train_feature[i] for i in index_train][:10000]
        #self.train_y = [self.train_y[i] for i in index_train][:10000]
        
        self.test_edge=np.load('./data_4/test_edge.npy')
        self.test_feature=np.load('./data_4/test_feature.npy')
        self.test_y=np.load('./data_4/test_label.npy')
        
        self.test_edge = self.adj_to_edge_index_np(self.test_edge)
        self.test_edge = np.tile(self.test_edge, (500, 1, 1))
        
        self.test_feature,self.test_y = self.sample_balanced_classes(self.test_feature,self.test_y,50)
        
        #index_test = list(range(len(self.test_feature)))
        #random.shuffle(index_test)
        #self.test_feature = [self.test_feature[i] for i in index_test][:1000]
        #self.test_y = [self.test_y[i] for i in index_test][:1000]
        
        self.train_edge = torch.tensor(self.train_edge)
        self.train_feature = torch.tensor(self.train_feature)
        self.train_y = torch.tensor(self.train_y)

        self.test_edge = torch.tensor(self.test_edge)
        self.test_feature = torch.tensor(self.test_feature)
        self.test_y = torch.tensor(self.test_y)
        
        print(self.train_edge.shape)
        print(self.train_feature.shape)
        print(self.train_y.shape)
        print(self.test_edge.shape)
        print(self.test_feature.shape)
        print(self.test_y.shape)

def plt_hist(lst,name='1'):
    plt.figure(figsize=(10, 6))
    plt.hist(lst, bins=20, color='aqua', edgecolor='black', alpha=0.3)
    
    plt.title('error', fontsize=16)
    plt.xlabel('error values', fontsize=14)
    plt.ylabel('number of error', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('error_%s.png'%(name),dpi=600)
    plt.close()

def dra_fit(true_values,predicted_values):
    #slope, intercept = np.polyfit(true_values, predicted_values, 1)
    #fit_values = slope * true_values + intercept

    plt.scatter(true_values, predicted_values, label="Data Points", color='blue')
    #plt.plot(true_values, fit_values, color='red', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values with Linear Fit')
    plt.legend()
    plt.savefig('nihe.png',dpi=600)
    plt.close()

def plt_bar(values,qujian,min_val, max_val):
    bin_size = (max_val - min_val) / qujian
    labels = [f"{min_val + i*bin_size:.2f}-{min_val + (i+1)*bin_size:.2f}" for i in range(qujian)]
    
    plt.bar(labels, values)
    plt.xlabel('Bins')
    plt.ylabel('Average Value')
    plt.title('Average relative error')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('bar.png',dpi=600)
    plt.close()

def assign_to_bins(data,pre,qujian):
    min_val, max_val = min(data), max(data)
    
    bin_size = (max_val - min_val) / qujian
    
    bins = [[] for _ in range(qujian)]
    
    for num,p in zip(data,pre):
        if num == max_val:
            index = qujian-1
        else:
            index = int((num - min_val) / bin_size)
        num_1 = (num-p)/num
        bins[index].append(num_1)
    bins = [np.mean(x) for x in bins]
    plt_bar(bins,qujian,min_val, max_val)

if __name__ == '__main__':
    D=give_batch()