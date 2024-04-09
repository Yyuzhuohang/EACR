import torch
import torch.nn as nn
import torch.optim as optim
from util import give_batch,CustomDataset
import os
from tqdm import tqdm
import sys
from model import GCNconv
import logging
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_dim = 2
dim = 128

batch_size = 128
learning_rate = 0.003
epochs = 1000
dropout_prob = 0.5

model = GCNconv(num_node_features=1, num_classes=10)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total number of parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

get_data = give_batch()

train_dataset = []
for i in range(len(get_data.train_feature)):
    x = get_data.train_feature[i]  
    edge_index = get_data.train_edge[i]#[i]  
    y = get_data.train_y[i]  
    data = Data(x=x, edge_index=edge_index, y=y)
    train_dataset.append(data)

test_dataset = []
for i in range(len(get_data.test_feature)):
    x = get_data.test_feature[i]  
    edge_index = get_data.test_edge[i]#[i]  
    y = get_data.test_y[i]  
    data = Data(x=x, edge_index=edge_index, y=y)
    test_dataset.append(data)

def train(train_dataloader,test_dataloader):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_total = 0
        train_correct = 0
        for data in tqdm(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            _, predicted = torch.max(out, 1)
            train_total += data.y.size(0)
            train_correct += (predicted == data.y).sum().item()
        train_accuracy = 100 * train_correct / train_total
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}, Train Accuracy: {train_accuracy}%')
        model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()
            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy}%')
            logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}' + '\n' + str(accuracy) + '\n')
        if (epoch+1) % 10==0:
            torch.save(model, "ckpt/transformer_{}.pth".format(epoch+1))
            print("save model")

def test(train_dataloader,test_dataloader):
    model = torch.load('ckpt/transformer_50.pth')
    all_preds = []
    all_labels = []
    num_classes = 10
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
        accuracy = 100 * correct / total
        #conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}%')

if __name__=="__main__":
    if sys.argv[1] == 'train':
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                            filename='log.log', 
                            filemode='w') 
        logger = logging.getLogger(__name__)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train(train_dataloader,test_dataloader)
    if sys.argv[1] == 'test':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test(train_dataloader,test_dataloader)