import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from util import give_batch,CustomDataset,get_test_acc_regression,find_intersection,plt_hist,dra_fit,plt_bar,give_class,assign_to_bins,saveval_file
import os
from tqdm import tqdm
import sys
from model import TransformerEncoderModel
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim1 = 2  # Example vocabulary size for sequences1
input_dim2 = 2   # Example vocabulary size for sequences2
seq_len_1 = 196
seq_len_2 = 196
d_model = 128
nhead = 4  
num_encoder_layers = 1
dim_feedforward = 512 

batch_size = 64
learning_rate = 0.0003
epochs = 100
dropout = 0.5

model = TransformerEncoderModel(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, input_dim1)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total number of parameters: {total_params}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
get_data = give_batch()
train_dataset = CustomDataset(torch.tensor(get_data.train_x_1), torch.tensor(get_data.train_y))
test_dataset = CustomDataset(torch.tensor(get_data.test_x_1), torch.tensor(get_data.test_y))

def train(train_dataloader,test_dataloader):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x_1, batch_y in tqdm(train_dataloader):
            batch_x_1 = batch_x_1.long().to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x_1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}')
        
        model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for test_x_1, test_y in test_dataloader:
                test_x_1 = test_x_1.long().to(device)
                test_y = test_y.to(device)
                outputs = model(test_x_1)
                _, predicted = torch.max(outputs.data, 1)
                total += test_y.size(0)
                correct += (predicted == test_y).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}%')
            logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}' + '\n' + str(accuracy) + '\n')
        if (epoch+1) % 50==0:
            torch.save(model, "ckpt/transformer_{}.pth".format(epoch+1))
            print("save model")

def test(train_dataloader,test_dataloader):
    model = torch.load('ckpt/transformer_100.pth')
    model.eval()
    all_preds = []
    all_labels = []
    num_classes = 10
    total = 0
    correct = 0
    with torch.no_grad():
        for test_x_1, test_y in test_dataloader:
            test_x_1 = test_x_1.long().to(device)
            test_y = test_y.to(device)
            outputs,_ = model(test_x_1)
            _, predicted = torch.max(outputs.data, 1)
            
            #all_preds.extend(predicted.cpu().numpy())
            #all_labels.extend(test_y.cpu().numpy())
            total += test_y.size(0)
            correct += (predicted == test_y).sum().item()
        accuracy = 100 * correct / total
        print(accuracy)
        #conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        #print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}%')
        exit()
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()
        #plt.show()

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