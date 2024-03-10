import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from util import give_batch,CustomDataset,get_test_acc_regression,find_intersection,plt_hist,dra_fit,plt_bar,give_class,assign_to_bins,saveval_file
import os
from tqdm import tqdm
import sys
from model import SwinTransformer
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim1 = 22  # Example vocabulary size for sequences1
input_dim2 = 25   # Example vocabulary size for sequences2
seq_len_1 = 1200
seq_len_2 = 85
hidden_dim = 128
num_heads = 4
num_layers = 6
dim = 128
patch_size = 9

batch_size = 64
learning_rate = 0.0003
epochs = 50
dropout_prob = 0.5

model = SwinTransformer(image_size=(seq_len_2, seq_len_1), patch_size=patch_size, dim=dim, num_heads=num_heads, num_layers=num_layers, input_dim1=input_dim1, input_dim2=input_dim2, hidden_dim=hidden_dim, dropout_prob=dropout_prob,
                        embed_dim=seq_len_1)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

get_data = give_batch()

train_dataset = CustomDataset(torch.tensor(get_data.train_x_1), torch.tensor(get_data.train_x_2), torch.tensor(get_data.train_y))
test_dataset = CustomDataset(torch.tensor(get_data.test_x_1), torch.tensor(get_data.test_x_2), torch.tensor(get_data.test_y))

def train(train_dataloader,test_dataloader):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x_1, batch_x_2, batch_y in tqdm(train_dataloader):
            batch_x_1 = batch_x_1.to(device)
            batch_x_2 = batch_x_2.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions,_ = model(batch_x_1,batch_x_2)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}')
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            label = []
            pre = []
            for test_x_1, test_x_2, test_y in test_dataloader:
                test_x_1 = test_x_1.to(device)
                test_x_2 = test_x_2.to(device)
                test_y = test_y.to(device)
                outputs,_ = model(test_x_1, test_x_2)
                label.extend(test_y)
                pre.extend(outputs)
            label = torch.cat(label, dim=-1)
            pre = torch.cat(pre, dim=-1)
            to_print = get_test_acc_regression(label,pre,'Test:')
            logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}' + '\n' + to_print + '\n')
        if (epoch+1) % 50==0:
            torch.save(model, "transformer_{}.pth".format(epoch+1))
            print("save model")

def test(train_dataloader,test_dataloader):
    model = torch.load('ckpt/transformer_950.pth')
    model.eval()
    with torch.no_grad():
        label = []
        pre = []
        ecod = []
        mat_1 = []
        mat_2 = []
        dict_1 = {}
        dict_2 = {}
        for test_x_1, test_x_2, test_y in test_dataloader:
            test_x_1 = test_x_1.to(device)
            test_x_2 = test_x_2.to(device)
            test_y = test_y.to(device)
            outputs,mat = model(test_x_1, test_x_2)
            mat_ = mat.squeeze()
            mat_ = torch.sum(mat_,dim=-1).cpu().numpy()
            mat_1.extend(mat_)
            label.extend(test_y)
            pre.extend(outputs)
        label = torch.cat(label, dim=-1).cpu().numpy()
        pre = torch.cat(pre, dim=-1).cpu().numpy()
        with open("result/shiyan.txt",'w',encoding="utf-8")as f, open('result/shiyan1.txt','w',encoding="utf-8")as f1:
            for amn,smi,p,m in zip(get_data.amnio_,get_data.new_test_smi,pre,mat_1):
                s = ','.join([amn,smi,str(p)])
                m = m[:len(smi)]
                mm = ','.join([str(x) for x in m])
                ss = '<=>'.join([smi,mm])
                f.write(s+'\n')
                f1.write(ss+'\n')
        to_print = get_test_acc_regression(label,pre,'Test:')
        
if __name__=="__main__":
    if sys.argv[1] == 'train':
        # Configure the logging settings
        logging.basicConfig(level=logging.INFO,  # Set the logging level to INFO
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of log messages
                            filename='log.log',  # File to save logs
                            filemode='w')  # Overwrite the log file each time
        logger = logging.getLogger(__name__)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train(train_dataloader,test_dataloader)
    if sys.argv[1] == 'test':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test(train_dataloader,test_dataloader)