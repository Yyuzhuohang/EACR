import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import sys

class MyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

    def __len__(self):
        return len(self.sequences)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)  
     
    def forward(self, x):   
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, cnn_model, input_size=784, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.cnn_model = cnn_model
        self.hidden = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        cnn_output = self.cnn_model(x)   
        cnn_output = cnn_output.view(cnn_output.size(0), -1) 
        x = F.relu(self.hidden(cnn_output))
        x = self.output(x)
        return x

def train_and_test(mode):
    batch = 16
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    def process_data(path):
        data = []
        label = []
        all_folders = os.listdir(path) 
        for idx, folder in enumerate(all_folders):
            numbers = os.listdir(os.path.join(path, folder))  
            for number in numbers:
                img = cv.imread(os.path.join(path, folder, number), 0)  
                img = np.expand_dims(img, axis=0)  
            
                data.append(img)
                label.append(idx)
        return data, label

    train_data, train_label = process_data('DL.TRAIN/')
    test_data, test_label = process_data('DL.TEST/')

    train_dataset = MyDataset(train_data, train_label)
    test_dataset = MyDataset(test_data, test_label)

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)  
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    cnn_model = CNN().to(device)
    mlp_model = MLP(cnn_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    accuracy = []

    if mode == 'train':
        for epoch in range(num_epochs):
            mlp_model.train()
            correct = 0
            total = 0
            for images, labels in train_dataloader:
                labels = labels.to(device)
                images = images.to(device).float()
                
                outputs = mlp_model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

            mlp_model.eval()
            accu_test = 0
            for images, labels in test_dataloader:
                labels = labels.to(device)
                images = images.to(device).float()
                with torch.no_grad():
                    outputs = mlp_model(images)

                accu_test += (outputs.argmax(1) == labels).float().sum()

            accuracy.append(accu_test.cpu().numpy() / len(test_label))
        plt.plot(accuracy)
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

        torch.save(mlp_model.state_dict(), 'cnn+mlp_model.pth')

    elif mode == 'test':
        mlp_model.load_state_dict(torch.load('cnn+mlp_model.pth'))
        mlp_model.eval()

        accu_test = 0
        for images, labels in test_dataloader:
            labels = labels.to(device)
            images = images.to(device).float()
            with torch.no_grad():
                outputs = mlp_model(images)

            accu_test += (outputs.argmax(1) == labels).float().sum()

        accuracy = accu_test / len(test_label)
        print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        print("Usage: python script.py [train/test]")
    else:
        train_and_test(sys.argv[1])
