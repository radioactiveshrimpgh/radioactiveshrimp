import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import glob
import csv
import os
import polars as pl
from sklearn.preprocessing import StandardScaler

from pathlib import Path

home_dir = Path.home()


np.set_printoptions(suppress=True, precision=8)
torch.set_printoptions(sci_mode=False, precision=8)

class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride:int=1):
        super(ResBlock, self).__init__()
        """
        Each block:
        Conv1D
        BatchNorm
        ReLU
        Conv1d
        BatchNorm
        SKIP CONNECT
        ReLU
        """
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride=stride
        # First convolutional layer: in/out layers from arguments 
        self.conv1 = nn.Conv1d(self.in_chs, self.out_chs, kernel_size=3, padding=1, stride=self.stride)
        #batch normalization
        self.bn1 = nn.BatchNorm1d(self.out_chs)
        #relu
        self.relu1 = nn.ReLU()
        # Scond convolutional layer: in/out layers from arguments 
        self.conv2 = nn.Conv1d(self.out_chs, self.out_chs, kernel_size=3, padding=1, stride=1)
        #seocnd batch normalization 
        self.bn2 = nn.BatchNorm1d(self.out_chs)
        #perform check for same size
        self.downsample = (
            nn.Conv1d(in_chs, out_chs, kernel_size=1, stride=self.stride)
            if in_chs != out_chs else nn.Identity()
        )
        #final activation post skip connection
        self.relu2 = nn.ReLU()

    def forward(self, x):
        initial_x = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + initial_x #need to perform check to make sure same size?
        x = self.relu2(x)
        return x


class ACCNet(nn.Module):
    def __init__(self, drop: float=0.0): #num classes = 2 (binary classification)
        super(ACCNet, self).__init__()
        """
        Blocks:
        1. Conv1D
        2. ResBlock1
        3. ResBlock2
        4. ResBlock3
        5. ResBlock4
        6. Global avg pool
        8. FC 
        """
        self.num_classes = 2 #binary classificaion - either ACC enabled or not
        self.drop = drop

        self.block1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.block2 = ResBlock(64, 64, stride=1)
        self.block3 = ResBlock(64,128,stride=2)
        self.block4 = ResBlock(128,256,stride=2) #if takes too long try reduucing to 3 resblocks with last as (128,128,stride=1)?
        self.block5 = ResBlock(256,256,stride=1)
        self.gapool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=self.drop)
        # self.fc1 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(p=self.dropout))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gapool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

class ACCTrainer():
    def __init__(self,
        train_loader,
        val_loader,
        #train_ratio,
        #val_ratio, 
        # num_classes:int=1000, 
        dropout:float=0.0, 
        device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        # eta:float=1e-6, 
        epochs:int=1000, 
        lossfn=None, 
        model=None,
        optimizer=None, 
        scheduler=None,
        loss_vector=None, 
        accuracy_vector=None):
        
        self.device = device
        # self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        # self.y_train = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.num_classes=num_classes
        self.dropout= dropout

        self.scheduler = scheduler
        # self.eta = eta
        self.epochs=epochs


        if lossfn is None:
            # lossfn=nn.BCEWithLogitsLoss()
            lossfn=FocalTverskyLoss()
        self.loss = lossfn
        
        if model is None:
            model=ACCNet(dropout=self.dropout).to(self.device)
        self.model = model.to(self.device)
        
        if optimizer is None:
            optimizer=optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.optimizer = optimizer

        if scheduler is None:
            scheduler=optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)    
        self.scheduler = scheduler    
        
        if loss_vector is None:
            loss_vector=torch.zeros(self.epochs, device=self.device)
        self.train_loss_vector = loss_vector
        self.val_loss_vector = torch.zeros(self.epochs, device=self.device)

        if accuracy_vector is None:
            accuracy_vector=torch.zeros(self.epochs, device=self.device)
        self.train_accuracy_vector = accuracy_vector
        self.val_accuracy_vector = torch.zeros(self.epochs, device=self.device)
        
        self.fitted = False

        self.X_test = None
        self.y_test = None
        self.classes_ = None
        self.test_out = None

    def train(self): #takes no args (hw2)
        #should use training and validation DataLoader showin in hw03 assignment
        #print epoch num for every 10th batch, every epoch, loss+training accuracy, validation accuracy
        self.model.train()
        total_loss = 0

        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')
        with open(f"{home_dir}/ACCTrainer_train_progress.txt", "w") as file:
            file.write(f"Training started at {date}\n" )
        
        for epoch in range(self.epochs):
            with open(f"{home_dir}/ACCTrainer_train_progress.txt", "a") as file:
                file.write(f"\nEpoch {epoch+1}/{self.epochs}\n")
                file.write("-" * 40+"\n")

            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)

            avg_train_loss, train_accuracy = self.train_epoch()
            self.train_loss_vector[epoch]=avg_train_loss
            self.train_accuracy_vector[epoch]=train_accuracy
            
            avg_val_loss, val_acc = self.val_eval()
            self.val_loss_vector[epoch]=avg_val_loss
            self.val_accuracy_vector[epoch]=val_acc
            # self.val_accuracy_vector[epoch] = accuracy_score(label.cpu(), preds.cpu())

            self.scheduler.step()

            with open(f"{home_dir}/ACCTrainer_train_progress.txt", 'a') as file:
                file.write(f"Train Loss: {avg_train_loss:.4f}\n")
                file.write(f"Train Accuracy: {train_accuracy:.4f}\n")
                file.write(f"Val Loss: {avg_val_loss:.4f}\n")
                file.write(f"Val Accuracy: {val_acc:.4f}\n")

            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")

        self.fitted=True

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x,y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            # batch_size = x.size(0)

            self.optimizer.zero_grad()

            outputs = self.model(x)

            # Reshape for CrossEntropyLoss
            loss = self.loss(outputs, y)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                pred = (torch.sigmoid(outputs)>=0.5).float()
                total += y.size(0)
                correct += (pred == y).sum().item()

            accuracy = 100 * correct / total

            if (batch_idx +1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
                with open(f"{home_dir}/ACCTrainer_train_progress.txt", 'a') as file:
                    file.write(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}\n")
                    
        return total_loss / len(self.train_loader), accuracy

    def val_eval(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x,y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)

                loss = self.loss(outputs, y)
                total_loss += loss.item()

                with torch.no_grad():
                    pred = (torch.sigmoid(outputs)>=0.5).float()
                    total += y.size(0)
                    correct += (pred == y).sum().item()
            
            accuracy = 100 * correct / total
            
        return total_loss / len(self.val_loader), accuracy

    def save(self, filename:str='ACCNetModel.onnx'): #taken direct from CNNTrainer may need modification
        if not self.fitted:
            raise ValueError("Model must be trained before saving")
        
        self.model.eval()
        self.model.to('cpu')
        for x,y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            dummy_input = x[0].unsqueeze(0).to('cpu')  # Adds batch dimension
            break

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,              # Model to export
            dummy_input,            # Example model input
            filename,           # Output file name
            export_params=True,     # Store trained parameters
            opset_version=11,       # ONNX opset version
            do_constant_folding=True,  # Optimize constants
            input_names=['input'],  # Input tensor name
            output_names=['output'], # Output tensor name
            dynamic_axes={          # Allow dynamic batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        self.model.to(self.device)
        self.model.train()

    def evaluation(self):
        self.model.eval()

        if not self.fitted:
            raise ValueError("Model must be trained before plotting loss")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))

        axs[0][0].plot(self.train_loss_vector.cpu())
        axs[0][0].set_title('Training Loss')
        axs[0][0].set_xlabel('Epoch')
        axs[0][0].set_ylabel('Loss')
        axs[0][0].grid(True)
        # axs[0].show()

        axs[0][1].plot(self.train_accuracy_vector.cpu())
        axs[0][1].set_title('Training Accuracy')
        axs[0][1].set_xlabel('Epoch')
        axs[0][1].set_ylabel('Accuracy')
        axs[0][1].grid(True)
        # axs[1].show()

        axs[1][0].plot(self.val_loss_vector.cpu())
        axs[1][0].set_title('Validation Loss')
        axs[1][0].set_xlabel('Epoch')
        axs[1][0].set_ylabel('Loss')
        axs[1][0].grid(True)
        # axs[0].show()

        axs[1][1].plot(self.val_accuracy_vector.cpu())
        axs[1][1].set_title('Validation Accuracy')
        axs[1][1].set_xlabel('Epoch')
        axs[1][1].set_ylabel('Accuracy')
        axs[1][1].grid(True)
        # axs[1].show()

        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')
        fileName=date+'_evaluation_plot.pdf'
        plt.savefig(fileName)

        print("training/validation loss and accuracy plots saved to file", fileName)
        return

class ACCDataset(Dataset):
    def __init__(self, X, y, train_ratio=0.01, val_ratio=0.005, split:str="train", scaler=None):

        if split.lower() not in ["train", "val", "validaion", "test"]:
            raise ValueError(f"split type {split} not valid type train, validation, or test")
            exit()
        elif (split.lower() != "train") and (scaler is None):
            raise ValueError(f"provide StandardScaler from training set for normalization")
            exit()

        self.og_X = X
        self.og_y = y
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_type = split
        self.scaler = scaler

        ds_size = len(self.og_X)
        test_ratio = 1-(self.train_ratio+self.val_ratio)
        #might need to turn off shuffle?
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
        if self.split_type == "test":
            self.X = self.scaler.transform(X_test)
            self.y = y_test
        else:
            val_ratio = self.val_ratio/(self.train_ratio+self.val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, random_state=0)
            
            if self.split_type == "train":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                self.scaler = scaler
                self.X = X_train
                self.y = y_train
            else:
                self.X = self.scaler.transform(X_val)
                self.y = y_val

        print(f"end of ACCDataset init, split {split}...")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X=torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y=torch.tensor(self.y[idx], dtype=torch.float32)

        return (X,y)

#The following class implementation of Focal Tversky Loss was aquired from google AI result. 
# The search was "focal tversky loss 1d python" and google convineiently produced this output as the very first result

class FocalTverskyLoss(nn.Module): 
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Penalty for false positives
        self.beta = beta    # Penalty for false negatives (recall)
        self.gamma = gamma  # Focal parameter (gamma < 1 focuses on hard examples)
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: (Batch, Channel, Length) - Probability predictions (after sigmoid)
        y_true: (Batch, Channel, Length) - Ground truth binary mask
        """
        y_pred = torch.sigmoid(y_pred) #modified to prevent having to  make changes in loss function ( more modular?)
        # Flatten the 1D sequences
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = torch.sum(y_true * y_pred)
        FP = torch.sum((1 - y_true) * y_pred)
        FN = torch.sum(y_true * (1 - y_pred))
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal Tversky Loss
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)
        
        return focal_tversky_loss

# Usage Example
# model_output = torch.sigmoid(model_1d(inputs))
# criterion = FocalTverskyLoss()
# loss = criterion(model_output, targets)
