import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

home_dir = Path.home()

np.set_printoptions(suppress=True, precision=8)
torch.set_printoptions(sci_mode=False, precision=8)


class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ClassTrainer(): 
    def __init__(self, X_train, y_train, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.model = SimpleNN(in_features = self.X_train.shape[1], num_classes=num_classes).to(self.device)
        self.eta = 0.00001
        self.epochs = 100000
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.eta)
        self.loss_vector = torch.zeros(self.epochs, device=self.device)
        self.accuracy_vector = torch.zeros(self.epochs, device=self.device)
        self.fitted = False

        self.X_test = None
        self.y_test = None
        self.classes_ = None
        self.test_out = None

    def train(self):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.loss(outputs, self.y_train)
            loss.backward()
            
            self.loss_vector[epoch] = loss.item()
            self.optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            self.accuracy_vector[epoch] = accuracy_score(self.y_train.cpu(), preds.cpu())


            if (epoch + 1) % 1000 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')
                print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB')
        self.fitted=True
   
    def test(self, X_test, y_test):
        if not self.fitted:
            raise ValueError("Model must be trained before testing")
        self.X_test=X_test
        self.y_test = y_test
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.loss(outputs, y_test)
            preds = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(y_test.cpu(), preds.cpu())
        self.model.train()
        self.test_out = [loss.item(), accuracy, preds.cpu().numpy()]
        return [loss.item(), accuracy, preds.cpu().numpy()]

    def predict(self,X):
        #predcits for multiple samples, to predict for one make sure X is tensor and unqueeze(0) before passing
        if not self.fitted:
            raise ValueError("Model must be trained before predicting")
        
        self.model.eval()
       
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.model(X)
            pred = torch.argmax(outputs, dim=1)
            # print(pred)
        
        self.model.train()
        return pred.cpu()

    def save(self, fileName: str="multiclass.onnx"):
        if not self.fitted:
            raise ValueError("Model must be trained before saving")
        
        self.model.eval()
        dummy_input = self.X_train[0].unsqueeze(0)  # Adds batch dimension

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,              # Model to export
            dummy_input,            # Example model input
            fileName,           # Output file name
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
        self.model.train()

    def evaluation(self, loss_vector, accuracy_vector):
        self.model.eval()

        if not self.fitted:
            raise ValueError("Model must be trained before plotting loss")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))

        axs[0][0].plot(loss_vector.cpu())
        axs[0][0].set_title('Training Loss')
        axs[0][0].set_xlabel('Epoch')
        axs[0][0].set_ylabel('Loss')
        axs[0][0].grid(True)
        # axs[0].show()

        axs[0][1].plot(accuracy_vector.cpu())
        axs[0][1].set_title('Training Accuracy')
        axs[0][1].set_xlabel('Epoch')
        axs[0][1].set_ylabel('Accuracy')
        axs[0][1].grid(True)
        # axs[1].show()

        y_train_pred = self.predict(self.X_train)
        y_test_pred = self.test_out[2]


        print("Training Metrics:")
        cf_matrix_train = confusion_matrix(self.y_train.cpu(), y_train_pred,labels=self.classes_)
        train_f1 = f1_score(self.y_train.cpu(), y_train_pred, average='macro')
        print("F1 Score:",train_f1)
        train_prec = precision_score(self.y_train.cpu(), y_train_pred, average='macro', zero_division=0)
        print("Precision Score:",train_prec)
        train_recall=recall_score(self.y_train.cpu(), y_train_pred, average='macro')
        print("Recall Score:",train_recall)
        train_acc = accuracy_score(self.y_train.cpu(), y_train_pred)
        print("Accuracy Score:", train_acc)
        # disp_train = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_train,display_labels=self.classes_)
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_train,display_labels=self.classes_)
        disp_train.plot(ax=axs[1][0], colorbar=False)
        axs[1][0].set_title("Confustion Matrix - Training") 



        print("Testing Metrics:")
        cf_matrix_test = confusion_matrix(self.y_test.cpu(), y_test_pred,labels=self.classes_)
        test_f1=f1_score(self.y_test.cpu(), y_test_pred, average='macro')
        print("F1 Score:",test_f1)
        test_prec=precision_score(self.y_test.cpu(), y_test_pred, average='macro', zero_division=0)
        print("Precision Score:",test_prec)
        test_recall = recall_score(self.y_test.cpu(), y_test_pred, average='macro')
        print("Recall Score:",test_recall)
        test_acc=accuracy_score(self.y_test.cpu(), y_test_pred)
        print("Accuracy Score:", test_acc)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_test,display_labels=self.classes_)
        disp_test.plot(ax=axs[1][1], colorbar=False)
        axs[1][1].set_title("Confustion Matrix - Testing") 

        # disp_train.plot()
        # disp_test.plot()
        # plt.show()
        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')
        fileName=date+'_evaluation_plot.pdf'
        plt.savefig(fileName)

        return train_f1,train_prec,train_recall,train_acc,test_f1,test_prec,test_recall,test_acc


class ConvLayer(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(ConvLayer, self).__init__()
        """
        Each block (from diagram)
        Conv(#)
        BatchNorm
        ReLU
        MaxPool(2x2, stride=2)
        """
        self.in_chs = in_chs
        self.out_chs = out_chs
        # First convolutional layer: in/out layers from arguments, 3x3 kernel, and padding of 1, stride of 1
        self.conv1 = nn.Conv2d(self.in_chs, self.out_chs, kernel_size=3, padding=1, stride=1)
        #batch normalization
        self.bn = nn.BatchNorm2d(self.out_chs)
        #relu
        self.relu = nn.ReLU()
        # Max-pooling layer: 2x2 kernel and stride of 2 (reduces spatial dimensions by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ImageNetCNN(nn.Module):
    def __init__(self, num_classes: int=1000, dropout: float=0.5):
        super(ImageNetCNN, self).__init__()
        """
        Blocks:
        1. ConvLayer(3, 64)
        2. ConvLayer(, 128)
        3. ConvLayer(, 256)
        4. ConvLayer(, 512)
        5. ConvLayer(,512)
        6. Global avg pool + flatten (512 feat)
        7. FC1, 1024 nodes,  ReLU+Dropout
        8. FC2, num_classes, softmax/logits
        """
        self.num_classes = num_classes
        self.dropout = dropout

        self.block1 = ConvLayer(3, 64)
        self.block2 = ConvLayer(64, 128)
        self.block3 = ConvLayer(128, 256)
        self.block4 = ConvLayer(256, 512)
        self.block5 = ConvLayer(512, 512)
        self.gapool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(p=self.dropout))
        self.fc2 = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gapool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CNNTrainer():
    def __init__(self, #modified tab spapcing see if this breaks
                train_loader,
                val_loader,
                #train_ratio,
                #val_ratio, 
                num_classes:int=1000, 
                dropout:float=0.5, 
                device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                # eta:float=1e-6, 
                epochs:int=1000, 
                lossfn=None, 
                model=None,
                optimizer=None, 
                scheduler=None,
                loss_vector=None, 
                accuracy_vector=None): #the .todevice in args might break this?
        
        self.device = device
        # self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        # self.y_train = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes=num_classes
        self.dropout= dropout

        self.scheduler = scheduler
        # self.eta = eta
        self.epochs=epochs
        

        if lossfn is None:
            lossfn=nn.CrossEntropyLoss()
        self.loss = lossfn
        
        if model is None:
            model=ImageNetCNN(num_classes=self.num_classes, dropout=self.dropout).to(self.device)
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
        with open(f"{home_dir}/CNNTrainer_multiclass_train_progress.txt", "w") as file:
            file.write(f"Training started at {date}\n" )
        
        for epoch in range(self.epochs):
            with open(f"{home_dir}/CNNTrainer_multiclass_train_progress.txt", "a") as file:
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

            with open(f"{home_dir}/CNNTrainer_multiclass_train_progress.txt", 'a') as file:
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

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch['pixel_values'].to(self.device)
            y = batch['labels'].to(self.device)
            # x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            self.optimizer.zero_grad()

            outputs = self.model(x)
            # logits: (batch_size, seq_length, vocab_size)
            # y:      (batch_size, seq_length)

            # Reshape for CrossEntropyLoss
            loss = self.loss(outputs, y)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            accuracy = 100 * correct / total

            if (batch_idx +1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
                with open(f"{home_dir}/CNNTrainer_multiclass_train_progress.txt", 'a') as file:
                    file.write(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}\n")
                    
        return total_loss / len(self.train_loader), accuracy

        # for batch_idx, (img, label) in enumerate(train_loader):
        #         img, label = img.to(device), label.to(device)
        #         batch_size = img.size(0)

        #         self.optimizer.zero_grad()
        #         outputs = self.model(img) #am im passign the right stuff here???
        #         loss = self.loss(outputs, label)
        #         loss.backward()
                
        #         self.loss_vector[epoch] = loss.item()
        #         total_loss += loss.item()
        #         self.optimizer.step()
                
        #         preds = torch.argmax(outputs, dim=1)
        #         self.accuracy_vector[epoch] = accuracy_score(label.cpu(), preds.cpu())

        #         if (batch_idx % 10 ==0):
        #             print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        # return 

    def val_eval(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['pixel_values'].to(self.device)
                y = batch['labels'].to(self.device)
                # x, y = x.to(self.device), y.to(self.device)
                batch_size = x.size(0)

                outputs = self.model(x)

                loss = self.loss(outputs, y)
                total_loss += loss.item()

                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            accuracy = 100 * correct / total
            
        return total_loss / len(self.val_loader), accuracy

    # def test(self, test_loader):
        if not self.fitted:
            raise ValueError("Model must be trained before testing")
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['pixel_values'].to(self.device) #[pixelvaluse][0]??? for one sample? maybe?
                labels = batch['labels'].to(self.device)
                # images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.loss(outputs, labels)
                total_loss += loss.item()

        accuracy = 100*correct/total
        self.model.train()
        # self.test_out = [loss.item(), accuracy, preds.cpu().numpy()]
        # return [loss.item(), accuracy, preds.cpu().numpy()]
        return total_loss, accuracy, _, predicted

    # def predict(self,X):
        #predcits for single sample
        if not self.fitted:
            raise ValueError("Model must be trained before predicting")
        
        self.model.eval()
        
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        input_tensor = transform(X).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        return predicted_class
        #this returns the class value but will need to translated to actual label

    def save(self, filename:str='ImageNetCNN.onnx'):
        if not self.fitted:
            raise ValueError("Model must be trained before saving")
        
        self.model.eval()
        self.model.to('cpu')
        for batch in self.train_loader:
            x = batch['pixel_values'].to(self.device)
            y = batch['labels'].to(self.device)
            dummy_input = batch['pixel_values'][0].unsqueeze(0).to('cpu')  # Adds batch dimension
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
