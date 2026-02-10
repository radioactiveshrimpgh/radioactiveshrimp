import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime


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

