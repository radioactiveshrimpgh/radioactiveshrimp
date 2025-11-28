import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort
import numpy as np


class TorchNet:
    """
    A PyTorch-based Neural Network implementation for classification.
    
    Model: Neural Network
    Loss: BCE with Logits
    Optimizer: Adam

    """
    
    def __init__(self, input_dim: int=2, hidden_dim: list=[4,3], output_dim: int =1, lr: float =0.01, max_epoch: int=1000):
        """
        Initialize the Neural Network model.
        
        Args:
            input_dim: dimension of input
            hidden_dim: list of hidden layers containing number of neruons in each layer
            output_dim: dimension of output
            lr: learning rate
            max_epoch: maximum epoch value
        """
        self.model = torch.nn.Sequential()
        # add layers of model in order
        act = 1
        if len(hidden_dim) > 1:
            self.model.add_module('input_layer',torch.nn.Linear(input_dim, hidden_dim[0]))
            self.model.add_module(f'act{act}', torch.nn.ReLU())
            act +=1
            prev_dim = hidden_dim[0]
            for l in range(len(hidden_dim)-1):
                self.model.add_module(f'hidden_{l+1}',torch.nn.Linear(prev_dim, hidden_dim[l+1]))
                self.model.add_module(f'act{act}', torch.nn.ReLU())
                prev_dim = hidden_dim[l+1]
                act+=1
            #add output layer
            self.model.add_module('output_layer',torch.nn.Linear(prev_dim, output_dim))
        elif len(hidden_dim) == 1:
            self.model.add_module('input_layer',torch.nn.Linear(input_dim, hidden_dim[0]))
            self.model.add_module('act1', torch.nn.ReLU())
            self.model.add_module('output_layer',torch.nn.Linear(hidden_dim[0], output_dim))
        else:
            self.model.add_module('input_layer', torch.nn.Linear(input_dim, 1))       
       
        for l in self.model:
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(l.weight,  mode='fan_in',nonlinearity='relu')


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.max_epoch = max_epoch

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        self.loss_history = []
        self.fitted = False
        self.ScalarFit = False
        self.scalar = None

    def standardize(self, X_train, X_test=None):
        # returns the scaled version of X_train and/or X_test

        if self.ScalarFit == False:
            # Define the scaler 
            self.scaler = StandardScaler().fit(X_train)
            self.ScalarFit=True

        # Scale the train set
        X_train = self.scaler.transform(X_train)

        if X_test is None:
            return X_train
        else:
            # Scale the test set
            X_test = self.scaler.transform(X_test)
            return X_train, X_test

    def fit(self, X_train, y_train):
        y_train = y_train.unsqueeze(1)

        for epoch in range(self.max_epoch):
            # Forward pass
            outputs = self.model(X_train)
            loss = self.loss_fn(outputs, y_train)
            self.loss_history.append(loss.detach().item())
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        self.fitted = True

    def plot_loss(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting loss")

        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('BCE with Logits Loss')
        plt.grid(True)
        plt.show()

    def save_ONNX(self, X_train:np.ndarray, out_file_name: str="saved_model.onnx"):
        self.model.eval()
        dummy_input = X_train[0].unsqueeze(0)  # Adds batch dimension

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,              # Model to export
            dummy_input,            # Example model input
            out_file_name,           # Output file name
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
            
    def predict(self, new_df, load_file_name: str="saved_model.onnx"):
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        self.model.eval()
        ort_session = ort.InferenceSession(load_file_name)
        # Run inference
        logits = ort_session.run(None, {'input': new_df.astype(np.float32)})[0]
        print(logits)
        sig = 1 / (1 + np.exp(-logits))
        pred_indices = (sig >= 0.5).astype(int)
        print(sig)
        self.model.train()
        return pred_indices


#--------------------------------------------------------------------------------------------------

#demo the use....
import polars as p
import pandas as pd
from sklearn.model_selection import train_test_split

data = p.read_csv('/home/radioactiveshrimp/datasets/diabetes.csv')
X = data.drop('Outcome').to_numpy()
X = torch.tensor(X, dtype=torch.float32)
y = data.get_column("Outcome").to_numpy()
y = torch.tensor(y,dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

tnModel = TorchNet(X_train.shape[1], [4,3],1,0.01,1000)
print(tnModel.fitted)
X_train, X_test=tnModel.standardize(X_train, X_test)

X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)

# print(type(X_train), type(X_test))

tnModel.fit(X_train, y_train)
print(tnModel.fitted)

tnModel.plot_loss()

tnModel.save_ONNX(X_train)

newData = pd.DataFrame({
    "Pregnancies":[7],
    "Glucose":[149],
    "BloodPressure":[73],
    "SkinThickness":[94],
    "Insulin":[94],
    "BMI":[32],
    "DiabetesPedigreeFunction":[0.672],
    "Age":[45]
})

# newData = pd.DataFrame({
#     "Pregnancies":[5],
#     "Glucose":[139],
#     "BloodPressure":[64],
#     "SkinThickness":[35],
#     "Insulin":[140],
#     "BMI":[28.6],
#     "DiabetesPedigreeFunction":[0.441],
#     "Age":[26]
# })

newData = tnModel.standardize(newData)

preds = tnModel.predict(newData)
print(preds)


# print(tnModel.model)
# 7	114	66	0	0	32.8	0.258	42	1
# 4	146	85	27	100	28.9	0.189	27	0
# 2	100	66	20	90	32.9	0.867	28	1
# 13	126	90	0	0	43.4	0.583	42	1
# 4	129	86	20	270	35.1	0.231	23	0

