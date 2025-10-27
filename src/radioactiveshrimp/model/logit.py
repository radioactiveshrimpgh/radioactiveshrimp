import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report

class LogisticRegression(nn.Module):
    def __init__(self, num_features:int = 3, learning_rate: float=.01, max_epochs:int=1000):
        super(LogisticRegression, self).__init__()
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.X_train = None
        self.y_train = None

        self.w = None
        self.b = None

        self.linear = nn.Linear(self.num_features,1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), self.learning_rate)

        self.fitted = False

    def forward(self,x):
        return self.sigmoid(self.linear(x))
    
    def fit(self, x_train, y_train, x_test, y_test):
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.max_epochs):

            self.optimizer.zero_grad()
            y_pred = self.forward(self.X_train)
            loss = self.criterion(y_pred, self.y_train)

            loss.backward()

            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Loss: {loss.item():.4f}')

        self.fitted = True

        self.w = self.linear.weight.detach().numpy()
        self.b = self.linear.bias.item()
        return self


    def predict(self, X, boundry:float=.5):
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
            predictions = (predictions >= boundry).float()
        
        return predictions
    
    def prediction_accuracy(self, pred, y_test):
        print('The accuracy of the Logistic Regression is',metrics.accuracy_score(pred,y_test))
        confusion_matrix = metrics.confusion_matrix(y_test, pred)
        print("Confusion Matrix:\n", confusion_matrix)
        print("Classification Report:\n",classification_report(y_test, pred))

    def get_parameters(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        return (self.w,self.b)
    
    def class_imbalace_check(self, data, label):
        sns.countplot(x=label, data=data)
        plt.title('Class Distribution')
        plt.show()


#------------------------------------------------------------------------------------
# import polars as p
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder

# # import data set (spotify)
# data = p.read_csv('/home/radioactiveshrimp/datasets/spotify_churn_dataset.csv')
# X = data[['age','listening_time', "skip_rate"]]
# y = data[['is_churned']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# # Define the scaler 
# scaler = StandardScaler().fit(X_train)
# # Scale the train set
# X_train = scaler.transform(X_train)
# # Scale the test set
# X_test = scaler.transform(X_test)

# # Reshape y_train and y_test to be 1D arrays
# y_train_label = np.ravel(y_train)
# y_test_label = np.ravel(y_test)

# encoder = LabelEncoder()
# encoder.fit(y_train_label)
# y_train = encoder.transform(y_train_label)
# y_test = encoder.transform(y_test_label)

# print(np.unique(y_train_label))

# # print(X.shape[1])
# L = LogisticRegression(num_features=X.shape[1])
# model = L.fit(X_train,y_train,X_test,y_test)
# print(L.get_parameters())
# pred = L.predict(X_test)
# print("pred:", pred)
# L.prediction_accuracy(pred,y_test)
# L.class_imbalace_check(data, 'is_churned')

#-------------------------------------------------------------------------------
# trying with other dataset....
# import polars as p
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder

# # import data set (spotify)
# data = p.read_csv('/home/radioactiveshrimp/datasets/defaultCreditCards.csv')
# data=data[1:]
# X = data[['X1','X2', 'X3', 'X4','X5','X12','X18']]
# y = data[['Y']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# # Define the scaler 
# scaler = StandardScaler().fit(X_train)
# # Scale the train set
# X_train = scaler.transform(X_train)
# # Scale the test set
# X_test = scaler.transform(X_test)

# # Reshape y_train and y_test to be 1D arrays
# y_train_label = np.ravel(y_train)
# y_test_label = np.ravel(y_test)

# encoder = LabelEncoder()
# encoder.fit(y_train_label)
# y_train = encoder.transform(y_train_label)
# y_test = encoder.transform(y_test_label)

# print(np.unique(y_train_label))

# # print(X.shape[1])
# L = LogisticRegression(num_features=X.shape[1])
# model = L.fit(X_train,y_train,X_test,y_test)
# print(L.get_parameters())
# pred = L.predict(X_test, .335)
# print("pred:", pred)
# L.prediction_accuracy(pred,y_test)
# L.class_imbalace_check(data, 'Y')
