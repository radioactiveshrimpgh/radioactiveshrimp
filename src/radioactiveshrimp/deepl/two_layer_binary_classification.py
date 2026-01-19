import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class binary_classification():
    def __init__(self,d, n, epochs: int=10000, eta: float=0.001):
        self.d = d #number of features
        self.n = n # number of samples
        self.epochs = epochs #number of epochs
        self.eta = eta #learning rate
        self.tolerance = 1e-6 # tolerance for convergence

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.device = None
        self.loss_history = []
        self.fitted = False

    def generateMatrix(self):
        if self.device is not None:
            X = torch.randn(self.n, self.d, dtype=torch.float32, device=self.device)
        else:
            X = torch.randn(self.n, self.d, dtype=torch.float32)
        self.X = X
        return
    
    def generateLabels(self):
        y = []
        row_sums = torch.sum(self.X, dim=1)
        # print(row_sums)
        for sample in range(len(row_sums)):
            if row_sums[sample] > 2:
                y.append(1)
            else:
                y.append(0)
        # print("y len:", len(y))
        # print(y)
        if self.device is not None:
            self.y=torch.tensor(y, device=self.device)
        else:
            self.y=torch.tensor(y)
        # return y
        return 
    
    def initializeWeights(self):
        W1 = np.random.normal(0, np.sqrt(2/self.d), size=(self.d,48))
        W2 = np.random.normal(0, np.sqrt(2/48), size=(48,16))
        W3 = np.random.normal(0, np.sqrt(2/16), size=(16,32))
        W4 = np.random.normal(0, np.sqrt(2/32), size=(32,1))

        if self.device is not None:
            self.W1 = torch.as_tensor(W1, dtype=torch.float32, device=self.device).requires_grad_()
            self.W2 = torch.as_tensor(W2, dtype=torch.float32, device=self.device).requires_grad_()
            self.W3 = torch.as_tensor(W3, dtype=torch.float32, device=self.device).requires_grad_()
            self.W4 = torch.as_tensor(W4, dtype=torch.float32, device=self.device).requires_grad_()
        else:
            self.W1 = torch.as_tensor(W1, dtype=torch.float32).requires_grad_()
            self.W2 = torch.as_tensor(W2, dtype=torch.float32).requires_grad_()
            self.W3 = torch.as_tensor(W3, dtype=torch.float32).requires_grad_()
            self.W4 = torch.as_tensor(W4, dtype=torch.float32).requires_grad_()
        # print(type(W1))
        # print(self.W1.requires_grad)
        # print(W2)
        # print(W3)
        # print(W4)
        return

    def fit(self):
        y = self.y.unsqueeze(1).float()
        prev_loss = float('inf')
        for t in range(self.epochs):
            # XW1 ->sigmoid w/ W2 = A1
            # A1*W3 -> sigmoid w/W4 = A2 = y_pred
            sig = torch.nn.Sigmoid()
            Z1 = (self.X).mm(self.W1)
            A1 = sig(Z1.mm(self.W2))
            Z2 = A1.mm(self.W3)
            logit = Z2.mm(self.W4)

            # Compute and print loss
            loss = self.criterion(logit, y)
            # print(loss)

            current_loss=loss.item()
            self.loss_history.append(current_loss)

            if t % 100 == 99:
                print(f"Iteration {t+1}: {current_loss}")

            # Use autograd to compute the backward pass.
            # This computes the gradient of loss with respect to all Tensors with requires_grad=True.
            loss.backward()

            # Manually update weights using gradient descent. 
            # Wrap in torch.no_grad() because weights have requires_grad=True, 
            # but we don't need to track this step in autograd.
            with torch.no_grad():
                self.W1 -= self.eta * self.W1.grad
                self.W2 -= self.eta * self.W2.grad
                self.W3 -= self.eta * self.W3.grad
                self.W4 -= self.eta * self.W4.grad

                # Manually zero the gradients after updating weights
                self.W1.grad.zero_()
                self.W2.grad.zero_()
                self.W3.grad.zero_()
                self.W4.grad.zero_()

            # Check for convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {t + 1} epochs")
                break
            
            prev_loss = current_loss

        self.fitted = True
        return self.W1, self.W2, self.W3, self.W4, self.loss_history
    
    def checkGPU(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU available, using GPU")
            self.device = device
        else:
            print("GPU not available, using CPU")
            self.device = None
        return

    def plotLoss(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting loss")
        
        # print(self.loss_history)
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        # plt.show()
        return plt
