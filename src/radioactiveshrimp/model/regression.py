import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import math as m
import seaborn as sns
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

class LinearRegression:
    """
    A PyTorch-based Linear Regression implementation for one variable.
    
    Model: y = w_1 * x + w_0
    Loss: Mean Squared Error
    
    Features:
    - Gradient-based optimization using PyTorch
    - Confidence intervals for parameters w_1 and w_0
    - Visualization with confidence bands
    """
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, 
                 tolerance: float = 1e-6):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Model parameters
        self.w_1 = nn.Parameter(torch.randn(1, requires_grad=True))  # slope
        self.w_0 = nn.Parameter(torch.randn(1, requires_grad=True))  # intercept
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        
        # Model statistics for confidence intervals
        self.n_samples = None
        self.residual_sum_squares = None
        self.X_mean = None
        self.X_var = None
        self.fitted = False
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD([self.w_1, self.w_0], lr=self.learning_rate)
        
        # Training history
        self.loss_history = []
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.
        
        Args:
            X: Input tensor of shape (n_samples,)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        return self.w_1 * X + self.w_0
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_x:np.ndarray=None, test_y:np.ndarray=None) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.
        
        Args:
            X: Input features of shape (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns the fitted model instance
        """
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.n_samples = len(X)
        
        # # Store statistics for confidence intervals
        # self.X_mean = float(np.mean(X))
        # self.X_var = float(np.var(X, ddof=1))  # Sample variance
        
        # Training loop
        prev_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.forward(self.X_train)
            
            # Compute loss
            loss = self.criterion(y_pred, self.y_train)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Store loss history
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            # Check for convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break
            
            prev_loss = current_loss
        
        # Compute residual sum of squares for confidence intervals
        with torch.no_grad():
            y_pred = self.forward(self.X_train)
            residuals = self.y_train - y_pred
            self.residual_sum_squares = float(torch.sum(residuals ** 2))
        
        self.fitted = True

        # compute r^2 on optional test data, print result
        if ((test_x is not None) and (test_y is not None)):
            y_mean = float(torch.mean(self.y_train))
            ss_tot = float(torch.sum((self.y_train - y_mean) ** 2))
            r_squared = 1 - (self.residual_sum_squares / ss_tot)
            print('R^2 for test data: ',r_squared)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples,)
            
        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()
    
    def get_parameters(self) -> Tuple[float, float]:
        """
        Get the fitted parameters.
        
        Returns:
            Tuple of (w_1, w_0) - slope and intercept
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        
        return float(self.w_1.item()), float(self.w_0.item())
    
    def parameter_confidence_intervals(self, confidence_level: float = 0.95) -> dict:
        """
        Compute confidence intervals for parameters w_1 and w_0.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing confidence intervals for both parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before computing confidence intervals")
        
        # Degrees of freedom
        df = self.n_samples - 2
        
        # Critical t-value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Standard error of regression
        mse = self.residual_sum_squares / df
        se_regression = np.sqrt(mse)
        
        # Standard error for w_1 (slope)
        se_w1 = se_regression / np.sqrt(self.n_samples * self.X_var)
        
        # Standard error for w_0 (intercept)
        se_w0 = se_regression * np.sqrt(1/self.n_samples + self.X_mean**2 / (self.n_samples * self.X_var))
        
        # Get current parameter values
        w_1_val, w_0_val = self.get_parameters()
        
        # Compute confidence intervals
        w_1_ci = (
            w_1_val - t_critical * se_w1,
            w_1_val + t_critical * se_w1
        )
        
        w_0_ci = (
            w_0_val - t_critical * se_w0,
            w_0_val + t_critical * se_w0
        )
        
        return {
            'w_1_confidence_interval': w_1_ci,
            'w_0_confidence_interval': w_0_ci,
            'confidence_level': confidence_level,
            'standard_errors': {
                'se_w1': se_w1,
                'se_w0': se_w0,
                'se_regression': se_regression
            }
        }
    
    def plot_regression_with_confidence_band(self, confidence_level: float = 0.95, 
                                           figsize: Tuple[int, int] = (10, 6),
                                           title: Optional[str] = None) -> plt.Figure:
        """
        Plot the fitted regression line with confidence band.
        
        Args:
            confidence_level: Confidence level for the band
            figsize: Figure size tuple
            title: Optional plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert training data to numpy for plotting
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        
        # Create prediction range
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)
        
        # Compute confidence band
        df = self.n_samples - 2
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mse = self.residual_sum_squares / df
        se_regression = np.sqrt(mse)
        
        # Standard error for predictions (confidence band)
        X_centered = X_range - self.X_mean
        se_pred = se_regression * np.sqrt(1/self.n_samples + X_centered**2 / (self.n_samples * self.X_var))
        
        # Confidence band bounds
        margin_of_error = t_critical * se_pred
        y_upper = y_pred_range + margin_of_error
        y_lower = y_pred_range - margin_of_error
        
        # Plot data points
        ax.scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')
        
        # Plot regression line
        ax.plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
        
        # Plot confidence band
        ax.fill_between(X_range, y_lower, y_upper, alpha=0.3, color='red', 
                       label=f'{int(confidence_level*100)}% Confidence band')
        
        # Get parameter values for display
        w_1_val, w_0_val = self.get_parameters()
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        if title is None:
            title = f'Linear Regression: y = {w_1_val:.3f}x + {w_0_val:.3f}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> dict:
        """
        Provide a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating summary")
        
        w_1_val, w_0_val = self.get_parameters()
        
        # R-squared calculation
        y_mean = float(torch.mean(self.y_train))
        ss_tot = float(torch.sum((self.y_train - y_mean) ** 2))
        r_squared = 1 - (self.residual_sum_squares / ss_tot)
        
        # Adjusted R-squared
        adj_r_squared = 1 - ((1 - r_squared) * (self.n_samples - 1) / (self.n_samples - 2))
        
        # RMSE
        rmse = np.sqrt(self.residual_sum_squares / self.n_samples)
        
        return {
            'parameters': {
                'w_1 (slope)': w_1_val,
                'w_0 (intercept)': w_0_val
            },
            'model_fit': {
                'r_squared': r_squared,
                'adjusted_r_squared': adj_r_squared,
                'rmse': rmse,
                'residual_sum_squares': self.residual_sum_squares
            },
            'training_info': {
                'n_samples': self.n_samples,
                'epochs_trained': len(self.loss_history),
                'final_loss': self.loss_history[-1] if self.loss_history else None
            }
        }

    def analysis_plot(self, figsize: Tuple[int, int] = (10, 6),
                            title: Optional[str] = None) -> plt.Figure:
        """
        Plot the fitted regression line.
        
        Args:
            figsize: Figure size tuple
            title: Optional plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(3,1,figsize=figsize)
        
        # Convert training data to numpy for plotting
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        
        # Create prediction range
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)
        
        # Plot data points
        ax[0].scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')
        
        # Plot regression line with original data
        ax[1].plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
        ax[1].scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')

        
        # Get parameter values for display
        w_1_val, w_0_val = self.get_parameters()

        # Plot loss as training occured
        ax[2].plot(self.loss_history, label='Training Loss', color='green')
        
        # Labels and title
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('y')
        ax[0].set_title("Original Data")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel('X')
        ax[1].set_ylabel('y')
        if title is None:
            title = f'Linear Regression: y = {w_1_val:.3f}x + {w_0_val:.3f}'
        ax[1].set_title(title)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Loss')
        ax[2].set_title('Loss Though Training')
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig

class CauchyRegression(nn.Module):
    """
    A PyTorch-based Multiple Linear Regression implementation for multiple variables.
    
    Model: y = w_0 + w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + w_4 * x_4 
    Loss: Cauchy Loss Function
    
    Features:
    - Gradient-based optimization using PyTorch
    - Confidence intervals for parameters w_1 and w_0
    - Visualization with confidence bands
    """
    def __init__(self, learning_rate: float = 0.00001, max_epochs: int = 100000, 
                 input_dim: int = 1,
                 cauchyLossRate: float = 1,
                 tolerance: float = 1e-6):
        """
        Initialize the Cauchy Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            imput_dim: dimention of the input
            cauchyLossRate: Cauchy Loss Function Constant for loss function
            tolerance: Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.input_dim = input_dim
        self.cauchyLossRate = cauchyLossRate
        self.tolerance = tolerance
        
        # Model parameters
        self.W = torch.randn(input_dim, 1, requires_grad=True, dtype=torch.float32)  # Random initialization for weights
        self.b = torch.randn(1, requires_grad=True, dtype=torch.float32) # Random initialization for bias
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        
        self.fitted = False
        
        # Loss function and optimizer
        self.criterion = self.cauchyLoss
        
        # Training history
        self.loss_history = []
        self.test_loss = None
    
    def model(self, X: torch.tensor):
        return X @ self.W + self.b

    def cauchyLoss(self, Y_pred: torch.Tensor, Y_actual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.
        
        Args:
            Y_pred: Input tensor of predicted values of shape (n_samples,)
            Y_actual: Input tensor of y value
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        cauchyLoss = .5*(self.cauchyLossRate**2)*torch.log(1+((Y_actual-Y_pred)/self.cauchyLossRate)**2)
        return torch.mean(cauchyLoss)    
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> 'CauchyRegression':
        """
        Fit the linear regression model to the training data.
        
        Args:
            X: Input features of shape (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns the fitted model instance
        """
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


        # Training loop
        prev_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Forward pass: Compute predictions
            Y_pred = self.model(self.X_train)

            # Compute loss
            loss = self.criterion(Y_pred, self.y_train)

            # Backward pass: Compute gradients
            loss.backward()

            # Update weights and bias manually
            with torch.no_grad():  # Disable gradient tracking during updates
                self.W -= self.learning_rate * self.W.grad
                self.b -= self.learning_rate * self.b.grad

            # Zero gradients for the next iteration
            self.W.grad.zero_()
            self.b.grad.zero_()

            # Print progress every 100 epochs
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Loss: {loss.item():.4f}')
            
            # Store loss history
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            # Check for convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break
            
            prev_loss = current_loss

        # Compute residual sum of squares for confidence intervals
        with torch.no_grad():
            # y_pred = self.forward(self.X_train)
            residuals = self.y_train - Y_pred
            self.residual_sum_squares = float(torch.sum(residuals ** 2))
        
        self.fitted = True

        # Evaluate on test data
        with torch.no_grad():  # Disable gradient computation
            y_test_pred = self.model(X_test)
            self.test_loss = self.criterion(y_test_pred, y_test)
            print(f'Test CauchyLoss: {self.test_loss.item():.4f}')

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples,)
            
        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()
    
    def get_parameters(self) -> Tuple[float, ...]:
        """
        Get the fitted parameters.
        
        Returns:
            Tuple of (w_1, w_0) - slope and intercept
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        
        return self.W.detach().numpy(), self.b.detach().numpy()
      
    def plot_regression_with_confidence_band(self, confidence_level: float = 0.95, 
                                           figsize: Tuple[int, int] = (10, 6),
                                           title: Optional[str] = None) -> plt.Figure:
        """
        Plot the fitted regression line with confidence band.
        
        Args:
            confidence_level: Confidence level for the band
            figsize: Figure size tuple
            title: Optional plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert training data to numpy for plotting
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        
        # Create prediction range
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)
        
        # Compute confidence band
        df = self.n_samples - 2
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mse = self.residual_sum_squares / df
        se_regression = np.sqrt(mse)
        
        # Standard error for predictions (confidence band)
        X_centered = X_range - self.X_mean
        se_pred = se_regression * np.sqrt(1/self.n_samples + X_centered**2 / (self.n_samples * self.X_var))
        
        # Confidence band bounds
        margin_of_error = t_critical * se_pred
        y_upper = y_pred_range + margin_of_error
        y_lower = y_pred_range - margin_of_error
        
        # Plot data points
        ax.scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')
        
        # Plot regression line
        ax.plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
        
        # Plot confidence band
        ax.fill_between(X_range, y_lower, y_upper, alpha=0.3, color='red', 
                       label=f'{int(confidence_level*100)}% Confidence band')
        
        # Get parameter values for display
        w_1_val, w_0_val = self.get_parameters()
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        if title is None:
            title = f'Linear Regression: y = {w_1_val:.3f}x + {w_0_val:.3f}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> dict:
        """
        Provide a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating summary")
        
        w_1_val, w_0_val = self.get_parameters()
        
        # R-squared calculation
        y_mean = float(torch.mean(self.y_train))
        ss_tot = float(torch.sum((self.y_train - y_mean) ** 2))
        r_squared = 1 - (self.residual_sum_squares / ss_tot)
        
        # Adjusted R-squared
        adj_r_squared = 1 - ((1 - r_squared) * (self.n_samples - 1) / (self.n_samples - 2))
        
        # RMSE
        rmse = np.sqrt(self.residual_sum_squares / self.n_samples)
        
        return {
            'parameters': {
                'w_1 (slope)': w_1_val,
                'w_0 (intercept)': w_0_val
            },
            'model_fit': {
                'r_squared': r_squared,
                'adjusted_r_squared': adj_r_squared,
                'rmse': rmse,
                'residual_sum_squares': self.residual_sum_squares
            },
            'training_info': {
                'n_samples': self.n_samples,
                'epochs_trained': len(self.loss_history),
                'final_loss': self.loss_history[-1] if self.loss_history else None
            }
        }

    def analysis_plot(self, figsize: Tuple[int, int] = (10, 6),
                            title: Optional[str] = None) -> plt.Figure:
        """
        Plot the fitted regression line.
        
        Args:
            figsize: Figure size tuple
            title: Optional plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(3,1,figsize=figsize)
        
        # Convert training data to numpy for plotting
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        
        # Create prediction range
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)
        
        # Plot data points
        ax[0].scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')
        
        # Plot regression line with original data
        ax[1].plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
        ax[1].scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')

        
        # Get parameter values for display
        w_1_val, w_0_val = self.get_parameters()

        # Plot loss as training occured
        ax[2].plot(self.loss_history, label='Training Loss', color='green')
        
        # Labels and title
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('y')
        ax[0].set_title("Original Data")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel('X')
        ax[1].set_ylabel('y')
        if title is None:
            title = f'Linear Regression: y = {w_1_val:.3f}x + {w_0_val:.3f}'
        ax[1].set_title(title)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Loss')
        ax[2].set_title('Loss Though Training')
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig

    def scatterplotMatrix(self, data):
        sns.pairplot(data)
        plt.show()

    def residualPlot(self, X, y):
        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            y_pred = self.model(X)
            residual = y_pred - y
        # print(self.X_train)
        plt.figure(figsize=(10,6))
        plt.scatter(X[:,0], residual, label='$X_{1}$')
        plt.scatter(X[:,1], residual, label='$X_{2}$')
        plt.scatter(X[:,2], residual, label='$X_{3}$')
        plt.scatter(X[:,3], residual, label='$X_{4}$')
        plt.legend()
        plt.xlabel('Predictor X', fontsize=14)
        plt.ylabel(r'$\^y-y$', fontsize=14)
        plt.show()

    def finalLoss(self):
        return self.test_loss, self.loss_history[-1]

class CauchyModel(GenericLikelihoodModel):
    def __init__(self, endog, exog, c: float=1.0, **kwds):
        self.c = c
        super(CauchyModel, self).__init__(endog, exog, **kwds)

    def loglike(self, params):
        exog = self.exog #matrix of regressors
        endog = self.endog #dependent variable
        yhat = exog @ params
        residual = yhat-endog

        return np.sum(stats.cauchy.logpdf(residual, scale=self.c))


#--------------------------------------------------------------------------------------------------------
# # multiple linear regression w/ statsmodel
# import polars as p
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split

# data = p.read_csv('/home/radioactiveshrimp/datasets/Folds5x2_pp.csv')
# X = data.select(['AT','V', 'AP', 'RH']).to_numpy().astype(np.float32)
# y = data['PE'].to_numpy().astype(np.float32)
# X = sm.add_constant(X, prepend=True)

# model = CauchyModel(y, X, c=1.0).fit()
# print(model.summary())

#---------------------------------------------------------------------------------------------------------
# multiple regression w/ pytorch
import polars as p
import pandas
from sklearn.model_selection import train_test_split

data = p.read_csv('/home/radioactiveshrimp/datasets/Folds5x2_pp.csv')
X = data.select(['AT','V', 'AP', 'RH']).to_numpy().astype(np.float32)
y = data['PE'].to_numpy().astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

input_dim = X_train.shape[1]  # Number of features

Reg = CauchyRegression(input_dim=input_dim)
print(input_dim)
fittedMod = Reg.fit(X_train, y_train, X_test, y_test)
print(Reg.fitted)
print(Reg.get_parameters())
Reg.scatterplotMatrix(pandas.read_csv('/home/radioactiveshrimp/datasets/Folds5x2_pp.csv'))
Reg.residualPlot(X,y)
print(Reg.finalLoss())

#-------------------------------------------------------------------------------------
# simple linear regression:
# import polars as p
# from sklearn.model_selection import train_test_split

# L = LinearRegression()
# data = p.read_csv('/home/radioactiveshrimp/radioactiveshrimp/Hydropower.csv')
# x = data.get_column('BCR')
# y = data.get_column('AnnualProduction')

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# y_pred = L.forward(.9)


# fittedMod = L.fit(X_train, y_train, X_test, y_test)
# # fittedMod = L.fit(X_train, y_train)
# print(L.predict(.9))
# # print(L.loss_history)

# fig = L.analysis_plot()
# plt.show()

#----------------------------------------------------------------
# np.random.seed(42)
# torch.manual_seed(42)

# n_samples = 100
# X = np.random.uniform(-5, 5, n_samples)
# true_slope = 2.5
# true_intercept = -1.2
# noise = np.random.normal(0, 1, n_samples)
# y = true_slope * X + true_intercept + noise

# print("=== Linear Regression with PyTorch ===")
# print(f"True parameters: slope={true_slope}, intercept={true_intercept}")
# print()

# # Create and fit model
# model = LinearRegression(learning_rate=0.01, max_epochs=10)
# model.fit(X, y)

# # Get fitted parameters
# fitted_slope, fitted_intercept = model.get_parameters()
# print(f"Fitted parameters: slope={fitted_slope:.4f}, intercept={fitted_intercept:.4f}")

# # Model summary
# summary = model.summary()
# print(f"\nModel Summary:")
# print(f"R-squared: {summary['model_fit']['r_squared']:.4f}")
# print(f"RMSE: {summary['model_fit']['rmse']:.4f}")
# print(f"Training epochs: {summary['training_info']['epochs_trained']}")

# # Confidence intervals
# ci_results = model.parameter_confidence_intervals(confidence_level=0.95)
# print(f"\n95% Confidence Intervals:")
# print(f"w_1 (slope): [{ci_results['w_1_confidence_interval'][0]:.4f}, {ci_results['w_1_confidence_interval'][1]:.4f}]")
# print(f"w_0 (intercept): [{ci_results['w_0_confidence_interval'][0]:.4f}, {ci_results['w_0_confidence_interval'][1]:.4f}]")

# # Plot results
# fig = model.plot_regression_with_confidence_band(confidence_level=0.95)
# plt.show()

# # Test different confidence levels
# for conf_level in [0.90, 0.95, 0.99]:
#     ci = model.parameter_confidence_intervals(confidence_level=conf_level)
#     print(f"\n{int(conf_level*100)}% CI for slope: [{ci['w_1_confidence_interval'][0]:.4f}, {ci['w_1_confidence_interval'][1]:.4f}]")