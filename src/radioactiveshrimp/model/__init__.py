"""
Model subpackage containing model functions.
"""
from .regression import LinearRegression, CauchyModel, CauchyRegression
__all__=['LinearRegression,CauchyModel,CauchyRegression']

from .logit import LogisticRegression
__all__=['LogisticRegression']

from .neuralnet import TorchNet
__all__=['TorchNet']
