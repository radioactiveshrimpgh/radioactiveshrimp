"""
Deep learning subpackage containing binary classification function.
"""
from .two_layer_binary_classification import binary_classification
from .multiclass import SimpleNN, ClassTrainer, ConvLayer, ImageNetCNN, CNNTrainer
__all__=['binary_classification, SimpleNN, ClassTrainer, ConvLayer, ImageNetCNN, CNNTrainer']