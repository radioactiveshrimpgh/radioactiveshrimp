"""
Animation subpackage containing modules for weight animations.
"""
from .weight_animation import WeightMatrixAnime, generate_weight_history, animate_weight_heatmap
from .largewt_animation import LargeWeightMatrixAnime,animate_large_heatmap
__all__=['WeightMatrixAnime,generate_weight_history, animate_weight_heatmap, LargeWeightMatrixAnime, animate_large_heatmap']