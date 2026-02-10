from radioactiveshrimp import deepl as d
from radioactiveshrimp import animation
import time
# import matplotlib.pyplot as plt
from datetime import datetime
import os
from manim import config

#demonstrate use of binary_classification (deepl)
print("starting test script....")

b = d.binary_classification(40000, 200, 5000, 0.01)
#epochs 5000
#eta 0.01
#features = 200
#samples 40000

b.checkGPU()
time.sleep(5)
b.generateMatrix()
b.generateLabels()
b.initializeWeights()
# print(b.fit())
_1,_2,_3,_4,lh,w1,w2,w3,w4=b.fit()
print(w1)
print(w1.shape)
# lossPlot = b.plotLoss()

dt = datetime.now()
date = dt.strftime('%Y%m%d%H%M%S')
if not os.path.exists('media/'):
    os.mkdir('media')

config.media_dir = "/home/mes0063/cpe587/radioactiveshrimp/scripts/media"
fileName1 = 'W1_'+date+'.mp4'
fileName2 = 'W2_'+date+'.mp4'
fileName3 = 'W3_'+date+'.mp4'
fileName4 = 'W4_'+date+'.mp4'
print(fileName1)

# WeightMatrixAnime(matrix_stack=w1,dt=0.04, title_str="TESTING_RAH")
# matx1= animation.LargeWeightMatrixAnime(matrix_stack=w1,dt=0.04, title_str=fileName1)
# matx1.construct()
animation.animate_large_heatmap(
        matrix_stack=w1, 
        dt=0.04,
        file_name=fileName1,
        title_str="Weight1 Evolution"
    )
animation.animate_large_heatmap(
        matrix_stack=w2, 
        dt=0.04,
        file_name=fileName2,
        title_str="Weight2 Evolution"
    )
animation.animate_large_heatmap(
        matrix_stack=w3, 
        dt=0.04,
        file_name=fileName3,
        title_str="Weight3 Evolution"
    )
animation.animate_large_heatmap(
        matrix_stack=w4, 
        dt=0.04,
        file_name=fileName4,
        title_str="Weight4 Evolution"
    )
# matx2= animation.LargeWeightMatrixAnime(matrix_stack=w2,dt=0.04, title_str=fileName2)
# matx2.construct()
# matx3= animation.LargeWeightMatrixAnime(matrix_stack=w3,dt=0.04, title_str=fileName3)
# matx3.construct()
# matx4= animation.LargeWeightMatrixAnime(matrix_stack=w4,dt=0.04, title_str=fileName4)
# matx4.construct()