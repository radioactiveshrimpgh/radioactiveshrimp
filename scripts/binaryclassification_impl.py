from radioactiveshrimp import deepl as d
import time
import matplotlib.pyplot as plt
from datetime import datetime
#demonstrate use of binary_classification (deepl)
print("starting test script....")

b = d.binary_classification(100,48)
b.checkGPU()
time.sleep(5)
b.generateMatrix()
b.generateLabels()
b.initializeWeights()
# print(b.fit())
_1,_2,_3,_4,lh,w1,w2,w3,w4=b.fit()
print(w1)
print(w1.shape)
lossPlot = b.plotLoss()

dt = datetime.now()
date = dt.strftime('%Y%m%d%H%M%S')
fileName = 'crossentropyloss_'+date+'.pdf'
print(fileName)
plt.savefig(fileName)

lossPlot.show()