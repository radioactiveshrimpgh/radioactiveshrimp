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
b.fit()
lossPlot = b.plotLoss()

dt = datetime.now()
date = dt.strftime('%Y%m%d%H%M%S')
fileName = 'crossentropyloss_'+date+'.pdf'
print(fileName)
plt.savefig(fileName)

lossPlot.show()