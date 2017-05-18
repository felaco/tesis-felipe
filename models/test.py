import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

y_true = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1])
y_pred = np.array([0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1])

fpr, tpr, res = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.show()