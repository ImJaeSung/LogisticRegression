import numpy as np
import matplotlib.pyplot as plt
from model import LogisticRegression
from data_utils import x, y

model = LogisticRegression(lr = 0.001, epochs = 50000)
model.fit(x, y)

#%%
plt.plot(model.losses)
plt.text(np.argmin(model.losses), np.min(model.losses),
    'Min epoch {}'.format(np.round(np.min(model.losses), 4)),
    color='r',
    horizontalalignment='center',
    verticalalignment='top')