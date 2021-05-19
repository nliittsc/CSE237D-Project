# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:11:32 2021

@author: 18315
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

y = np.random.choice(np.arange(80, 140), size=1000)
x = np.random.choice(np.arange(50, 60), size=1000).reshape(-1, 1)
xticks = np.arange(50, 60, 1)
yticks = np.arange(80, 140, 10)

phrenology_predictor = LinearRegression()
phrenology_predictor.fit(x, y)
pred_y = phrenology_predictor.predict(x)

plt.title("Bullshit Regression")
plt.ylabel("Fake IQ")
plt.xlabel("Fake Head Circumference")
plt.xticks(xticks)
plt.yticks(yticks)
plt.grid()
plt.scatter(x, y, marker="D")
plt.plot(x, pred_y, color="darkorange")