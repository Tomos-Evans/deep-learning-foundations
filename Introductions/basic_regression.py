import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read Data
dataFrame = pd.read_fwf('brain_body.txt')

xVals = dataFrame[['Brain']]
yVals = dataFrame[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(xVals, yVals)


plt.scatter(xVals, yVals)

plt.plot(xVals, body_reg.predict(xVals))

plt.show()
