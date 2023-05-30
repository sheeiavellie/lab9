
# import packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
  
# reading the csv file
#data = pd.read_csv('headbrain3.csv')

weights = np.array([1,2,3,4,5,6,7,8,9,10,11 ])
heights = np.array([124.9,127.1,134.0,139.1,147.3,155.0,159.8,165.4,172.5,177.4,182.1])

data = pd.DataFrame({'heights': heights, 'weights': list(weights)}, columns=['heights', 'weights'])

# fit simple linear regression model
linear_model = ols('heights ~ weights', data=data).fit()
  
# display model summary
print(linear_model.summary())
  
# modify figure size
fig = plt.figure(figsize=(14, 8))
  
# creating regression plots
fig = sm.graphics.plot_regress_exog(linear_model, 'weights', fig=fig)

plt.show()