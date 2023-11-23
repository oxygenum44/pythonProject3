stress1 = [29.0801910508518,
           24.8140822695698,
           30.0019419808698,
           28.1413123400374,
           28.5993479517619,
           27.7622179059724,
           27.9227758824528,
           24.3538385531122]

#stress1 = [stress + 4 for stress in stress1]

numbers1 = [24352,
            523208,
            83716,
            25256,
            123174,
            57463,
            316735,
            401972]

numbers1 = [math.log10]

stress2 = [37.936341824767,
           32.5274958066904,
           35.4810886810467,
           26.1083224369141,
           34.1865550326855,
           32.8593523393368,
           33.4067469271145
           ]

numbers2 = [158720,
            483282,
            98131,
            640857,
            8210,
            53128,
            90315]

stress3 = [31.4482996989313,
           31.8807178972596,
           32.3802643505282,
           33.1850931110044,
           33.3813870598882,
           34.2896139449241,
           34.9375843824991
           ]

numbers3 = [333330,
           600465,
           168787,
           233455,
           99020,
           63407,
           140662]

stress4 = [1,2,3,4]
numbers4 = [2,3.9,6.15,8.1]
stress5 = [1.2,2.1,2.9,3.8]
numbers5 = [2,4,6,8]

import matplotlib.pyplot as plt

plt.scatter(numbers1, stress1, color='r')
plt.scatter(numbers2, stress2, color='b')
plt.scatter(numbers3, stress3, color='g')
plt.ylim(bottom=0)
plt.xscale('log')
plt.show()


import numpy as np
import pandas as pd
from scipy.stats import levene
from statsmodels.tools import add_constant
from statsmodels.formula.api import ols  ## use formula api to make the tests easier

np.inf == float('inf')

df1 = pd.DataFrame(
    {'x': numbers1,
     'y': stress1,
     })

df2 = pd.DataFrame(
    {'x': numbers3,
     'y': stress3,
     })

df1 = add_constant(df1)
df2 = add_constant(df2)

print(df1)
print(df2)

formula1 = 'y ~ x + const'  ## define formulae
formula2 = 'y ~ x + const'

model1 = ols(formula1, df1).fit()
model2 = ols(formula2, df2).fit()

print(levene(model1.resid, model2.resid))

df1['c'] = 1  ##  add indicator variable that tags the first groups of points

df_all = df1.append(df2, ignore_index=True).drop('const', axis=1)
# df_all = df_all.rename(columns={'index': 'x', 0: 'y'})  ## the old index will now be called x and the old values are now y
df_all = df_all.fillna(0)  ## a bunch of the values are missing in the indicator columns after stacking
df_all['int'] = df_all['x'] * df_all['c']  # construct the interaction column

print(df_all)  ## look a the data

formula = 'y ~ x + c + int'  ## define the linear model using the formula api
result = ols(formula, df_all).fit()
hypotheses = '(c = 0), (int = 0)'

f_test = result.f_test(hypotheses)
print(f_test)
