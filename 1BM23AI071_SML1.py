#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
data = {
    'shear': [1.2, 2.5, 3.1, 4.0, 5.2, 6.3, 7.4, 8.1, 9.0, 10.2, 11.1, 12.4, 13.3, 14.5, 15.0, 16.2, 17.5, 18.1, 19.3, 20.0],
    'age': [21, 22, 24, 25, 28, 30, 31, 34, 35, 37, 40, 42, 43, 45, 48, 50, 51, 53, 55, 57]
}
df = pd.DataFrame(data)
X = sm.add_constant(df['shear'])
y = df['age']
model = sm.OLS(y, X).fit()
intercept = model.params['const']
slope = model.params['shear']
print(f"Intercept (β₀): {intercept:.4f}")
print(f"Slope (β₁): {slope:.4f}")
print(f"Equation of best fit line: age = {intercept:.4f} + {slope:.4f} * shear")
y_pred = intercept + slope * df['shear']
plt.figure(figsize=(8, 5))
plt.scatter(df['shear'], df['age'], color='blue', label='Data Points')
plt.plot(df['shear'], y_pred, color='red', label='Best Fit Line')
plt.xlabel('Shear')
plt.ylabel('Age')
plt.title('Linear Regression of Age on Shear')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




