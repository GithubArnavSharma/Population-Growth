import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm

df = pd.read_csv('WorldPopulation.csv')
df = df.sort_values('Year')
X = np.array(df['Year'])
y_change = np.array(df['ChangePerc'])

model = pm.auto_arima(y_change, seasonal=True, m=12)
preds = np.ravel(model.predict(n_periods=80))

X_test = [i for i in range(2020, 2101)]
preds = [y_change[-1]] + list(preds)

sns.set_theme()
plt.plot(X, y_change, color='blue')
plt.plot(X_test, preds, color='red')
plt.xlabel('Year')
plt.ylabel('Population Growth %')
plt.show()
