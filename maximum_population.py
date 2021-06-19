import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm

df = pd.read_csv('WorldPopulation.csv')
df = df.sort_values('Year')
X = np.array(df['Year'])
y = np.array(df['Population']) / 1000000000

model = pm.auto_arima(y, seasonal=True, m=12)
preds = np.ravel(model.predict(n_periods=100))[:80]
years = [i for i in range(2020, 2101)]

sns.set_theme()
plt.plot(X, y, color='blue')
plt.plot(years, preds, color='red')
plt.xlabel('Year')
plt.ylabel('Population(billion)')
plt.show()
