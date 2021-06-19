import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm

df = pd.read_csv('WorldPopulation.csv')
df = df.sort_values('Year')
y_population = np.array(df['Population']) / 1000000000
y_change = np.array(df['ChangePerc'])

model = pm.auto_arima(y_change, seasonal=True, m=12)
preds = np.ravel(model.predict(n_periods=100))[:80]

X_test = [i for i in range(2020, 2101)]
population_pred = [y_population[-1]]
for i in range(len(preds)):
  new_population = population_pred[i] + (population_pred[i] * (preds[i] / 100))
  population_pred.append(new_population)

df = pd.DataFrame({'year':X_test, 'population':population_pred})
df.to_csv('Population_Pred.csv', index=False)

sns.set_theme()
plt.plot(X, y_population, color='blue')
plt.plot(X_test, population_pred, color='red')
plt.xlabel('Year')
plt.ylabel('Population(billion)')
plt.show()
