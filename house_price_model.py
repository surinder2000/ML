import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

dataset = pd.read_csv('USA_Housing.csv')

X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

y=dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

error = np.abs(y_test - y_pred)

mape = np.mean(error/y_test) * 100

accuracy = 100 - mape

joblib.dump(model,'Housepricepred.pk1')


