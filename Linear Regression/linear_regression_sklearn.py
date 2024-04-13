"""
This script loads Size(sqft),Bedrooms,Floors,Age,Price(usd) for 22 houses
from a text file, normalizes it using Z-score normalization,
fits a Linear Regression model, and plots the predictions against the original features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
def load_house_data():
    """
        Function to load the house data from a text file.

        Returns:
        X : numpy array
            Features of the houses.
        y : numpy array
            Prices of the houses.
    """
    data = np.loadtxt("house_data.txt", delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    return X, y

# Load the data set
X_train, y_train = load_house_data()
X_features = ['Size (sqft)', 'Bedrooms', 'Floors', 'Age']

# Scale/normalize the training data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Print the peak-to-peak range of the data before and after normalization
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}\n")

# Create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}\n")

# View parameters
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}\n")

# Make predictions

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)

# make a prediction using w,b
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# Plot Results

# plot predictions and targets vs original features
fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred, c='r', label='predict')

ax[0].set_ylabel("Price (usd)"); ax[0].legend()
fig.suptitle("Target vs prediction using Z-score normalized model")
plt.show()
