import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def create_data():
    window_size = 20
    filepath = "BND.csv"
    data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    data['returns'] = np.log(data['Adj Close']).diff()
    data['squared_returns'] = data['returns'] ** 2
    cols = []
    for i in range(1, window_size + 1):
        col = f'lag_{i}'
        data[col] = data['squared_returns'].shift(i)
        cols.append(col)
    data.dropna(inplace=True)
    X = np.array(data[cols])
    y = np.array(data['squared_returns'])
    return X, y


def create_polynomial_model(degree):
    model = LinearRegression()
    return f"Linear Regression (Degree {degree})", model


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return model, mse


def print_coeffs(label, model):
    print(label, model.coef_, model.intercept_)


def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree + 1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
            print_coeffs("Coefficients: ", model)
            mse_sum += mse
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        print(f"for degree: {degree}, MSE: {avg_mse}")
        model.fit(X, y)
        print_coeffs("Final Coefficients: ", model)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)

    # Plot the evolution of model performance
    degrees = list(results.keys())
    mses = list(results.values())
    plt.plot(degrees, mses, marker='o')
    plt.xlabel('Degree of Polynomial Regression')
    plt.ylabel('Mean Squared Error')
    plt.title('Evolution of Model Performance')
    plt.xticks(degrees)
    plt.grid(True)
    plt.show()

    return best_model


# Main code
X, y = create_data()
best_model = cross_validate(X, y, n_splits=5, from_degree=1, to_degree=20)
