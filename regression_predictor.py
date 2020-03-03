import random
import time
import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from calculations import mean_square_error

class RegressionPredictor:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def fit(self, train_data, order, error_bound=0.01, timeout=60):
        # Create random initial paramters for polyonomial of provided order
        # Random values range based on min/max of y values
        self.x_max = train_data[:, 0].max()
        self.x_min = train_data[:, 0].min()
        self.y_max = train_data[:, 1].max()
        self.y_min = train_data[:, 1].min()
        guess_max = self.y_max
        guess_min = self.y_min
        self.thetas = np.array([random.uniform(guess_max, guess_min) for i in range(order+1)])

        # build symbolic expression to evaluate polynomial expression
        self.expr = parse_expr("+".join("N_{}*x**{}".format(i,i) for i in range(order+1)))
        self.order = order

        # Normalize data
        x = (train_data[:, 0] - train_data[:, 0].min()) / (train_data[:, 0].max() - train_data[:, 0].min())
        y = (train_data[:, 1] - train_data[:, 1].min()) / (train_data[:, 1].max() - train_data[:, 1].min())

        old_learning_rate = self.learning_rate
        old_mse = 0
        new_mse = 0
        start = time.time()
        old_thetas = np.array(self.thetas) * 2
        while abs(max(self.thetas - old_thetas)) > error_bound:
            old_thetas = np.array(self.thetas)
            results = list(map(self.evaluate_expr_raw, x))
            results = np.array(results)
            self.update_weights(results, y, x)
            old_mse = new_mse
            new_mse = mean_square_error(y, results)
            if new_mse < old_mse:
                self.learning_rate = self.learning_rate * 2
            else:
                self.learning_rate = self.learning_rate / 2
            if timeout is not None and time.time() - start > timeout:
                break

    def update_weights(self, predictions, actuals, x):
        same_size = len(predictions) == len(actuals) and len(predictions) == len(x)
        assert(same_size)
        m = len(predictions)
        for i in range(self.order+1):
            change = self.learning_rate * (1 / m) * sum((predictions - actuals)**2 * x)
            self.thetas[i] = self.thetas[i] - change

    def evaluate_expr_raw(self, x):
        substitutions = [("N_{}".format(i), self.thetas[i]) for i in range(self.order+1)] + [("x", x)]
        substitutions = dict(substitutions)
        return self.expr.evalf(subs=substitutions)

    def evaluate_expr(self, x):
        x = self.normalize_input(x)
        return self.denormaize_output(self.evaluate_expr_raw(x))

    def normalize_input(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    def denormaize_output(self, y):
        return (self.y_max - self.y_min)*y + self.y_min

    def predict(self, x):
        return self.evaluate_expr(x)

    def predict_func(self):
        substitutions = [("N_{}".format(i), self.thetas[i]) for i in range(self.order+1)]
        substitutions = dict(substitutions)
        return self.expr.subs(substitutions)
