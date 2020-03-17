import random
import time
import numpy as np
import math
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from calculations import mean_square_error

class RegressionPredictor:
    def __init__(self, learning_rate, T):
        self.learning_rate = learning_rate
        self.T = T

    def fit(self, train_data, order, min_iterations=float("inf"), error_bound=0.01, timeout=60):
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
        self.expr = parse_expr("+".join("N_{}*x**{}".format(i, i) for i in range(order+1)))
        self.order = order

        # Normalize data
        #x = (train_data[:, 0] - train_data[:, 0].min()) / (train_data[:, 0].max() - train_data[:, 0].min())
        #y = (train_data[:, 1] - train_data[:, 1].min()) / (train_data[:, 1].max() - train_data[:, 1].min())
        x = train_data[:, 0]
        y = train_data[:, 1]

        old_learning_rate = self.learning_rate
        start = time.time()
        old_thetas = np.array(self.thetas) * 2
        iterations = 0
        self.lambdify_expression()
        while abs(max(self.thetas - old_thetas)) > error_bound or iterations < min_iterations:
            old_thetas = np.array(self.thetas)
            results = self.numeric_func(x)
            self.update_weights(results, y, x)
            iterations = iterations + 1
            self.learning_rate = old_learning_rate / (1 + iterations/self.T)
            self.lambdify_expression()
            if timeout is not None and time.time() - start > timeout \
                    and (min_iterations == float("inf") or iterations > min_iterations):
                break
        self.learning_rate = old_learning_rate

    def update_weights(self, predictions, actuals, x):
        same_size = len(predictions) == len(actuals) and len(predictions) == len(x)
        assert(same_size)
        m = len(predictions)
        for i in range(self.order+1):
            change = self.learning_rate * (1 / m) * sum((predictions - actuals) * x**i)
            self.thetas[i] = self.thetas[i] - change

    def evaluate_expr(self, x):
        return self.numeric_func(x)

    def predict(self, x):
        return self.evaluate_expr(x)

    def predict_func(self):
        substitutions = [("N_{}".format(i), self.thetas[i]) for i in range(self.order+1)]
        substitutions = dict(substitutions)
        return self.expr.subs(substitutions)

    def lambdify_expression(self):
        substitutions = [("N_{}".format(i), self.thetas[i]) for i in range(self.order+1)]
        substitutions = dict(substitutions)
        x = symbols("x")
        self.numeric_func = lambdify(x, self.expr.subs(substitutions), "numpy")


class InterpolatingRegressionPredictor(RegressionPredictor):
    def fit(self, train_data, order, min_iterations=float("inf"), error_bound=0.01, timeout=60):
        # build symbolic expression to evaluate polynomial expression
        self.expr = parse_expr("+".join("N_{}*x**{}".format(i, i) for i in range(order+1)))
        self.order = order

        # Create inital paramters for polyonomial of provided order
        # by interpolating points from dataset
        self.thetas = self.interpolated_weights(train_data)

        x = train_data[:, 0]
        y = train_data[:, 1]

        old_learning_rate = self.learning_rate
        start = time.time()
        old_thetas = np.array(self.thetas) * 2
        iterations = 0
        self.lambdify_expression()
        while abs(max(self.thetas - old_thetas)) > error_bound or iterations < min_iterations:
            old_thetas = np.array(self.thetas)
            results = self.numeric_func(x)
            self.update_weights(results, y, x)
            iterations = iterations + 1
            self.learning_rate = old_learning_rate / (1 + iterations/self.T)
            self.lambdify_expression()
            if timeout is not None and time.time() - start > timeout \
                    and (min_iterations == float("inf") or iterations > min_iterations):
                break
        self.learning_rate = old_learning_rate

    def interpolated_weights(self, data):
        # To get unique interpolating polynomial of order k,
        # need to select k+1 points
        num_points = self.order + 1

        # Sort data by x value and pick enough points
        # to uniquely interpolate data, picking evenly spaced points
        data = np.sort(data, axis=0)
        selected_idx = np.round(np.linspace(0, len(data)-1, num_points)).astype(int)
        selected_points = data[selected_idx]

        # Cacluate weights of interpolating polynomial using in-place divided difference
        thetas = list(selected_points[i][1] for i in range(len(selected_points)))
        for i in range(0, len(selected_points)-1):
            for j in range(len(selected_points)-1, i, -1):
                thetas[j] = (thetas[j] - thetas[j-1]) / (selected_points[j][0] - selected_points[j-i-1][0])

        return np.array(thetas)
