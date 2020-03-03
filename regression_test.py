import numpy as np
import matplotlib.pyplot as plt
from regression_predictor import RegressionPredictor
from calculations import mean_square_error

if __name__ == "__main__":
    # Read in synthetic data for regressions
    synth1 = np.genfromtxt("./data/synthetic-1.csv", delimiter=",")
    synth2 = np.genfromtxt("./data/synthetic-2.csv", delimiter=",")
    synth3 = np.genfromtxt("./data/synthetic-3.csv", delimiter=",")

    predictor = RegressionPredictor(1*10**-6)

    predictor.fit(synth1, 2, error_bound=0.001, timeout=120)
    results = list(map(predictor.predict, synth1[:, 0]))
    results = np.array(results)

    print(predictor.predict_func())
    print(mean_square_error(synth1[:, 1], results))
