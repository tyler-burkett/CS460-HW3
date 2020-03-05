import numpy as np
import matplotlib.pyplot as plt
from regression_predictor import RegressionPredictor
from calculations import mean_square_error

if __name__ == "__main__":
    # Read in synthetic data for regressions
    synth_data = []
    for i in range(1, 4):
        synth_data.append(np.genfromtxt("./data/synthetic-{}.csv".format(i), delimiter=","))

    fig = plt.subplot(2,2,1)

    data_index = 1
    for data in synth_data:
        plt.subplot(2, 2, data_index)
        plt.title("synthetic-{} Data and Regression Lines".format(data_index))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(data[:, 0], data[:, 1], 'go')
        for order in [1, 2, 4, 7]:
            # Train regression predictor and measure error
            predictor = RegressionPredictor(1*10**-3, 1000)
            predictor.fit(data, order, min_iterations=1000, error_bound=0.0001, timeout=3*60)
            results = list(map(predictor.predict, data[:, 0]))
            results = np.array(results)
            print("synthetic-{} data, order={}:".format(data_index, order))
            print(predictor.predict_func())
            print(mean_square_error(data[:, 1], results))
            print("")

            # Plot points and line
            line_points = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)
            line = list(map(predictor.predict, line_points))
            line = np.array(line)
            plt.plot(line_points, line, '-', label="{}".format(order))
        data_index = data_index + 1
        plt.legend(title="Order")
    plt.show()
