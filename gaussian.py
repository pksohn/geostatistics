import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def covariance(x, y, l):
    d = spatial.distance_matrix(x, y)
    k = np.exp(-(d ** 2) / (2 * l * l))
    return k


def make_grid(bounding_box, ncell):
    xmax, xmin, ymax, ymin = bounding_box
    xgrid = np.linspace(xmin, xmax, ncell)
    ygrid = np.linspace(ymin, ymax, ncell)
    mX, mY = np.meshgrid(xgrid, ygrid)
    ngridX = mX.reshape(ncell*ncell, 1);
    ngridY = mY.reshape(ncell*ncell, 1);
    return np.concatenate((ngridX, ngridY), axis=1)


class SimpleKriging(object):

    def __init__(self, training_data):
        self.training_data = training_data
        self.X = training_data[:, :-1]
        self.Y = training_data[:, -1:]

    def predict(self, test_data, l, sigma, indices=False):

        K_xtest_x = covariance(test_data, self.X, l)
        K = covariance(self.X, self.X, l)

        sigma_sq_I = sigma**2 * np.eye(len(self.X))
        inv = np.linalg.inv(K + sigma_sq_I)

        predictions = K_xtest_x.dot(inv).dot(self.Y)

        if indices:
            return np.concatenate((test_data, predictions), axis=1)
        else:
            return predictions

    def simulate(self, bbox, ncells, l, sigma, gamma=0.001, indices=False, show_visual=False, save_visual=None):

        grid = make_grid(bounding_box=bbox, ncell=ncells)

        prediction = self.predict(test_data=grid, l=l, sigma=sigma, indices=True)

        K = covariance(grid, grid, l)
        L = np.linalg.cholesky(K + gamma * np.eye(len(K)))
        u = np.random.normal(size=len(L))

        result = prediction[:, -1] + L.dot(u)

        if show_visual or save_visual:
            x = grid[:, 0]
            y = grid[:, 1]
            plt.scatter(x , y, c=result)
            plt.colorbar(ticks=[np.min(result), np.max(result)], label='Rainfall in mm')
            if show_visual:
                plt.show()
            if save_visual:
                plt.savefig(save_visual)

        if indices:
            return np.concatenate((grid, result[:, None]), axis=1)
        else:
            return result
