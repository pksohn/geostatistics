import numpy as np
from scipy import spatial


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

    def predict(self, test_data, l, indices=False):

        K_xtest_x = covariance(test_data, self.X, l)
        K = covariance(self.X, self.X, l)

        sigma_sq_I = np.var(self.Y) * np.eye(len(self.X))
        inv = np.linalg.inv(K + sigma_sq_I)

        predictions = K_xtest_x.dot(inv).dot(self.Y)

        if indices:
            return np.concatenate([test_data, predictions], axis=1)
        else:
            return predictions


    def simulate(self, bbox, ncells, l, gamma=0.001, indices=False):

        grid = make_grid(bounding_box=bbox, ncell=ncells)

        prediction = self.predict(test_data=grid, l=l, indices=True)

        K = covariance(x=self.X, y=self.X, l=l)
        L = np.linalg.cholesky(K + gamma * np.eye(K.shape[0]))

        result = prediction + L

        if indices:
            return np.concatenate([grid, result], axis=1)
        else:
            return result

