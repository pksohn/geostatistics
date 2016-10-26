import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import simplekml


def sq_exponential(x, y, l):
    d = spatial.distance_matrix(x, y)
    k = np.exp(-(d ** 2) / (2 * l * l))
    return k


def exponential(x, y, l):
    d = spatial.distance_matrix(x, y)
    k = np.exp(-d/l)
    return k


def make_grid(bounding_box, ncell):
    xmax, xmin, ymax, ymin = bounding_box
    xgrid = np.linspace(xmin, xmax, ncell)
    ygrid = np.linspace(ymin, ymax, ncell)
    mX, mY = np.meshgrid(xgrid, ygrid)
    ngridX = mX.reshape(ncell*ncell, 1);
    ngridY = mY.reshape(ncell*ncell, 1);
    return np.concatenate((ngridX, ngridY), axis=1)


def cross_validate(train, l_values, sigma_values, rmse_opt, k_folds, cov_funcs=False, verbose=False):
    if not isinstance(l_values, (list, tuple, np.ndarray, np.array)):
        l_values = list(l_values)

    if not isinstance(sigma_values, (list, tuple, np.ndarray, np.array)):
        sigma_values = list(sigma_values)

    functions = [sq_exponential]
    if cov_funcs:
        functions.append(exponential)

    for f in functions:
        for l in l_values:
            for sigma in sigma_values:
                for k in range(k_folds):

                    folds = np.array_split(train, k_folds)
                    testing = folds.pop(k)
                    training = np.concatenate(folds)

                    krig = SimpleKriging(training_data=training)
                    pred = krig.predict(test_data=testing[:, :2], l=l, sigma=sigma)
                    rmse = (((pred - testing[:, -1:])**2)**.5).mean()

                    if k == 0:
                        local_error = rmse
                    else:
                        if rmse < local_error:
                            local_error = rmse

                    if rmse < rmse_opt:
                        rmse_opt = rmse
                        l_final = l
                        sigma_final = sigma
                        cov_func_final = f

                if verbose:
                    print "l={}, sigma={}, rmse={}".format(l, sigma, local_error)

    return l_final, sigma_final, cov_func_final, rmse_opt


class SimpleKriging(object):

    def __init__(self, training_data):
        self.training_data = training_data
        self.X = training_data[:, :-1]
        self.Y = training_data[:, -1:]

    def predict(self, test_data, l, sigma, indices=False, cov_function=sq_exponential):

        K_xtest_x = cov_function(test_data, self.X, l)
        K = cov_function(self.X, self.X, l)

        sigma_sq_I = sigma**2 * np.eye(len(self.X))
        inv = np.linalg.inv(K + sigma_sq_I)

        predictions = K_xtest_x.dot(inv).dot(self.Y)

        if indices:
            return np.concatenate((test_data, predictions), axis=1)
        else:
            return predictions

    def simulate(self, bbox, ncells, l, sigma, gamma=0.001, indices=False, cov_function=sq_exponential,
                 show_visual=False, save_kml=None):

        grid = make_grid(bounding_box=bbox, ncell=ncells)

        prediction = self.predict(test_data=grid, l=l, sigma=sigma, indices=True)

        K = cov_function(grid, grid, l)
        L = np.linalg.cholesky(K + gamma * np.eye(len(K)))
        u = np.random.normal(size=len(L))

        result = prediction[:, -1] + L.dot(u)

        if show_visual:
            y = grid[:, 0]
            x = grid[:, 1]
            plt.scatter(x, y, c=result)
            plt.colorbar(ticks=[np.min(result), np.max(result)], label='Rainfall in mm')
            plt.show()

        if save_kml:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            y = grid[:, 0]
            x = grid[:, 1]
            ax.scatter(x, y, c=result, s=50, edgecolors='', marker='s', alpha=0.5)
            ax.axis([bbox[2], bbox[3], bbox[0], bbox[1]])
            ax.axis('off')
            plt.savefig('{}.png'.format(save_kml))

            kml = simplekml.Kml()
            ground = kml.newgroundoverlay(name='GroundOverlay')
            ground.icon.href = '{}.png'.format(save_kml)
            ground.latlonbox.north = bbox[1]
            ground.latlonbox.south = bbox[0]
            ground.latlonbox.east = bbox[3]
            ground.latlonbox.west = bbox[2]
            kml.save('{}.kml'.format(save_kml))

        if indices:
            return np.concatenate((grid, result[:, None]), axis=1)
        else:
            return result

