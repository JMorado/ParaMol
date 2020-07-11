from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------ #
#                                                              #
#                      BAYESIAN OPTIMIZER                      #
#                                                              #
# ------------------------------------------------------------ #
class Bayesian:
    def __init__(self, options=None):
        self._objective_function = None

        if options is None:
            # Define all default options for SciPy optimiser
            pass
        else:
            self._options = options

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def noisy_objective_function(self, parameters, noise=0.1):
        return self._objective_function(parameters) #+ noise * np.random.randn()

    def propose_location(self, acquisition, x_sample, y_sample, gpr, bounds, n_restarts=16):
        """
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            x_sample: Sample locations (n x d).
            y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        """
        dim = x_sample.shape[1]
        min_val = 1000
        min_x = None

        def min_obj(X):
            # Maximize
            return gpr.predict(X.reshape(-1,dim))#acquisition(X.reshape(-1, dim), x_sample, y_sample, gpr)

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, method="SLSQP")#bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(-1, 1)


    def expected_improvement(self, x, x_sample, y_sample, gpr, xi=1e-2):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gpr.predict(x, return_std=True)

        z = (mean - np.min(y_sample) - xi) / std
        return norm.cdf(z)

        """
        mu, sigma = gpr.predict(x, return_std=True)
        mu_sample = gpr.predict(x_sample)

        sigma = sigma.reshape(-1, 1)

        mu_sample_opt = np.max(y_sample)
        #mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
"""
    def run_optimization(self, f, parameters, constraints=None):
        self._objective_function = f
        # Gaussian process with Mat??rn kernel as surrogate model
        noise = 1.0
        # Define the GP
        kernel = Matern()
        gpr = GaussianProcessRegressor(kernel=kernel,
                                            alpha=1e-4,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

        # set bounds
        bounds = self._define_bounds(parameters)
        # Initialize samples
        x_sample, y_sample = self._perform_initial_sampling(200, bounds)

        n_iter = 500
        for i in range(n_iter):
            print("Iteration {}".format(i))
            print(np.argmin(y_sample))
            print(y_sample)

            # Update Gaussian process with existing samples
            gpr.fit(x_sample, y_sample)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            x_next = self.propose_location(self.expected_improvement, x_sample, y_sample, gpr, bounds)

            # Obtain next noisy sample from the objective function
            y_next = np.asarray(self.noisy_objective_function(x_next.reshape(1, -1).tolist()[0]))

            # Add sample to previous samples
            x_sample = np.vstack((x_sample, x_next.reshape(1, -1)))
            y_sample = np.concatenate((y_sample, y_next.reshape(1,)))

            print("MIN:", np.min(y_sample))
        arg_min = np.argmin(y_sample)
        parameters = x_sample[arg_min]
        return parameters

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def _perform_initial_sampling(self, n_samples, bounds):
        y_sample = np.zeros(n_samples)
        x_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))

        for i in range(x_sample.shape[0]):
            y_sample[i] = self.noisy_objective_function(x_sample[i,:])

        return x_sample, y_sample

    def _define_bounds(self, parameters):
        x = 0.05
        bounds = [[p*(1-x), p*(1+x)] for p in parameters]
        return np.asarray(bounds)


