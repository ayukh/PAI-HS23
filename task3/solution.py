"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math
import scipy
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, Sum, Product
from warnings import catch_warnings, simplefilter
import warnings
warnings.filterwarnings('ignore')
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        '''kernel_f = .5*Matern(length_scale=.5, nu=2.5)
        kernel_v = math.sqrt(2)*Matern(length_scale=.5, nu=1.5)
        self.f_map = GaussianProcessRegressor(kernel = kernel_f)
        self.v_map = GaussianProcessRegressor(kernel = kernel_v)'''

        ### define kernels ###

        kernel1 = Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5)
        kernel2 = ConstantKernel(constant_value=0.5, constant_value_bounds="fixed")
        kernel3 = WhiteKernel(noise_level=0.15)  # 0.4, 0.5
        self.kernel_f = Sum(Product(kernel1, kernel2), kernel3)
        self.f_map = GaussianProcessRegressor(kernel=self.kernel_f, alpha=1e-10,
                                              optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False,
                                              copy_X_train=True, random_state=None)

        vkernel1 = Matern(length_scale=0.5, length_scale_bounds=[1e-5, 1e5], nu=2.5)
        vkernel2 = ConstantKernel(constant_value=np.sqrt(2))
        vkernel3 = WhiteKernel(noise_level=0.0001)
        vkernel4 = ConstantKernel(constant_value=1.5)

        self.kernel_v = Sum(Sum(vkernel4, Product(vkernel1, vkernel2)), vkernel3)
        # Default GaussianProcessRegressor (check argument possibilities)
        self.v_map = GaussianProcessRegressor(kernel=self.kernel_v, alpha=1e-10,
                                              optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False,
                                              copy_X_train=True, random_state=None)

        ######################

        self.past_points = np.empty([0, 3])  # np.zeros([1, 3])
        self.gpucb_kappa = 1.2
        self.j_rec = 0
        self.rec_warmup = 3
        self.trade_off = 0.01#.05  # 0.01
        self.x_train = []
        self.f_train = []
        self.v_train = []

    def next_recommendation(self):
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x DOMAIN.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code

        if self.j_rec < self.rec_warmup:
            recommendation = np.array([np.ones((DOMAIN.shape[0]))])
            recommendation *= DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                              np.random.rand(DOMAIN.shape[0])
        else:
            recommendation = self.optimize_acquisition_function()

        self.j_rec += 1
        return recommendation

        # return self.optimize_acquisition_function()

        # In implementing this function, you may use optimize_acquisition_function() defined below.

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x DOMAIN.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []
        x0_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])
            x0_values.append(x0)

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.
        Parameters
        ----------
        x: np.ndarray
            x in DOMAIN of f
        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        eps = 0.01
        #x = x + eps * np.random.rand()     # uncomment for eps greedy solution

        mu_f, sigma_f = self.f_map.predict([x], return_std=True)
        mu_v, sigma_v = self.v_map.predict([x], return_std=True)
        PI_v = norm.cdf(SAFETY_THRESHOLD + self.trade_off, loc=mu_v, scale=sigma_v)

        if PI_v < .5:
            return PI_v
        else:
            # Expected Improvement computation
            opt_f = max(self.past_points[:, 1])
            a = (mu_f - opt_f - self.trade_off) / sigma_f
            erf = math.erf(a / math.sqrt(2))
            # PI_f = scipy.stats.norm.cdf(a) # Probability of Improvement
            EI_f = (.5 * a * (1 + erf) + np.exp(-a ** 2 / 2) / math.sqrt(2 * math.pi)) * sigma_f    # Expected Improvement

            # gpucb =  mu_f + self.gpucb_kappa*sigma_f # UCB

            return EI_f     # PI_f, gpucb

        # raise NotImplementedError

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.
        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        x = np.atleast_2d(x)

        new_point = np.append(np.append(x, np.atleast_2d(f), axis=1), np.atleast_2d(v), axis=1)
        self.past_points = np.concatenate((self.past_points, new_point), axis=0)

        x_train = np.transpose(np.atleast_2d(self.past_points[:, 0]))
        f_train = np.transpose(np.atleast_2d(self.past_points[:, 1]))
        v_train = np.transpose(np.atleast_2d(self.past_points[:, 2]))
        self.f_map.fit(x_train, f_train)
        self.v_map.fit(x_train, v_train)

        # TODO: enter your code here
        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.
        Returns
        -------
        solution: np.ndarray
            1 x DOMAIN.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here

        filter = (self.past_points[:, 2] > SAFETY_THRESHOLD + 0)#self.trade_off)
        sol = self.past_points[filter]

        try:
            opt_index_x = np.argmax(sol[:, 1])
        except:
            opt_index_x = np.argmax(self.past_points[:, 1])
            sol = self.past_points

        return sol[opt_index_x, 0]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
