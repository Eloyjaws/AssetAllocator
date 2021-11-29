import numpy as np

def softmax(x, axis = 0):
    # Use the LogSumExp Trick
    max_val = np.amax(x, axis=axis, keepdims = True)
    x = x - max_val

    # Softmax
    num = np.exp(x)
    denum = num.sum(axis = axis, keepdims = True)
    softmax = num/denum
    return softmax

def log_to_simple(x):
    return np.exp(np.array(x)) - 1

def simple_to_log(x):
    return np.log(np.array(x) + 1)

def sharpe_ratio(return_series, N = 255, rf = 0.01, annualized = True):
    mean = return_series.mean() * N -rf
    sigma = return_series.std()
    if annualized:
        sigma *= np.sqrt(N)
    return mean / sigma

class DifferentialSharpeRatio: 
    def __init__(self, eta=1e-6, last_A = 0, last_B = 0): 
        self.eta = eta 
        self.last_A = last_A
        self.last_B = last_B
        
    def _differential_sharpe_ratio(self, rt, eps=np.finfo('float64').eps):
        delta_A = rt - self.last_A
        delta_B = rt**2 - self.last_B

        top = self.last_B * delta_A - 0.5 * self.last_A * delta_B
        bottom = (self.last_B - self.last_A**2)**(3 / 2) + eps
        
        return (top / bottom)

    def get_reward(self, rt):
        dsr = self._differential_sharpe_ratio(rt)

        self.last_A += self.eta * (rt - self.last_A)
        self.last_B += self.eta * (rt**2 - self.last_B)

        return dsr