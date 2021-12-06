import numpy as np

def softmax(x, axis = 0):
    """
    Helper method to calculate softmax values
    """
    # Use the LogSumExp Trick
    max_val = np.amax(x, axis=axis, keepdims = True)
    x = x - max_val

    # Softmax
    num = np.exp(x)
    denum = num.sum(axis = axis, keepdims = True)
    softmax = num/denum
    return softmax

def log_to_simple(x):
    """
    Helper method to convert log returns to simple returns
    """
    return np.exp(np.array(x)) - 1

def simple_to_log(x):
    """
    Helper method to convert simple returns to log returns
    """
    return np.log(np.array(x) + 1)

def sharpe_ratio(return_series, N = 255, rf = 0.01, annualized = True):
    """
    Helper method to calculate sharpe ratio
    """
    mean = return_series.mean() * N -rf
    sigma = return_series.std()
    if annualized:
        sigma *= np.sqrt(N)
    return mean / sigma