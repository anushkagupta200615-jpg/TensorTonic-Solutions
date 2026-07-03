import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    cdf = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
    return float(pmf), float(cdf)