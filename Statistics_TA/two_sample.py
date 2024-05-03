# two sample t-test
import random
import numpy as np
from scipy.stats import t as t_table

def two_sample_t_test(sample1: np.ndarray, sample2: np.ndarray, D0, alpha, two_tailed = True):
    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    var1 = np.var(sample1, ddof = 1)
    var2 = np.var(sample2, ddof = 1)
    
    equal_var = (var1 == var2)

    if equal_var:
        pooled_std = (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5
        SE = pooled_std * (1 / n1 + 1 / n2) ** 0.5
        df = n1 + n2 - 2
        t = (mean1 - mean2 - D0) / SE
        print(t)
        if two_tailed:
            p = 2 * (1 - t_table.cdf(abs(t), df))
            critical_t = t_table.ppf(1 - alpha / 2, df)
        else:
            p = 1 - t_table.cdf(abs(t), df)
            critical_t = t_table.ppf(1 - alpha, df)
    else:
        SE = (var1 / n1 + var2 / n2) ** 0.5
        c = (var1 / n1) / (var1 / n1 + var2 / n2) 
        df = (n1-1) * (n2-1) / ((1-c) ** 2 * (n1-1) + (c ** 2) * (n2-1))

        t = (mean1 - mean2 - D0) / SE
        if two_tailed:
            p = 2 * (1 - t_table.cdf(abs(t), df))
            critical_t = t_table.ppf(1 - alpha / 2, df)
        else:
            p = 1 - t_table.cdf(abs(t), df)
            critical_t = t_table.ppf(1 - alpha, df)

    return t, df, p, critical_t


if __name__ == "__main__":
    # sample1 = [random.gauss(0, 1) for _ in range(10)]
    # sample2 = [random.gauss(0, 1) for _ in range(9)]
    sample1 = [40, 54, 26, 63, 21, 37, 39, 23, 48, 58, 28, 39]
    sample2 = [18, 43, 28, 50, 16, 32, 13, 35, 38, 33, 6, 7]

    t, df, p, critical_t = two_sample_t_test(sample1, sample2, 0, 0.05, two_tailed = True)
    
    if (t > critical_t) or (t < -critical_t):
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")
    
    print("t-statistic:", t)
    print("p-value:", p)
    print("Critical t:", critical_t)
    print("Degrees of freedom:", df)
    