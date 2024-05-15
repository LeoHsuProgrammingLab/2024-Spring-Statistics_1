import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import t as t_table

def p1_1(A: np.ndarray, B: np.ndarray):
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    return round(mean_A, 2), round(mean_B, 2)

def p1_2(A: np.ndarray, B: np.ndarray):
    var_A = np.var(A, ddof=1)
    var_B = np.var(B, ddof=1)

    return round(var_A, 2), round(var_B, 2)

def p1_3(mean_A, mean_B, D0, SE, siginificant_level, df, is_two_tailed=True):
    t_stat = (mean_A - mean_B - D0) / SE
    if is_two_tailed:
        p_value = 2 * (1 - t_table.cdf(abs(t_stat), df))
    else:
        p_value = 1 - t_table.cdf(abs(t_stat), df)

    t_critical = t_table.ppf(1 - siginificant_level, df) if is_two_tailed else t_table.ppf(1 - siginificant_level / 2, df)
    margin_err = t_critical * SE

    lower = (mean_A - mean_B - D0) - margin_err
    upper = (mean_A - mean_B - D0) + margin_err

    return lower, upper, t_stat, p_value

def p1():
    # P1
    brand_A = np.array([24, 26, 25, 22, 23, 27, 28, 21, 20, 28, 24, 21, 24, 25, 26, 23, 22])
    brand_B = np.array([27, 28, 25, 29, 26, 30, 31, 24, 23, 30, 31, 24, 27, 28, 26, 25, 29, 25, 23, 26, 28, 29])
    len_A = len(brand_A)
    len_B = len(brand_B)

    mean_A, mean_B = p1_1(brand_A, brand_B)
    print("Mean of brand A:", mean_A)
    print("Mean of brand B:", mean_B)

    var_A, var_B = p1_2(brand_A, brand_B)
    print("Variance of brand A:", var_A)
    print("Variance of brand B:", var_B)

    if var_A == var_B:
        df = len_A + len_B - 2
        Sp = np.sqrt(((len_A - 1) * var_A + (len_B - 1) * var_B) / df)
        SE = Sp * np.sqrt(1 / len_A + 1 / len_B)
        print("Pooled standard deviation:", round(Sp, 2))

        lower, upper, t, p = p1_3(mean_A, mean_B, 0, SE, 0.05, df)
    else:
        c = (var_A / len_A) / (var_A / len_A + var_B / len_B)
        df = (len_A - 1) * (len_B - 1) / \
            ((1 - c) ** 2 * (len_A - 1) + c ** 2 * (len_B - 1))
        # df < (n1 + n2 - 2) if var_A != var_B
        SE = np.sqrt(var_A / len_A + var_B / len_B)
        lower, upper, t, p = p1_3(mean_A, mean_B, 0, SE, 0.05, df)

    print("Confidence interval:", round(lower, 2), round(upper, 2)) 
    print("t-statistic:", t)
    print("p-value:", p)

    type1_error = 0.05
    if p < type1_error:
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")

    # T-test for the means
    t_stat, p_value = ttest_ind(brand_A, brand_B, equal_var=False)  # Welch's t-test

    print(t_stat, p_value)

def p2():
    # P2
    # Data
    without_coffee = np.array([23, 35, 29, 33, 43, 32, 35, 41, 27, 28, 40])
    with_coffee = np.array([28, 38, 29, 37, 42, 30, 39, 43, 26, 32, 41])

    # Paired differences
    differences = with_coffee - without_coffee

    # Mean and standard deviation of differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Degrees of freedom
    df = len(differences) - 1
    D0 = 0
    t = (mean_diff - D0) / (std_diff / np.sqrt(len(differences)))
    alpha = 0.05
    is_two_tailed = True
    if is_two_tailed:
        t_critical = t_table.ppf(1 - alpha / 2, df)
    else:
        t_critical = t_table.ppf(1 - 0.05, df)
    
    margin_err = t_critical * std_diff / np.sqrt(len(differences))
    lower = mean_diff - margin_err
    upper = mean_diff + margin_err

    # Output
    print("Mean of differences:", round(mean_diff, 2))
    print("Standard deviation of differences:", round(std_diff, 2))
    print("Degrees of freedom:", df)
    print("t-statistic:", t)

    t_stat_diff, p_value_diff = ttest_rel(with_coffee, without_coffee)
    print("t-statistic:", t_stat_diff)
    print("p-value:", p_value_diff)


if __name__ == "__main__":
    p1()
    # p2()
