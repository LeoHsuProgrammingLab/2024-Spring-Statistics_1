def variance_SD(variable_list, frequency_list, is_sample=False):
    mean = sum(score * freq for score, freq in zip(variable_list, frequency_list)) / sum(frequency_list)
    print("Mean:", mean)

    if is_sample:
        variance = sum((score - mean) ** 2 * freq for score, freq in zip(variable_list, frequency_list)) / (sum(frequency_list) - 1)
    else:
        variance = sum((score - mean) ** 2 * freq for score, freq in zip(variable_list, frequency_list))  / sum(frequency_list)
    return variance, variance ** 0.5

if __name__ == "__main__":
    # Scores and frequencies
    scores = [1, 4, 7, 10, 13, 16, 19, 22]
    frequencies = [4, 8, 10, 12, 25, 32, 6, 2]
    variance, SD = variance_SD(scores, frequencies, is_sample=0)
    print("Variance:", variance)
    print("Standard deviation:", SD)
