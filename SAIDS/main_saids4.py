import numpy as np
from scipy import stats

# Hardcoded Data and Parameters
# Sample data generated around an actual mean of ~53
data = np.array([52.1, 49.5, 55.3, 48.1, 56.4, 51.2, 53.8, 50.9, 54.2, 57.1])
pop_mean = 50
alpha = 0.05

# Running the One-Sample T-Test
t_stat, p_value = stats.ttest_1samp(data, pop_mean)

# Printing Observed Metrics
print("OBSERVED METRICS:")
print(f"Sample Mean		:{np.mean(data):.2f}")
print(f"Hypothesized Mean	:{pop_mean}")
print(f"T-Statistic		:{t_stat:.4f}")
print(f"P-Value			:{p_value:.4f}")
print(f"Significance Level	:{alpha}\n")

# Strict If-Else Conclusion
print("CONCLUSION:")
if p_value < alpha:
    print(f"alternate hypothesis satisfied: The p-value ({p_value:.4f}) is less than the significance level ({alpha}), indicating a statistically significant difference from the hypothesized mean.")
else:
    print(f"null hypothesis satisfied: The p-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}), indicating no statistically significant difference from the hypothesized mean.")

