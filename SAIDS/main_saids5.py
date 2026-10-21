import scipy.stats as stats

# Sample data with 10 numbers for three distinct groups
group1 = [25, 30, 28, 32, 29, 26, 31, 27, 33, 30]
group2 = [45, 48, 50, 42, 47, 46, 49, 44, 51, 48]
group3 = [22, 25, 20, 24, 21, 23, 19, 26, 22, 25]

# Perform One-Way ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)
print("P-value:", p_val)

import scipy.stats as stats

# Same or non-normal data samples with 10 numbers for each group
group1 = [25, 30, 28, 32, 29, 26, 31, 27, 33, 30]
group2 = [45, 48, 50, 42, 47, 46, 49, 44, 51, 48]
group3 = [22, 25, 20, 24, 21, 23, 19, 26, 22, 25]

# Perform Kruskal-Wallis H-test
h_stat, p_val = stats.kruskal(group1, group2, group3)
print("H-Statistic:", h_stat)
print("P-value:", p_val)

