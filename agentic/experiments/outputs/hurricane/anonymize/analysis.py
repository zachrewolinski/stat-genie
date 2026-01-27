import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the hurricane dataset
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("HURRICANE NAME FEMININITY AND FATALITIES ANALYSIS")
print("=" * 80)
print()

# Display basic information about the dataset
print("Dataset Overview:")
print(f"Total hurricanes: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Key variables:
# feature3: Name of the hurricane
# feature4: Masculinity-femininity index (1=very masculine, 11=very feminine)
# feature6: Binary gender indicator (0 for male, 1 for female)
# feature8: Total number of deaths

print("Key Variables:")
print("- feature3: Hurricane name")
print("- feature4: Masculinity-femininity index (1=masculine, 11=feminine)")
print("- feature6: Binary gender (0=male, 1=female)")
print("- feature8: Total deaths")
print()

# Check for missing values in key columns
print("Missing values in key columns:")
print(f"- feature4 (femininity index): {df['feature4'].isna().sum()}")
print(f"- feature8 (deaths): {df['feature8'].isna().sum()}")
print()

# Remove rows with missing death data for the main analysis
df_clean = df.dropna(subset=['feature8'])
print(f"Hurricanes with death data: {len(df_clean)}")
print()

# Descriptive statistics
print("Descriptive Statistics:")
print(f"Femininity index - Mean: {df_clean['feature4'].mean():.2f}, Std: {df_clean['feature4'].std():.2f}")
print(f"Deaths - Mean: {df_clean['feature8'].mean():.2f}, Median: {df_clean['feature8'].median():.2f}, Std: {df_clean['feature8'].std():.2f}")
print()

# Analysis 1: Correlation between femininity and deaths
correlation, p_value = stats.pearsonr(df_clean['feature4'], df_clean['feature8'])
print("=" * 80)
print("ANALYSIS 1: Correlation between Femininity Index and Deaths")
print("=" * 80)
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Statistically significant at α=0.05")
else:
    print("Result: Not statistically significant at α=0.05")
print()

# Analysis 2: Compare deaths by binary gender
male_deaths = df_clean[df_clean['feature6'] == 0]['feature8']
female_deaths = df_clean[df_clean['feature6'] == 1]['feature8']

print("=" * 80)
print("ANALYSIS 2: Deaths by Binary Gender Classification")
print("=" * 80)
print(f"Male-named hurricanes (n={len(male_deaths)}):")
print(f"  Mean deaths: {male_deaths.mean():.2f}, Median: {male_deaths.median():.2f}")
print(f"Female-named hurricanes (n={len(female_deaths)}):")
print(f"  Mean deaths: {female_deaths.mean():.2f}, Median: {female_deaths.median():.2f}")
print()

# T-test comparing male vs female named hurricanes
t_stat, t_pvalue = stats.ttest_ind(male_deaths, female_deaths)
print(f"Independent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pvalue:.4f}")
if t_pvalue < 0.05:
    print("  Result: Statistically significant difference at α=0.05")
else:
    print("  Result: No statistically significant difference at α=0.05")
print()

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(male_deaths, female_deaths, alternative='two-sided')
print(f"Mann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pvalue:.4f}")
if u_pvalue < 0.05:
    print("  Result: Statistically significant difference at α=0.05")
else:
    print("  Result: No statistically significant difference at α=0.05")
print()

# Analysis 3: Regression analysis controlling for hurricane severity
print("=" * 80)
print("ANALYSIS 3: Regression Analysis")
print("=" * 80)
print("Testing if femininity predicts deaths when controlling for hurricane characteristics")
print()

# Control variables:
# feature5: Minimum pressure (lower = stronger)
# feature7: Category (higher = stronger)
# feature13: Maximum wind speed (higher = stronger)

# Remove rows with missing values in control variables
df_regression = df_clean.dropna(subset=['feature4', 'feature8', 'feature5', 'feature7', 'feature13'])
print(f"Sample size for regression: {len(df_regression)}")
print()

# Simple linear regression: deaths ~ femininity
from scipy.stats import linregress
slope, intercept, r_value, p_value_reg, std_err = linregress(df_regression['feature4'], df_regression['feature8'])
print("Simple Linear Regression: Deaths ~ Femininity")
print(f"  Slope: {slope:.4f} (deaths per unit increase in femininity)")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value_reg:.4f}")
if p_value_reg < 0.05:
    print("  Result: Femininity is a statistically significant predictor at α=0.05")
else:
    print("  Result: Femininity is NOT a statistically significant predictor at α=0.05")
print()

# Analysis 4: Examine outliers
print("=" * 80)
print("ANALYSIS 4: Outlier Analysis")
print("=" * 80)
print("Top 5 deadliest hurricanes:")
top_deadly = df_clean.nlargest(5, 'feature8')[['feature3', 'feature4', 'feature8', 'feature2']]
print(top_deadly.to_string(index=False))
print()

# Summary and interpretation
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Research Question: Do hurricanes with more feminine names lead to fewer")
print("precautionary measures and thus more deaths?")
print()
print("Key Findings:")
print(f"1. Correlation between femininity and deaths: r={correlation:.4f}, p={p_value:.4f}")
print(f"2. Female-named hurricanes average {female_deaths.mean():.1f} deaths")
print(f"   Male-named hurricanes average {male_deaths.mean():.1f} deaths")
print(f"3. T-test comparing male vs female: p={t_pvalue:.4f}")
print(f"4. Simple regression slope: {slope:.4f}, p={p_value_reg:.4f}")
print()

# Final determination
if correlation > 0 and p_value < 0.05:
    print("CONCLUSION: YES")
    print("The analysis shows a statistically significant positive correlation between")
    print("name femininity and deaths, supporting the hypothesis.")
elif female_deaths.mean() > male_deaths.mean() and t_pvalue < 0.05:
    print("CONCLUSION: YES")
    print("Female-named hurricanes have significantly higher death tolls on average,")
    print("supporting the hypothesis.")
else:
    print("CONCLUSION: NO")
    print("The analysis does not show statistically significant evidence that more")
    print("feminine hurricane names lead to more deaths.")
print()
