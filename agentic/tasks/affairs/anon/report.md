# Does Having Children Decrease Extramarital Affairs? Analysis Report

## Summary

This analysis examined the association between having children and engagement in extramarital affairs using data from 601 currently married individuals from a 1969 Psychology Today survey. Contrary to the hypothesis, the crude analysis shows that individuals with children reported *higher* mean affair frequency (1.67 vs 0.91, p=0.011) and a higher proportion reporting any affairs (28.6% vs 15.8%). However, after adjusting for key confounders including years married, age, religiousness, and marital satisfaction, the association becomes non-significant (adjusted β = -0.22, 95% CI: -0.90 to 0.46, p=0.525). This suggests that the crude association is confounded by factors that correlate with both having children (e.g., longer marriage duration) and affair risk. The adjusted models consistently show no significant association between children and affairs.

## Descriptive Statistics

| Characteristic | No Children (n=171) | Yes Children (n=430) |
|----------------|---------------------|----------------------|
| Mean affairs frequency | 0.91 | 1.67 |
| Median affairs | 0 | 0 |
| Proportion with any affairs | 15.8% | 28.6% |
| Mean age (years) | 32.5 | - |
| Mean years married | 8.2 | - |
| Mean religiousness (1-5) | 3.1 | - |
| Mean marital rating (1-5) | 3.9 | - |

Note: 75% of the sample reported zero affairs in the past year.

## Main Findings

**Model A (Unadjusted):** A simple linear regression shows a positive association between having children and affair frequency (β = 0.76, 95% CI: 0.18 to 1.34, p=0.011), suggesting those with children have 0.76 higher affair frequency on average.

**Model B (Adjusted):** After controlling for age, years married, religiousness, marital rating, education, and gender, the association reverses direction and becomes statistically non-significant (β = -0.22, 95% CI: -0.90 to 0.46, p=0.525). Key confounders identified include:
- Years married: β = 0.17 (p<0.001)
- Religiousness: β = -0.48 (p<0.001)
- Marital rating: β = -0.72 (p<0.001)
- Age: β = -0.05 (p=0.030)

**Robustness checks:** Poisson regression (IRR = 0.97, p=0.730) and logistic regression for any affairs (OR = 1.46, p=0.189) both confirm no significant association after adjustment. The inconsistency across model specifications reflects modeling uncertainty given the highly skewed outcome distribution.

## Robustness Discussion

The reversal from a significant crude association to a null adjusted association highlights the importance of confounding control. Years married is likely a major confounder: couples married longer are both more likely to have children and may face different affair risks. The choice of modeling approach matters given that 75% reported zero affairs. We examined OLS, Poisson, and logistic models. While effect magnitudes differ, all adjusted models consistently suggest no significant association. Results could change with different confounder selection, non-linear age/marriage duration effects, or interaction terms.

## Limitations

This is observational data; we cannot claim causal effects. The 1969 survey relied on self-reported sexual behavior, which may suffer from social desirability bias. The non-probability sample (Psychology Today readers who chose to respond) limits generalizability. The outcome variable uses a non-linear scale (0, 1, 2, 3, 7, 12) which complicates interpretation. Unmeasured confounders (e.g., relationship quality, opportunity for affairs) could bias results. The "children" variable is binary and provides no information about number or age of children, which may moderate effects. Findings from 1969 may not reflect contemporary relationships.
