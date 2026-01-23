# Research Report: Children and Extramarital Affairs

## Summary

The unadjusted analysis shows having children is associated with higher affairs rates. However, this association disappears after controlling for confounders. After adjusting for marriage duration, satisfaction, and demographics, the association becomes negligible (coefficient: 0.005, p=0.989). The crude association is driven by confounding: couples with children have been married longer, and marriage duration independently predicts affairs.

## Key Descriptive Statistics

**Sample Overview**
- Total sample size: 601 individuals
- Proportion with no affairs: 75.0% (451/601)
- Proportion with any affairs: 25.0% (150/601)

**Children Status**
- Marriages with children: 71.5% (430/601)
- Marriages without children: 28.5% (171/601)

**Affairs by Children Status**
| Group | N | Mean Affairs | Median | % Any Affairs |
|-------|---|--------------|--------|---------------|
| No children | 171 | 0.912 | 0.0 | 15.8% |
| Has children | 430 | 1.672 | 0.0 | 28.6% |

## Main Results

**Model A (Unadjusted):** Simple regression shows children associated with 0.760 more affairs (95% CI: [0.177, 1.343], p=0.011).

**Model B (Adjusted):** After controlling for age, years married, marriage rating, religiousness, education, occupation, and gender, the coefficient drops to 0.005 (95% CI: [-0.701, 0.711], p=0.989), essentially zero.

**Logistic Model:** Modeling any affairs (binary) shows 8.55 percentage point increase with children, non-significant (p=0.088).

**Key Confounders:** Marriage rating (happiness): coefficient -0.538 (p<0.001); years married: coefficient 0.195 (p<0.001). Children highly correlated with years married (r=0.573).

## Robustness Discussion

The stark contrast between adjusted and unadjusted estimates underscores the importance of controlling for marriage duration. Alternative specifications (logistic for binary outcome) show similar patterns: positive but non-significant effects. Results are sensitive to whether affairs is modeled as continuous or binary. The non-linear coding (0, 1, 2, 3, 7, 12) complicates interpretation of effect magnitudes.

## Limitations

This is observational data from a self-selected 1969 Psychology Today survey, limiting generalizability and precluding causal claims. Self-reported affairs likely suffer from social desirability bias. The children variable lacks detail on timing or number of children. Unmeasured confounders (relationship quality beyond the rating variable, personality, life stressors) may bias estimates. Selection into parenthood is non-random and may correlate with unobserved affair-related factors.
