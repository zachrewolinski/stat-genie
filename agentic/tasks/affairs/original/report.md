# Analysis Report: Children and Extramarital Affairs

## Summary

In this sample of 601 individuals from a 1969 Psychology Today survey, unadjusted comparisons suggest those with children report more affairs (mean 1.672 vs 0.912, p=0.011). However, after controlling for age, years married, religiousness, and marriage satisfaction, the association disappears (coefficient: -0.143, 95% CI: [-0.830, 0.545], p=0.684). The unadjusted association is confounded by the fact that individuals with children are older, married longer, and report lower marriage satisfaction. Multiple modeling approaches (linear, log-transformed, Poisson) consistently show no significant association after adjustment.

## Key Descriptive Statistics

| Statistic | With Children (n=430) | Without Children (n=171) |
|-----------|----------------------|--------------------------|
| Mean affairs | 1.672 | 0.912 |
| Median affairs | 0.0 | 0.0 |
| Proportion with zero affairs | 71.4% | 84.2% |
| Mean age | 35.0 years | 26.3 years |
| Mean years married | 10.2 years | 3.1 years |
| Mean religiousness | 3.21 | 2.88 |
| Mean marriage rating | 3.80 | 4.27 |

**Overall sample:** 75% reported zero affairs; mean=1.456, median=0.

## Main Results

**Simple model (unadjusted):** Having children is associated with 0.760 more affairs on average (p=0.011).

**Adjusted model (with covariates):** After controlling for gender, age, years married, religiousness, education, occupation, and marriage rating, having children shows a non-significant association with affairs (coefficient: -0.143, 95% CI: [-0.830, 0.545], p=0.684).

The adjusted model reveals that **years married** (β=0.170, p<0.001), **religiousness** (β=-0.478, p<0.001), and **marriage rating** (β=-0.712, p<0.001) are stronger predictors of affairs than children status. Alternative model specifications (log-transformed outcome, Poisson regression) yield similar null results (p=0.872 and p=0.979, respectively).

## Robustness and Modeling Choices

Results are sensitive to covariate adjustment. The crude association reflects confounding by age and marriage duration. Three model specifications (linear, log-transformed, Poisson) all show consistent null effects after adjustment. Model diagnostics reveal substantial confounding: age, years married, religiousness, and rating all differ significantly by children status (all p<0.002).

## Limitations

This is an observational study based on 1969 survey data; causal claims cannot be made. The sample is not representative (self-selected Psychology Today readers). The "affairs" measure uses a non-standard scale and may suffer from reporting bias. Key confounders like relationship quality and individual personality traits are imperfectly measured. The cross-sectional design cannot disentangle temporal ordering. Results may not generalize to contemporary populations given substantial changes in social norms since 1969.
