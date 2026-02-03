import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math


def main():
    df = pd.read_csv('amtl.csv')

    # Column mapping based on metadata/value patterns:
    # - sockets: tooth class (Anterior/Posterior/Premolar)
    # - genus: number of missing teeth (AMTL count)
    # - age: number of observable sockets (total trials)
    # - pop: estimated age at death (continuous)
    # - stdev_age: sex estimate (probability male, 0-1)
    # - tooth_class: taxon (Homo sapiens, Pan, Papio, Pongo)

    df = df.copy()
    df['total_sockets'] = df['age']
    df['missing'] = df['genus']

    # Guard against invalid rows
    df = df[df['total_sockets'] > 0].copy()

    # Proportion model with binomial weights
    df['prop_missing'] = df['missing'] / df['total_sockets']
    df['is_human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

    model = smf.glm(
        'prop_missing ~ is_human + pop + stdev_age + C(sockets)',
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df['total_sockets'],
    ).fit()

    coef = model.params['is_human']
    se = model.bse['is_human']
    or_val = math.exp(coef)
    ci_low = math.exp(coef - 1.96 * se)
    ci_high = math.exp(coef + 1.96 * se)

    print(model.summary())
    print('\nAdjusted odds ratio for Homo sapiens vs non-human primates:')
    print(f'OR = {or_val:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f}), p = {model.pvalues["is_human"]:.3g}')


if __name__ == '__main__':
    main()
