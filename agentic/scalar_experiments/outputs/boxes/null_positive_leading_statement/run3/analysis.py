import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('boxes.csv')

    # Map outcome codes: 1=unchosen, 2=majority, 3=minority
    n = len(df)
    majority_rate = (df['y'] == 2).mean()
    minority_rate = (df['y'] == 3).mean()
    unchosen_rate = (df['y'] == 1).mean()

    # By culture
    culture_group = df.groupby('culture')['y']
    majority_by_culture = culture_group.apply(lambda s: (s == 2).mean())
    minority_by_culture = culture_group.apply(lambda s: (s == 3).mean())

    # By age (treat as continuous, compute correlation with majority choice)
    df['is_majority'] = (df['y'] == 2).astype(int)
    df['is_minority'] = (df['y'] == 3).astype(int)

    age_majority_corr = df['age'].corr(df['is_majority'])
    age_minority_corr = df['age'].corr(df['is_minority'])

    # Simple age bins
    bins = [4, 6, 9, 12, 14.01]
    labels = ['4-5', '6-8', '9-11', '12-14']
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    majority_by_age_bin = df.groupby('age_bin')['is_majority'].mean()

    # Interaction: culture x age_bin majority preference
    culture_age_majority = df.pivot_table(
        index='culture', columns='age_bin', values='is_majority', aggfunc='mean'
    )

    print('N =', n)
    print('Overall rates:')
    print('  Majority choice   :', round(majority_rate, 3))
    print('  Minority choice   :', round(minority_rate, 3))
    print('  Unchosen third opt:', round(unchosen_rate, 3))
    print('\nBy culture (majority, minority rates):')
    for c in sorted(majority_by_culture.index):
        print(
            f"  Culture {c}: majority={majority_by_culture[c]:.3f}, minority={minority_by_culture[c]:.3f}"
        )

    print('\nAge correlations:')
    print('  Corr(age, majority) =', round(age_majority_corr, 3))
    print('  Corr(age, minority) =', round(age_minority_corr, 3))

    print('\nMajority by age bin:')
    for bin_label, val in majority_by_age_bin.items():
        print(f"  {bin_label}: {val:.3f}")

    print('\nCulture x age-bin majority rates:')
    print(culture_age_majority.round(3))


if __name__ == '__main__':
    main()
