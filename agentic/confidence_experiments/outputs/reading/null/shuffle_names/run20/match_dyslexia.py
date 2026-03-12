import pandas as pd

_df = pd.read_csv('reading.csv')

# candidates for dyslexia status
candidates = ['device', 'dyslexia']

binary_cols = ['correct_rate', 'dyslexia_bin', 'language']

for cand in candidates:
    cand_bin = (_df[cand] > 0).astype(int)
    print('\nCandidate', cand)
    for b in binary_cols:
        # align on non-missing
        sub = _df[[b]].copy()
        valid = sub[b].notna()
        match = (cand_bin[valid].values == _df.loc[valid, b].values).mean()
        # compute correlation
        corr = pd.Series(cand_bin[valid]).corr(_df.loc[valid, b])
        print('  vs', b, 'match', match, 'corr', corr)
