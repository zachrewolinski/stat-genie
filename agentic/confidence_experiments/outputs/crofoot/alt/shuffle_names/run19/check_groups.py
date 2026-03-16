import pandas as pd

pd.set_option('display.width', 140)

df = pd.read_csv('crofoot.csv')

# group sizes per id
for col_id, size_col, label in [
    ('n_other', 'f_other', 'f_other by n_other'),
    ('dist_other', 'f_other', 'f_other by dist_other'),
    ('n_other', 'win', 'win by n_other'),
    ('dist_other', 'win', 'win by dist_other'),
]:
    grp = df.groupby(col_id)[size_col].nunique().sort_index()
    print(label)
    print(grp)
    print()

