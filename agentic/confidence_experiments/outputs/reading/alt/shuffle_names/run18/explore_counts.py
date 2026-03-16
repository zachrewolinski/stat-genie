import pandas as pd

pd.set_option('display.max_rows', 20)

df = pd.read_csv('reading.csv')

for col in ['language','device','dyslexia','dyslexia_bin','correct_rate','Flesch_Kincaid','retake_trial','img_width','english_native','page_id']:
    if col not in df.columns:
        continue
    print('\n', col)
    print(df[col].value_counts(dropna=False).sort_index())
