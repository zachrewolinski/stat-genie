import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# quick correlations between numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
print('numeric columns', num_cols.tolist())

corr = df[num_cols].corr()
print('\ncorrelation with adjusted_running_time')
print(corr['adjusted_running_time'].sort_values(ascending=False).head(10))

print('\ncorrelation with age')
print(corr['age'].sort_values(ascending=False).head(10))

print('\ncorrelation with gender')
print(corr['gender'].sort_values(ascending=False).head(10))

# Check if adjusted_running_time approx age + gender or similar
for a,b in [('adjusted_running_time','age'),('adjusted_running_time','gender'),('age','gender')]:
    diff = (df[a] - df[b]).abs()
    print(f"\n{a}-{b} abs diff summary: min {diff.min()} median {diff.median()} mean {diff.mean()} max {diff.max()}")

# Explore if adjusted_running_time approx age + gender
sum_ag = df['age'] + df['gender']
ratio = df['adjusted_running_time'] / sum_ag
print('\nadjusted_running_time / (age+gender) summary', ratio.describe())

# Try compute speed from word count (retake_trial) and time columns
words = df['retake_trial']
for time_col in ['adjusted_running_time','age','gender']:
    # assume ms
    wpm = words / (df[time_col]/1000.0) * 60
    print(f"\nWPM using {time_col} (ms) summary")
    print(wpm.describe())

# Compare running_time to derived speeds
for time_col in ['adjusted_running_time','age','gender']:
    wpm = words / (df[time_col]/1000.0) * 60
    corr_rt = np.corrcoef(wpm, df['running_time'])[0,1]
    print(f"corr of derived wpm (from {time_col}) with running_time: {corr_rt}")

# Check if running_time equals words / (time) maybe? try a linear regression scale factor
for time_col in ['adjusted_running_time','age','gender']:
    wpm = words / (df[time_col]/1000.0) * 60
    # compute ratio running_time / wpm
    ratio = df['running_time'] / wpm
    print(f"\nratio running_time / wpm from {time_col} summary")
    print(ratio.describe())
