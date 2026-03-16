import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')
# treat retake_trial as word count; running_time as seconds
speed = df['retake_trial'] / df['running_time'] * 60
print(speed.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.99]))

# compare speed to plausible range
print('min', speed.min(), 'max', speed.max())

# also using num_words as word count
speed2 = df['num_words'] / df['running_time'] * 60
print('\nUsing num_words word count:')
print(speed2.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.99]))

