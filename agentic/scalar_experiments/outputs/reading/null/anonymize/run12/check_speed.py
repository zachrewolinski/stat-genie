import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

df = df.rename(columns={
    "feature4": "time_total_ms",
    "feature5": "time_reading_ms",
    "feature7": "word_count",
    "feature20": "reading_speed",
})

speed_total = df["word_count"] * 60000.0 / df["time_total_ms"]

# Show first 10 rows of key columns
sample = df[["time_total_ms", "time_reading_ms", "word_count", "reading_speed"]].head(10)

print(sample.to_string(index=False))
print("\nComputed speed_total (first 10):")
print(speed_total.head(10).to_string(index=False))

# Show some rows with extreme values
idx_max = df["reading_speed"].idxmax()
idx_min = df["reading_speed"].idxmin()
print("\nMax reading_speed row:")
print(df.loc[idx_max, ["time_total_ms","time_reading_ms","word_count","reading_speed"]])
print("Computed speed_total:", speed_total.loc[idx_max])

print("\nMin reading_speed row:")
print(df.loc[idx_min, ["time_total_ms","time_reading_ms","word_count","reading_speed"]])
print("Computed speed_total:", speed_total.loc[idx_min])

# Correlation
valid = np.isfinite(df["reading_speed"]) & np.isfinite(speed_total)
print("\nCorrelation with speed_total:", np.corrcoef(df.loc[valid, "reading_speed"], speed_total[valid])[0,1])
