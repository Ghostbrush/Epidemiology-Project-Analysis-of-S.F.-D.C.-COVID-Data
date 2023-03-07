import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import least_squares
import pandas as pd
import re
from tqdm import tqdm

df = pd.read_csv('Validation.csv')

# First to clean the table with only the time series data
col_list = []
for col in df.columns:
    if re.search("^[0-9]+", col):
        col_list.append(col)

# df_clean symbolizing all dates with given data
df_clean = df[col_list]
df_clean.columns = pd.to_datetime(df_clean.columns, format="%m/%d/%y")

# find t0 as the starting time for the simulation
for i in range(len(df_clean.columns)):
    if df_clean.iloc[0, i] >= 5:
        t0 = i
        break

print(t0)
print(df_clean.columns[i].date())

T_max = 40
# take T_max columns from t0
df_clean = df_clean.iloc[:, t0:t0+T_max]

# Use dual axis, one for infected and one for death
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax1.plot(df_clean.columns, df_clean.iloc[0, :], label="Infected", color='b')
ax2.plot(df_clean.columns, df_clean.iloc[1, :], label="Death", color='r')
plt.title('Time series of the COVID in District of Columbia')
plt.xlabel('Date')
plt.ylabel('Number of cases')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# plt.xticks(rotation=45)
# plt.xlim(df_clean.columns[0], df_clean.columns[-1])
plt.savefig('Time_series.png')
plt.show()
