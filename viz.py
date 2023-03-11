import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import least_squares
import pandas as pd
import re
from tqdm import tqdm
import datetime as dt

df = pd.read_csv('project10_data.csv')

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

T_max = 119
# take T_max columns from t0
df_clean = df_clean.iloc[:, t0:t0+T_max]

# Use dual axis, one for infected and one for death
fig, (ax1, ax3) = plt.subplots(nrows=2, figsize=(12, 7), dpi=200, sharex=True)
ax2 = ax1.twinx()
ax1.plot(df_clean.columns, df_clean.iloc[0, :], label="Infected", color='b')
ax2.plot(df_clean.columns, df_clean.iloc[1, :], label="Death", color='r')
plt.title('Time series of the COVID cases in San Francisco, California (Population: 881549)')
plt.xlabel('Date')
# plt.xlim("2020-03-07", "2020-07-15")
plt.xlim(dt.date(2020, 3, 7), dt.date(2020, 7, 15))
plt.ylabel('Number of cases')
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')


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


T_max = 119
# take T_max columns from t0
df_clean = df_clean.iloc[:, t0:t0+T_max]

ax4 = ax3.twinx()
ax3.plot(df_clean.columns, df_clean.iloc[0, :], label="Infected", color='b')
ax4.plot(df_clean.columns, df_clean.iloc[1, :], label="Death", color='r')
plt.xlim(dt.date(2020, 3, 7), dt.date(2020, 7, 15))
# plt.xlim("2020-03-07", "2020-07-15")
plt.title('Time series of the COVID cases in District of Columbia (Population: 705749)')
plt.ylabel('Number of cases')
ax3.legend(loc='upper left')
ax4.legend(loc='lower right')

plt.savefig('Time_series.png')
plt.show()
