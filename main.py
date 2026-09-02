import numpy as np


# Part A — Load and validate numeric import
# skiprows=1 skips the header row; usecols=1 selects the hires column.
hires_2018 = np.loadtxt(
    "tfl-daily-cycle-hires-2018.csv", delimiter=",", skiprows=1, usecols=1
)
hires_2019 = np.loadtxt(
    "tfl-daily-cycle-hires-2019.csv", delimiter=",", skiprows=1, usecols=1
)

print("PART A — Load and reshape")
print("2018 loaded shape:", hires_2018.shape)
print("2018 dtype:", hires_2018.dtype)
print("2018 sample:", hires_2018[:5])
print("2019 loaded shape:", hires_2019.shape)
print("2019 dtype:", hires_2019.dtype)
print("2019 sample:", hires_2019[:5])

hires_2018 = hires_2018.reshape(365, 1)
hires_2019 = hires_2019.reshape(365, 1)

print("2018 reshaped to:", hires_2018.shape)
print("2019 reshaped to:", hires_2019.shape)


# Part B — Combine years
# axis=1 joins the matching daily rows as columns: 2018 in column 0, 2019 in column 1.
tfl_1819 = np.concatenate((hires_2018, hires_2019), axis=1)

print("\nPART B — Combine years")
print("tfl_1819 shape:", tfl_1819.shape)
print("First five rows [2018, 2019]:\n", tfl_1819[:5])


# Part C — Slice a 30-day window and transpose it
# Day 100 is at index 99, and slice endpoint 129 is excluded, giving days 100–129.
window_30_days = tfl_1819[99:129, :]
window_30_days_transposed = window_30_days.T

print("\nPART C — 30-day window and transpose")
print("Days 100–129 shape:", window_30_days.shape)
print("Days 100–129 [2018, 2019]:\n", window_30_days)
print("Transposed window shape:", window_30_days_transposed.shape)
print("Transposed window [year, day]:\n", window_30_days_transposed)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
