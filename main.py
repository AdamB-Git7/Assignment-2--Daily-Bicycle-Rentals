import numpy as np


# Part A — Load and validate numeric import
hires_2018 = np.loadtxt(
    "dataset/tfl-daily-cycle-hires-2018.csv", delimiter=",", skiprows=1, usecols=1
)
hires_2019 = np.loadtxt(
    "dataset/tfl-daily-cycle-hires-2019.csv", delimiter=",", skiprows=1, usecols=1
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
tfl_1819 = np.concatenate((hires_2018, hires_2019), axis=1)

print("\nPART B — Combine years")
print("tfl_1819 shape:", tfl_1819.shape)
print("First five rows [2018, 2019]:\n", tfl_1819[:5])


# Part C — Slice a 30-day window and transpose it
window_30_days = tfl_1819[99:129, :]
window_30_days_transposed = window_30_days.T

print("\nPART C — 30-day window and transpose")
print("Days 100–129 shape:", window_30_days.shape)
print("Days 100–129 [2018, 2019]:\n", window_30_days)
print("Transposed window shape:", window_30_days_transposed.shape)
print("Transposed window [year, day]:\n", window_30_days_transposed)


# Part D - Boolean filtering and the flattening pitfall
mean_2019 = np.mean(tfl_1819[:, 1])
above_mean_2019_mask = tfl_1819[:, 1] > mean_2019
above_mean_2019_hires = tfl_1819[:, 1][above_mean_2019_mask]

print("\nPART D - Boolean filtering")
print("2019 mean:", mean_2019)
print("Boolean mask:", above_mean_2019_mask)
print("Boolean mask shape:", above_mean_2019_mask.shape)
print("Number of True values:", np.count_nonzero(above_mean_2019_mask))
print("Filtered 2019 hires:", above_mean_2019_hires)
print("Original tfl_1819 shape:", tfl_1819.shape)
print("2019 column before filtering shape:", tfl_1819[:, 1].shape)
print("Filtered result shape:", above_mean_2019_hires.shape)
print(
    "Boolean indexing packs the selected values into a flattened 1D array, "
    "so the result has shape (number of True values,) rather than (365, 2)."
)


# Part E - Vectorized arithmetic and broadcasting/shape compatibility
diff = tfl_1819[:, 1] - tfl_1819[:, 0]

print("\nPART E - Daily year-over-year difference")
print("2019 column shape:", tfl_1819[:, 1].shape)
print("2018 column shape:", tfl_1819[:, 0].shape)
print("diff shape:", diff.shape)
print("Daily differences (2019 - 2018):", diff)
print("Mean difference:", diff.mean())
print("Maximum difference:", diff.max())
print("Minimum difference:", diff.min())
print(
    "This is vectorized because one subtraction operates on all 365 matching "
    "elements. The two arrays have compatible identical shapes, so no manual "
    "loop is needed."
)

# Part F - Summary analytics + export results
total_hires = np.sum(tfl_1819, axis=0)
average_daily_hires = np.mean(tfl_1819, axis=0)
maximum_daily_hires = np.max(tfl_1819, axis=0)

print("\nPART F - Summary analytics")
print("Total hires [2018, 2019]:", total_hires)
print("Average daily hires [2018, 2019]:", average_daily_hires)
print("Maximum daily hires [2018, 2019]:", maximum_daily_hires)

# Export the combined 2018/2019 data
np.savetxt(
    "dataset/tfl_1819.csv",
    tfl_1819,
    delimiter=","
)

# Export the daily differences
np.savetxt(
    "dataset/diff.csv",
    diff,
    delimiter=","
)

print("Exported tfl_1819 to dataset/tfl_1819.csv")
print("Exported diff to dataset/diff.csv")

# Part G — Creation/manipulation check
temperatures = np.loadtxt(
    "dataset/warehouse_temperatures_jan.csv",
    delimiter=".",
    skiprows=1,
    usecols=1
)

temperature_count = np.count_nonzero(temperatures)
temperature_min = np.min(temperatures)
temperature_max = np.max(temperatures)
temperature_mean = np.mean(temperatures)

sorted_temperatures = np.sort(temperatures)
temperatures_matrix = temperatures.reshape(31, 1)

print("\nPART G — Temperature check")
print("Temperature count:", temperature_count)
print("Minimum temperature:", temperature_min)
print("Maximum temperature:", temperature_max)
print("Mean temperature:", temperature_mean)
print("Sorted temperatures:", sorted_temperatures)
print("Reshaped temperatures shape:", temperatures_matrix.shape)

