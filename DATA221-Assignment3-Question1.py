import pandas as pd

crime_data_csv = pd.read_csv("crime1.csv")
crime_column = crime_data_csv["ViolentCrimesPerPop"]
crime_data_mean = crime_column.mean()
crime_data_median = crime_column.median()
crime_data_standard_deviation = crime_column.std()
crime_data_minimum_value = crime_column.min()
crime_data_maximum_value = crime_column.max()

print("Results: ")
print("Mean Value: ", crime_data_mean)
print("Median Value: ", crime_data_median)
print("Standard Deviation Value: ", crime_data_standard_deviation)
print("Minimum Value: ", crime_data_minimum_value)
print("Maximum Value: ", crime_data_maximum_value)

"""
- For this specific case, the mean and median are approximately equal.This indicates that
the distribution is symmetric. 
- The mean is more affected by extreme values because it uses every data value in its 
calculation. Extremely large or small values would directly change the average. In addition, 
the median is less affected by extreme values because it only depends on the middle position
of the ordered data and not the actual size of the values.
"""