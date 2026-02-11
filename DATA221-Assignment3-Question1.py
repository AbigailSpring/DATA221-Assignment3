import pandas as pd
import numpy as py
crime_data_cvs = pd.read_cvs("crime.cvs")
crime_column = crime_data_cvs["ViolentCrimesPerPop"]
crime_data_mean = crime_column.mean()
crime_data_median = crime_column.median()
crime_data_standard_deviation = crime_column.std()
crime_data_minimum_value = crime_column.min()
crime_data_maximum_value = crime_column.max()