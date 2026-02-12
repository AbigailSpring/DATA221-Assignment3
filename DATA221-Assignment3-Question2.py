import pandas as pd
import matplotlib.pyplot as plt

crime_data_csv = pd.read_csv("crime1.csv")
crime_column = crime_data_csv["ViolentCrimesPerPop"]
plt.figure()
plt.hist(crime_column, bins = 20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")

plt.figure()
plt.boxplot(crime_column)
plt.title("Boxplot of Violent Crimes Per Population")
plt.ylabel("Violent Crimes Per Population")
plt.show()

"""
 - The histogram allows us to view how the data values are distributed across different 
 ranges. This allows us to see where the most observations are concentrated and whether 
 the distribution is symmetric or skewed.
 - The boxplot clearly displays the median within the box, showing the central value in
 the data set. It also shows the spread of the middle 50% through the interquartile range.
 If there are any individual points it suggests them as potential outliers.
"""