import pandas as pd
import numpy as np
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