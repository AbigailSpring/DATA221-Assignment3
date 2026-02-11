import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
crime_data_csv = pd.read_csv("crime1.csv")
crime_column = crime_data_csv["ViolentCrimesPerPop"]
plt.figure()
plt.hist(crime_column, bin = 20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
