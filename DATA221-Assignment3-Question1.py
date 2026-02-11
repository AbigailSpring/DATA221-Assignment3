import pandas as pd
import numpy as py
crime_data_cvs = pd.read_cvs("crime.cvs")
crime_column = crime_data_cvs["ViolentCrimesPerPop"]
