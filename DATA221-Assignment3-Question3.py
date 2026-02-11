import pandas as pd
from sklearn.model_selection import train_test_split

kidney_csv_data = pd.read("kidney_disease.csv")
kidney_specific_data = kidney_csv_data.dropna()
x = kidney_specific_data.drop("Classification", axis = 1)
y = kidney_specific_data["Classification"]
