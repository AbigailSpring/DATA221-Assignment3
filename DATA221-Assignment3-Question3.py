import pandas as pd
from sklearn.model_selection import train_test_split

kidney_csv_data = pd.read_csv("kidney_disease.csv")
kidney_specific_data = kidney_csv_data.dropna()
features = kidney_specific_data.drop("classification", axis = 1)
target_labels = kidney_specific_data["classification"]
x_train, x_test, y_train, y_test = train_test_split(features, target_labels, test_size = 0.3, random_state = 42)

print("Results: ")
print("Training set size: ", len(x_train))
print("Testing set size: ", len(x_test))

"""
 - We should not train and test a model on the same data because the model would 
 memorize the data rather than learn general patterns. It would give high performance
 results and would not reflect how the model performs on unseen data. 
 - The purpose of the testing set is to evaluate how well the trained model can make
 predictions on data its never seen before. It helps measure the model's ability to 
 generalize and ensures that the results are reliable.
"""