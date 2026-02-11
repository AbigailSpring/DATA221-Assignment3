import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

kidney_csv_data = pd.read_csv("kidney_disease.csv")
kidney_specific_data = kidney_csv_data.dropna()
x = kidney_specific_data.drop("classification", axis=1)
y = kidney_specific_data["classification"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_predictions = knn.predict(x_test)

cm = confusion_matrix(y_test, y_predictions)
accuracy = accuracy_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions, pos_label="ckd")
recall = recall_score(y_test, y_predictions, pos_label="ckd")
f1 = f1_score(y_test, y_predictions, pos_label="ckd")

print("Results: ")
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)