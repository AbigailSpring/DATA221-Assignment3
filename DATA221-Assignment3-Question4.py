import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

kidney_csv_data = pd.read_csv("kidney_disease.csv")
kidney_specific_data = kidney_csv_data.dropna()
features = kidney_specific_data.drop("classification", axis = 1)
features_encoded = pd.get_dummies(features)
target_labels = kidney_specific_data["classification"]
features_train, features_test, labels_train, labels_test = train_test_split(features_encoded, target_labels, test_size=0.3, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(features_train, labels_train)
predictions_of_target_labels = knn_classifier.predict(features_test)

conf_matrix = confusion_matrix(labels_test, predictions_of_target_labels)
test_accuracy = accuracy_score(labels_test, predictions_of_target_labels)
test_precision = precision_score(labels_test, predictions_of_target_labels , pos_label="ckd")
test_recall = recall_score(labels_test, predictions_of_target_labels, pos_label="ckd")
test_f1_score = f1_score(labels_test, predictions_of_target_labels, pos_label="ckd")

print("Results: ")
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1_score)

"""
In kidney disease prediction, a True Positive occurs when the model correctly predicts
that a patient has kidney disease, and a True Negative (TN) occurs when the model 
correctly predicts that a patient does not have the disease. A False Positive happens 
when the model incorrectly predicts disease for a healthy patient, while a False Negative 
occurs when the model fails to detect kidney disease in a patient who actually has it.
Accuracy alone may not be enough to evaluate the model because it does not show how well 
the model detects the positive cases, especially if the dataset is imbalanced. If missing 
a kidney disease case is very serious, sensitivity is the most important because it 
measures how well the model correctly identifies patients who truly have the disease, 
lowering the amounts of false negatives.
"""