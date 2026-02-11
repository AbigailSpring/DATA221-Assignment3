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