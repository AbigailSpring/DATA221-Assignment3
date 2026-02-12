import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

kidney_csv_data = pd.read_csv("kidney_disease.csv")
kidney_specific_data = kidney_csv_data.dropna()
features = kidney_specific_data.drop("classification", axis = 1)
features_encoded = pd.get_dummies(features)
target_labels = kidney_specific_data["classification"]
features_train, features_test, labels_train, labels_test = train_test_split(features_encoded, target_labels, test_size=0.3, random_state=42)
k_values_to_test = [1, 3, 5, 7, 9]
test_accuracies = []
print("Results: ")
for k_value in k_values_to_test:
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(features_train, labels_train)
    predicted_test_labels = knn_model.predict(features_test)
    accuracy_for_k = accuracy_score(labels_test, predicted_test_labels)
    test_accuracies.append(accuracy_for_k)
    print(f"k = {k_value}, Test Accuracy = {accuracy_for_k:.4f}")
    best_index = test_accuracies.index(max(test_accuracies))
    best_k = k_value[best_index]
    best_accuracy = test_accuracies[best_index]
    print("\nBest k value:", best_k)
    print("Test Accuracy for best k:", best_accuracy)