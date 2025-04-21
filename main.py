from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score

# Fetch the dataset
student_performance = fetch_ucirepo(id=320)
X, y_continuous = (
    student_performance.data.features.loc[:, ["freetime"]].values,
    student_performance.data.targets.loc[:, "G1"].values,
)

# Define a threshold for binary classification (e.g., 10)
threshold = 10
y_binary = (y_continuous >= threshold).astype(int)  # 1 for Pass, 0 for Fail

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Initialize the KNeighborsClassifier model
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_classifier.predict(X_test)

# Evaluate the model using Jaccard score
jaccard = jaccard_score(y_test, predictions)
print("Predictions (Pass/Fail):", predictions)
print(f"Jaccard Score: {jaccard}")
