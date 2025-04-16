from ucimlrepo import fetch_ucirepo
import numpy as np
from collections import Counter
import sklearn

student_performance = fetch_ucirepo(id=320)

X, y = (
    student_performance.data.features.loc[:, "freetime"].values,
    student_performance.data.targets.loc[:, "G1"].values,
)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2
)
k = 3
predictions = []
for x in X_test:
    distances = []
    for x_train in X_train:
        distance = np.sqrt(np.sum((x - x_train) ** 2))
        distances.append(distance)
    distances = np.array(distances)
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    predictions.append(most_common[0][0])
predictions = np.array(predictions)
print("Predições:", predictions)
