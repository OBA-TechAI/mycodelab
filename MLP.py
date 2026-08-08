from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
print("Loading digits dataset...\n")
digits = load_digits()
X = digits.data
y = digits.target

# 2. Preprocess
print("Preprocessing data...\n")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Define and train the model
print("Training the MLP model...\n")
mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                    max_iter=20, random_state=1, verbose=True)
mlp.fit(X_train, y_train)

# 5. Evaluate
print("Evaluating the model...\n")
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2%}")
