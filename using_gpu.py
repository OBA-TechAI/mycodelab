from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Multi-layer Perceptron on the digits dataset.')
parser.add_argument('--random_state', type=int, default=1, help='Random state for reproducibility')
args = parser.parse_args()

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
	X_scaled, y, test_size=0.2, random_state=args.random_state
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

# Save the results to a file
# create file name with random state
output_file = f'res_{args.random_state}.txt'
with open(output_file, 'w') as f:
	f.write(f"Test Accuracy: {acc:.2%}\n")
	f.write(f"Random State: {args.random_state}\n")
	f.write("Model parameters:\n")
	f.write(str(mlp.get_params()))  # Save model parameters for reproducibility
print(f"Results saved to {output_file}")
