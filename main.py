# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
iris_data = pd.read_csv('Iris.csv')

# Check if 'species' column exists in the dataframe before dropping it
if 'species' in iris_data.columns:
    # Separate features and target variable
    X = iris_data.drop('species', axis=1)
    y = iris_data['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    rf = RandomForestClassifier()

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    rf_pred = rf.predict(X_test)

    # Evaluate the model
    print("Random Forest Classifier Report:")
    print(classification_report(y_test, rf_pred))

    accuracy = accuracy_score(y_test, rf_pred)
    print("Accuracy:", accuracy)

    # Visualize the dataset
    fig, ax = plt.subplots()
    iris_data.plot.scatter(x='sepal_length', y='sepal_width', c='petal_length', cmap='viridis', ax=ax)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iris Dataset Visualization')
    plt.show()

else:
    print("'species' column not found in the dataframe.")

# plt.show()# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Separate features and target variable
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Make predictions
rf_pred = rf.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_pred))

accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy:", accuracy)

# Visualize the dataset
fig, ax = plt.subplots()
iris_data.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', c='petal length (cm)', cmap='viridis', ax=ax)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset Visualization')
plt.show()