# Kyphosis Prediction with Decision Tree

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image  
from six import StringIO
from sklearn.tree import export_graphviz
import pydot 

# 1. Data Loading and Exploration
# Load the Kyphosis dataset (assuming it's in the same directory)
df = pd.read_csv("kyphosis.csv")

# Display the first few rows, information about columns, and summary statistics
print(df.head())
print(df.info())
print(df.describe())

# Visualize relationships between pairs of variables
sns.pairplot(data=df, hue='Kyphosis')
plt.show()  # Display the plot

# 2. Model Training and Prediction
# Split data into features (X) and target (y)
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Create and train a Decision Tree classifier
dtree = DecisionTreeClassifier(criterion='entropy') 
dtree.fit(X_train, y_train)

# 3. Model Evaluation
# Make predictions on the test set
predictions = dtree.predict(X_test)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

# Generate and print the classification report (precision, recall, F1-score)
print(classification_report(y_test, predictions))

# 4. Decision Tree Visualization
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
# Get the feature names (excluding 'Kyphosis')
features = list(df.columns[1:])

# Create a visual representation of the tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())  # Display the tree directly
