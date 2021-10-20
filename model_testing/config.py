from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL = [
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    DecisionTreeClassifier(),
]
MODEL_TITLE = [
    "Random Forest Classifier",
    "Logistic Regression",
    "Support Vector Classifier",
    "Decision Tree Classifier",
]
DATASET_PATH = "dataset/online_shoppers_intention.csv"
OBJECT_COLUMNS = ["VisitorType", "Month", "Revenue"]
X_OBJECT_COLUMNS = ["VisitorType", "Month"]
Y_COLUMN = "Revenue"
X_CATEGORICAL_COLUMNS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
