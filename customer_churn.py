# Imports nécessaires
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_data(path="Churn_Modelling.csv"):
    """Charge et prétraite les données."""
    df = pd.read_csv(path)                                      # chargement
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])   # texte -> nombre
    df = df.drop(columns=["Surname", "Geography"])              # colonnes inutiles
    x = df.drop(columns=["Exited", "RowNumber", "CustomerId"])  # variables
    y = df["Exited"]                                            # cible
    return train_test_split(x, y, test_size=0.2, random_state=1)


def train_model(x_train, y_train):
    """Entraîne le modèle."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return rf


def evaluate_model(model, x_test, y_test):
    """Évalue les performances."""
    y_pred = model.predict(x_test)
    print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))


def save_model(model, filename="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, filename)
    print("Modèle sauvegardé dans", filename)


def load_model(filename="classifier.joblib"):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)
