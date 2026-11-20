import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_data(path="Churn_Modelling.csv"):
    """Charge et prétraite les données.

    Args:
        path (str): chemin du fichier CSV.

    Returns:
        tuple: x_train, x_test, y_train, y_test.
    """
    # Vérification de l'entrée
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    requises = ["Gender", "Surname", "Geography", "Exited", "RowNumber", "CustomerId"]
    manquantes = [c for c in requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=["Surname", "Geography"])
    x = df.drop(columns=["Exited", "RowNumber", "CustomerId"])
    y = df["Exited"]
    return train_test_split(x, y, test_size=0.2, random_state=1)


def train_model(x_train, y_train):
    """Entraîne le modèle.

    Args:
        x_train: variables d'entraînement.
        y_train: cible d'entraînement.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    if len(x_train) == 0 or len(x_train) != len(y_train):
        raise ValueError("x_train et y_train doivent avoir la même taille (non vide)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return rf


def evaluate_model(model, x_test, y_test):
    """Évalue les performances et les affiche.

    Args:
        model: modèle entraîné (avec la méthode predict).
        x_test: variables de test.
        y_test: cible de test.

    Returns:
        float: accuracy en pourcentage.
    """
    if not hasattr(model, "predict"):
        raise TypeError("Le modèle n'a pas de méthode predict")
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("Accuracy :", acc)
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    return acc


def save_model(model, filename="classifier.joblib"):
    """Sauvegarde le modèle avec joblib.

    Args:
        model: modèle entraîné.
        filename (str): nom du fichier de sauvegarde.
    """
    if not hasattr(model, "predict"):
        raise TypeError("Impossible de sauvegarder : modèle invalide")
    joblib.dump(model, filename)
    print("Modèle sauvegardé dans", filename)


def load_model(filename="classifier.joblib"):
    """Charge un modèle sauvegardé.

    Args:
        filename (str): nom du fichier joblib.

    Returns:
        modèle chargé.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Modèle introuvable : {filename} (lance d'abord --train)")
    return joblib.load(filename)
