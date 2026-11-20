import argparse
from customer_churn import prepare_data, train_model, evaluate_model, save_model, load_model

# Sans argument, main.py teste tout
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Entraîner et sauvegarder")
parser.add_argument("--evaluate", action="store_true", help="Charger et évaluer")
args = parser.parse_args()
tout = not (args.train or args.evaluate)

# 1) Préparation des données
print("1) prepare_data")
x_train, x_test, y_train, y_test = prepare_data()
print("   Données prêtes :", x_train.shape, x_test.shape)

# 2) Entraînement + sauvegarde
if args.train or tout:
    print("2) train_model + save_model")
    model = train_model(x_train, y_train)
    save_model(model)

# 3) Chargement + évaluation
if args.evaluate or tout:
    print("3) load_model + evaluate_model")
    model = load_model()
    evaluate_model(model, x_test, y_test)
