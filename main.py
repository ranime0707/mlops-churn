import argparse
from customer_churn import prepare_data, train_model, evaluate_model, save_model, load_model


def main():
    """Exécute les étapes du pipeline selon les arguments CLI."""
    parser = argparse.ArgumentParser(description="Pipeline ML : Customer Churn")
    parser.add_argument("--prepare", action="store_true", help="Tester prepare_data")
    parser.add_argument("--train", action="store_true", help="Entraîner et sauvegarder le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Charger et évaluer le modèle")
    args = parser.parse_args()

    # Sans argument, on exécute tout
    tout = not (args.prepare or args.train or args.evaluate)

    # 1) Préparation des données
    x_train, x_test, y_train, y_test = prepare_data()
    if args.prepare or tout:
        print("1) prepare_data OK :", x_train.shape, x_test.shape)

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


if __name__ == "__main__":
    main()
