import joblib
from sklearn.pipeline import Pipeline
from code_python.recherche_classifieur.data_projet import create_model
from code_python.recherche_classifieur.get_df import split_train_test, split_x_y
import os
from pathlib import Path

def save_model(df, classifier_name, best_params):
    print("###################################\n# SAUVEGARDE\n###################################")
    ROOT = Path().resolve() # chemin du fichier actuel
    name = input("nom de la pipeline : ")+".joblib"
    pipeline_path = ROOT / "src" / "pipelines" / name

    df2 = df.sample(n=4)
    df = df.drop(df2.index)

    df2.head()

    X, y, X_train, y_train, X_test, y_test = split_train_test(df)   # Séparation des données entre données d'entrainement, résultat à prédire, et données de test

    # Classifieur final (à adapter si besoin selon vos propres résultats)
    final_clf = create_model(classifier_name, best_params)
    print("model créé")

    # Pipeline minimal contenant le classifieur
    pipeline_final = Pipeline([
        ("clf", final_clf)
    ])
    print("pipeline créée")
    print(best_params)

    # Entraînement du pipeline sur toutes les données disponibles
    pipeline_final.fit(X, y)
    print("modèle entrainée")

    # Sauvegarde du pipeline
    joblib.dump(pipeline_final, pipeline_path)
    print("pipeline saved")
    print(f"Le modèle sera sauvegardé ici : {os.getcwd()}")


    # test sur les 4 données mises de cotées
    # Rechargement du modèle
    loaded_model = joblib.load(pipeline_path)
    print("modèle chargée")

    X, y = split_x_y(df2)   # Séparation des données entre données d'entrainement, résultat à prédire, et données de test


    # Prédictions
    y_pred = loaded_model.predict(X)
    print("prediction réalisée")

    # Affichage clair et simple
    resultats = df2.copy()
    resultats['prediction'] = y_pred
    resultats['reel'] = y
    
    print("Résultat du pipeline sur les données isolées :")
    print(resultats[['reel', 'prediction']])