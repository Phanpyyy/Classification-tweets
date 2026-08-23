from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
import time

from code_python.recherche_classifieur.get_df import get_data_from_csv, get_pipeline, split_x_y

# Le but est de récupérer les features (données qui contribuent à la classification) de nos modèles
# Pour ça on va récupéré le df et le x y (grace à la target)


def ranked_features_from_model(model, X_train,y_train, X_val, y_val):
    # Cas 1: modèles avec feature_importances_ (RF, DT...)
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_

    # Cas 2: modèles linéaires avec coef_ (LR, SVC kernel='linear')
    elif hasattr(model, "coef_"):
        coef = model.coef_
        scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    # Cas 3: fallback universel (SVC rbf, KNN, GNB...)
    else:
        print("SVC non linéaire détecté : Utilisation de l'ANOVA (f_classif) pour estimer l'importance des variables instantanément.")
        
        # f_classif calcule un score statistique de pertinence pour chaque colonne
        f_values, p_values = f_classif(X_train, y_train)
        
        # On remplace les potentiels NaN par 0
        scores = np.nan_to_num(f_values, 0)

    ranking = (
        pd.DataFrame({"feature": X_train.columns, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return ranking


def load_pipeline_and_get_estimator(path_joblib):
    start = time.perf_counter()
    print(f"Chargement du modèle: {path_joblib}")
    pipe = joblib.load(path_joblib)
    print(f"Modèle chargé en {time.perf_counter() - start:.2f}s")

    # Pour récupérer le pipeline est Pipeline([("clf", final_clf)])
    if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
        return pipe.named_steps["clf"]

    return pipe


def feature_ranking_for_task(df, target_col, model_path):
    X, y = split_x_y(df, target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Target: {target_col} | X_train: {X_train.shape} | X_val: {X_val.shape}")

    model = load_pipeline_and_get_estimator(model_path)
    print(f"Modèle utilisé: {type(model).__name__}")

    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Fit terminé en {time.perf_counter() - start_fit:.2f}s")

    start_rank = time.perf_counter()
    ranking = ranked_features_from_model(model, X_train, y_train, X_val, y_val)
    print(f"Ranking terminé en {time.perf_counter() - start_rank:.2f}s")
    return ranking


# fonction de test pour print et récupérer le modèle de classification
def print_test_pipe_classe():
    df = get_data_from_csv()
    pipe = get_pipeline()
    if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
        pipe = pipe.named_steps["clf"]

    print("Type Python:", type(pipe))
    print("Nom classifieur:", type(pipe).__name__)
    print("Classe complète:", pipe.__class__)


if __name__ == "__main__":
    df = get_data_from_csv()

    tasks = ["science_related"]  # ajoute "scientific_claim", "scientific_reference", etc.
    model_path = str(Path(__file__).resolve().parent.parent.parent / "pipelines" / "SMOTE.joblib")

    print(f"Dataset chargé: {df.shape}")
    print(f"Modèle chargé depuis: {model_path}")

    for t in tasks:
        r = feature_ranking_for_task(df, t, model_path)
        print(f"\nTop 20 pour {t}")
        print(r.head(20).to_string(index=False))

        r.to_csv(f"top_features_{t}.csv", index=False)
        print(f"Fichier généré: top_features_{t}.csv")