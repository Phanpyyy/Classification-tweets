import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
colonne_y = 'science_related'

######################################################
# Chargement du jeu de données
######################################################
def get_data_from_csv():
    ROOT = Path().resolve() # chemin du fichier actuel
    name = input("nom de la matrice : ") + ".csv"
    csv_path = ROOT / "src" / "matrices" / name


    # Charger le dataset (séparateur: point-virgule)
    df = pd.read_csv(csv_path, sep=';')

    print("------ Chargement du dataset depuis :", csv_path)
    print(df.head())
    print(df.shape)
    return df

def get_pipeline(path_joblib=None):
    ROOT = Path().resolve() / "src"

    if path_joblib is None:
        name = input("nom de la pipeline : ")+".joblib"
        pipeline_path = ROOT / "pipelines" / name
    else:
        pipeline_path = path_joblib
    pipe = joblib.load(pipeline_path)

    return pipe
######################################################
# Séparation des données d'apprentissage et de classification + entrainement vs test
######################################################
def split_train_test(df) -> tuple:
    y = df[colonne_y]
    X = df.drop(columns=['scientific_context','scientific_reference','scientific_claim',colonne_y,'tweet_id'], errors='ignore')

    print(X.shape)
    print(y.shape)

    # Séparation du jeu de données (70/30 )
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    
    print(f"format du jeu d'apprentissage : {x_train.shape}")
    print(f"format du jeu de test : {x_test.shape}")
    return (X, y, x_train, y_train, x_test, y_test)

def split_x_y(df, target=colonne_y) -> tuple:
    y = df[target]
        
    cols_to_drop = [
        "science_related",
        "scientific_claim",
        "scientific_reference",
        "scientific_context",
        "tweet_id",
    ]

    if target not in cols_to_drop:
        cols_to_drop.append(target)

    # Au cas où dict from keys pour pas avoir de double
    X = df.drop(columns=list(dict.fromkeys(cols_to_drop)), errors="ignore")

    return X, y

