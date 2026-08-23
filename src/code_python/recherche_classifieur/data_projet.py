from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_priors(y_train) -> list:
    # 1. Calcul des poids (ex: [0.52, 10.4])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    # 2. Normalisation pour que la somme soit égale à 1
    # On divise chaque poids par la somme totale des poids
    priors_normalized = weights / weights.sum()

    return priors_normalized


def get_dico_classifieur(y_train, use_class_weight:bool=False) -> dict:
    result = {}
    # à décommenter uniquement pour faire les tests plus vite
    return{"SVC":SVC()}


    if use_class_weight:
        result = {"KNN":KNeighborsClassifier(weights="distance"),   # n"a pas de class_weight
            "DT":DecisionTreeClassifier(class_weight="balanced"),
            "GNB":GaussianNB(priors = get_priors(y_train)), # n"a pas de class_weight
            "SVC":SVC(class_weight="balanced"),
            "RF":RandomForestClassifier(class_weight="balanced"),
            "LR":LogisticRegression(class_weight="balanced")}
    
    else :
        result = {"KNN":KNeighborsClassifier(),   # n"a pas de class_weight
                "DT":DecisionTreeClassifier(),
                "GNB":GaussianNB(), # n"a pas de class_weight
                "SVC":SVC(),
                "RF":RandomForestClassifier(),
                "LR":LogisticRegression()}
        
    return result

def create_model(classifier_name, params={}):
    """Fonction utilitaire pour centraliser la création des modèles."""
    if classifier_name == "RF":
        return RandomForestClassifier(**params, random_state=42)
    elif classifier_name == "DT":
        return DecisionTreeClassifier(**params, random_state=42)
    elif classifier_name == "SVC":
        return SVC(**params)
    elif classifier_name == "KNN":
        return KNeighborsClassifier(**params)
    elif classifier_name == "GNB":
        return GaussianNB(**params)
    elif classifier_name == "LR":
        return LogisticRegression(**params)
    raise ValueError(f"Classifieur {classifier_name} non supporté.")

def get_param(classifier_name):
    RF_param = {
        "n_estimators": [4, 6, 9],
        "max_features": ["log2", "sqrt"],
        "criterion": ["entropy", "gini"],
        "max_depth": [2, 3, 5, 10],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 5, 8]   
    }
    DT_param = {
        "max_depth": [1,2,3,4,5,6,7,8,9,10],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10]
    }
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    svc_param = {
        "C": [0.001, 0.01, 0.1, 0.4, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 3, 6, 6.5, 6.8, 7, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 8.1, 8.2, 8.3, 8.4, 8.5, 9, 10, 11, 12, 13, 15, 20],
        "gamma": [0.00005, 0.00008, 0.001, 0.002, 0.005, 0.01, 0.1, 1, 2, 5, 7, 10, "scale"],
        "kernel": ["linear", "poly", "rbf"],
        "degree":[2,3,4,5],     # uniquement pour le kernel poly
        "class_weight": ["balanced", None]
    }

    #     svc_param = {
    #    "C": [0.001, 0.01, 0.1, 0.4, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 10],
    #    "gamma": [0.00005, 0.00008, 0.001, 0.002, 0.005, 0.01, 0.1, 1, 2, 5, 7, 10, "scale"],
    #    "kernel": ["linear", "poly", "rbf"],
    #    "degree":[2,3,4,5],     # uniquement pour le kernel poly
    #    "class_weight": ["balanced", None]
    #}
    

    KNN_param = {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"]
        }
    GNB_param= {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6] # Le seul paramètre réel de GNB
        }
    nb_param = {
        "alpha" : [0.1 , 0.5, 1.0]
    }
    logistic_param = {
        "C" : [0.1, 1, 10]
    }

    hyper_param = {
        "KNN": KNN_param,
        "DT": DT_param,
        "GNB": GNB_param,
        "SVC": svc_param,
        "RF": RF_param,
        "LR" : logistic_param
        #"NB" : nb_param
    }

    return hyper_param[classifier_name]
