# IL FAUT VERIFIER QUE CETTE LISTE EST EXHAUSTIVE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import optuna
from code_python.recherche_classifieur.data_projet import create_model, get_param
from code_python.recherche_classifieur.visualisation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report


def get_best_params(classifier_name, x_train, y_train, x_test, y_test):
    # appel de la fonction GridSearchCV
    # get_best_params_GridSearchCV(classifier_name, x_train, y_train)
    # Adri2 : GridSearchCV est TRES long et fonctionne par combinaison "aveugle" de tous les hyperparam. Optuna est plus pertinent et plus rapide. On se passera de GridSearchCV pour les tests
    # appel de la fonction Optuna
    best_modele_best_param = get_best_params_optuna(classifier_name, x_train, y_train, x_test, y_test)

    return best_modele_best_param

def get_best_params_GridSearchCV(classifier_name, x_train, y_train):  
    # Affichage du meilleur score + paramètres associés 
    print("--------------------------- GridSearchCV ---------------------------\n\n")

    modele = create_model(classifier_name)
    print("modele créé")

    params_config = get_param(classifier_name)
    if params_config is None:
        print("pas de params")
        raise ValueError(f"Config pour {classifier_name} non trouvée.")
    
    
    search = GridSearchCV(modele, params_config, cv=5)
    print("GridSearchCV créé")
    search.fit(x_train, y_train)
    
    print(f"Meilleur score : {search.best_score_}")
    print(f"Meilleurs paramètres : {search.best_params_}")
    print(f"Meilleur estimateur : {search.best_estimator_}")
    return search.best_params_

# Utilisation de Optuna pour automatiser la recherhe des meilleurs hyperparamètres 
def objective(trial, classifier_name, X, y):
    """Fonction objectif pour Optuna qui évalue un classifieur donné avec des hyperparamètres suggérés."""
        
    # Récupérer la config du classifieur
    params_config = get_param(classifier_name)
        
    if params_config is None:
        raise ValueError(f"Config pour {classifier_name} non trouvée.")
    
    # Demander à Optuna de choisir une valeur dans chaque liste
    chosen_params = {}
    for param_name, values in params_config.items():
        chosen_params[param_name] = trial.suggest_categorical(param_name, values)


    # Instancier le bon modèle avec les paramètres choisis
    model = create_model(classifier_name, chosen_params)

    # Évaluation
    score = cross_val_score(model, X, y, cv=5, n_jobs=-1).mean()
    return score

def get_best_params_optuna(classifier_name, X_train, y_train, X_test, y_test):
    """ Fonction pour lancer l'optimisation avec Optuna et évaluer le meilleur modèle sur le set test."""
    print("--------------------------- Optuna ---------------------------\n\n")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, classifier_name, X_train, y_train), n_trials=50)

    print(f"Meilleur score pour {classifier_name} :", round(study.best_value, 3))
    print(f"Meilleurs paramètres pour {classifier_name} :", study.best_params)

    # reconstruction du meilleur modèle pour prédiction finale
    best_params = study.best_params

    best_model = create_model(classifier_name, best_params)   

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Matrice de confusion\n{conf_matrix}")
    # classification report
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Classification report\n{class_report}")
    
    plot_confusion_matrix(conf_matrix, title="Matrice de confusion", labels=["Not science related", "Science related"], cmap="Blues")

    print(f"Accuracy  sur le jeu de test : {round(accuracy_score(y_test, y_pred),3)}")
    return best_params


