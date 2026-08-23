######################################################
# Importation des différentes librairies utiles pour le notebook


######################################################
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from code_python.recherche_classifieur.meilleur_param import get_best_params
from code_python.recherche_classifieur.data_projet import get_dico_classifieur
from code_python.recherche_classifieur.mise_en_production import save_model
from code_python.recherche_classifieur.get_df import get_data_from_csv, split_train_test
from code_python.recherche_classifieur.visualisation import plot_class_distribution, plot_confusion_matrix, boite_a_moustache



######################################################
# Recherche du meilleur classifieur 
######################################################
def get_scores_classifieurs(x_train, y_train, x_test, y_test, dico_classifieur) -> dict:
    # Utilisation de plusieurs Classifieurs sur le même jeu de données (KNN, DT, GAUSSIAN , SVC, RF)

    cv = KFold(n_splits=10, shuffle=True, random_state=0)

    results = {}

    for name, clf in dico_classifieur.items():
        scores = cross_val_score(clf, x_train, y_train, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"KFold sur {name} : accuracy moyenne : {scores.mean()} écart-type : {scores.std()}")

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Matrice de confusion\n{conf_matrix}")
        
        plot_confusion_matrix(conf_matrix, title="Matrice de confusion", labels=["Not science related", "Science related"], cmap="Blues")


        # classification report
        class_report = classification_report(y_test, y_pred, zero_division=0)
        print(f"Classification report\n{class_report}")

    return results

df = get_data_from_csv()    # récupération des données et stockage dans un DataFrame
X, y, x_train, y_train, x_test, y_test = split_train_test(df)   # Séparation des données entre données d'entrainement, résultat à prédire, et données de test
plot_class_distribution(y, class_names=['Not science related', 'Science related'])

dico_classifieur = get_dico_classifieur(y_train=y_train, use_class_weight=False)    # je suis pas hyper sur de ceci, parce que pour GNB prend la proportion des classes de y_train en entier, alors qu'après on teste des échantillons plus petits
dico_classifieur = get_scores_classifieurs(x_train, y_train, x_test, y_test, dico_classifieur)    # Evaluation de chaque classifieur en attribuant un score

boite_a_moustache(dico_classifieur)    # Visualisation des résultats obtenus pour chaque classifieur

name_selected = input("Quel classifieur choisissez-vous\n:")
found = False
while not found:
    if name_selected not in dico_classifieur.keys():
        print(f"Classifieurs possibles : {dico_classifieur.keys()}")
        name_selected = input("Nom incorrect : Quel classifieur choisissez-vous\n:")
    else:
        found = True


# exit()  # commenter quand on cherchera les meilleurs params

plt.close('all')    # vide la mémoire
result = get_best_params(name_selected, x_train, y_train, x_test, y_test)

save_model(df, name_selected, result)

