#############################################################################################################################
### Imports
#############################################################################################################################
from wordcloud import WordCloud
import warnings # Librairie pour gérer les avertissements
warnings.filterwarnings("ignore", category=FutureWarning)
# Librairies générales
import pandas as pd # Manipulation de données sous forme de DataFrame
import numpy as np # Calcul numérique et manipulation de tableaux

# Librairies pour l'affichage
import matplotlib.pyplot as plt # Tracé de graphiques
import seaborn as sns # Visualisation avancée de données
# Scikit-learn : chargement de jeux de données
from sklearn.datasets import load_digits # Jeu de données Digits
from sklearn.datasets import load_iris # Jeu de données IRIS
from sklearn.datasets import load_breast_cancer # Données sur le cancer du sein
from sklearn.datasets import make_classification # Génération de jeux de données
from sklearn.datasets import make_multilabel_classification # Génération
# Scikit-learn : prétraitement des données
from sklearn.preprocessing import StandardScaler # Standardisation des données
from sklearn.preprocessing import label_binarize # Binarisation des labels pour
# courbes ROC multiclasse
# Scikit-learn : modèles de classification
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn : réduction de dimension
from sklearn.decomposition import PCA # Analyse en composantes principales(ACP)
from typing import Dict
import numpy as np
from numpy.typing import NDArray


#############################################################################################################################
### Fonctions
#############################################################################################################################
""""
Notebook_EvaluationModeles.pdf

citation sujet ML : 
N oubliez pas de bien évaluer vos modèles. L accuracy n est pas suffisante. Pensez à la matrice
de confusion, au rappel, à la précision, à la F-mesure

matric conf : ok
rappel : ok
precision : ok
f mesure : ok

je ne comprends pas à quoi sert la courbe roc page 21 - plot_roc_curve
je crois que c'est pour l'analyse under/over fitting. Il faudra effectivement ajouter cette analyse
"""

def boite_a_moustache(results: Dict[str, NDArray[np.float64]]) -> None:
    fig = plt.figure()
    fig.suptitle("Comparaison des algorithmes")
    ax = fig.add_subplot(111)
    
    plt.boxplot(list(results.values()))
    
    # Pour les labels, c'est pareil : on convertit en liste
    ax.set_xticklabels(list(results.keys()))
    
    plt.show()

def plot_class_distribution(y, class_names=None):
    """
    Affiche la répartition des classes et génère un histogramme.
    Paramètres :
    - y : array-like
    Labels des classes (int ou str).
    - class_names : list, optional
    Noms des classes pour l'axe des x (default: None,
    utilise les valeurs uniques de y).
    """
    # Calculer la répartition des classes
    unique, counts = np.unique(y, return_counts=True)
    print("Répartition des classes :")
    for cls, count in zip(unique, counts):
        print(f"Classe {cls} : {count} échantillons")
    
    # Si aucun nom de classe n'est fourni, utiliser les valeurs uniques de y
    if class_names is None:
        class_names = [f"Classe {cls}" for cls in unique]
    
    # Histogramme de la répartition des classes
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, color=['blue', 'orange'], alpha=0.7)
    plt.xticks(unique, class_names)
    plt.ylabel("Nombre d'échantillons")
    plt.title("Répartition des classes")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_confusion_matrix(conf_matrix, title="Matrice de confusion", labels=None, cmap="Blues"):
    """
    Trace une matrice de confusion sous forme de heatmap.
    Paramètres :
    - conf_matrix : array-like (2D)
    Matrice de confusion .
    - title : str, optional
    Titre du graphique (default : "Matrice de confusion").
    - labels : list, optional
    Liste des noms des classes pour les axes
    (default : None, utilise les indices numériques).
    - cmap : str, optional
    Colormap pour la heatmap (default : "Blues").
    """
    # Générer des labels par défaut si non spécifiés
    if labels is None:
        labels = [f"Classe {i}" for i in range(conf_matrix.shape[0])]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap,
    xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel("Classe Réelle")
    plt.xlabel("Classe Prédite")
    plt.tight_layout()
    plt.show()

    
def plot_roc_curve(fpr, tpr, auc_score):
    """
    Trace la courbe ROC (Receiver Operating Characteristic)
    pour évaluer la performance d'un modèle de classification.
    Paramètres :
    - fpr : array-like
    Taux de faux positifs (False Positive Rate) pour différents seuils.
    - tpr : array-like
    Taux de vrais positifs (True Positive Rate) pour différents seuils.
    - auc_score : float
    Aire sous la courbe ROC (Area Under the Curve, AUC),
    une mesure globale de performance.
    Fonctionnalités :
    - Trace la courbe ROC avec la courbe réelle en bleu.
    - Ajoute une diagonale en rouge indiquant la ligne de référence pour
    un modèle aléatoire.
    - Affiche l'AUC dans la légende.
    ```
    """
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.show()


def plot_pca_2d_3d(X, y, title_2d="ACP 2D", title_3d="ACP 3D"):
    """
    Affiche une ACP 2D et 3D pour visualiser la répartition des documents.
    X : matrice de features (numpy array ou matrice creuse)
    y : labels (array-like)
    """
    # Conversion si matrice creuse
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    y = np.array(y)
    # ACP 2D
    pca_2d = PCA(n_components=2, random_state=0)
    X_pca_2d = pca_2d.fit_transform(X_dense)
    # ACP 3D
    pca_3d = PCA(n_components=3, random_state=0)
    X_pca_3d = pca_3d.fit_transform(X_dense)
    # Figure
    fig = plt.figure(figsize=(12, 5))
    # ----- Plot 2D -----
    ax1 = fig.add_subplot(1, 2, 1)
    idx0 = (y == 0)
    idx1 = (y == 1)
    ax1.scatter(X_pca_2d[idx0, 0], X_pca_2d[idx0, 1], alpha=0.6, label="Not science related")
    ax1.scatter(X_pca_2d[idx1, 0], X_pca_2d[idx1, 1], alpha=0.6, label="Science related")
    ax1.set_title(title_2d)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    # ----- Plot 3D -----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(X_pca_3d[idx0, 0], X_pca_3d[idx0, 1], X_pca_3d[idx0, 2], alpha=0.6, label="Not science related")
    ax2.scatter(X_pca_3d[idx1, 0], X_pca_3d[idx1, 1], X_pca_3d[idx1, 2], alpha=0.6, label="Science related")
    ax2.set_title(title_3d)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()

def word_cloud(df):
    # 1. On exclut les colonnes qui ne sont pas des mots
    cols_to_drop = ['scientific_context', 'scientific_reference', 'scientific_claim', 'science_related', 'tweet_id']
    df_words = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2. On additionne les occurrences et on remplace les espaces par des "_"
    word_counts_brut = df_words.sum().to_dict()
    
    # Création d'un nouveau dictionnaire avec les clés modifiées
    word_counts = {mot.replace(' ', '_'): frequence for mot, frequence in word_counts_brut.items()}
    
    # 3. Création du word cloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        prefer_horizontal=0.7,      
        relative_scaling=0.5,       
        max_words=200,              
        min_font_size=10,           
        max_font_size=150           
    )
    wc.generate_from_frequencies(word_counts) # Génération basée sur le dictionnaire de fréquences
    
    # 4. Affichage
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    # (Optionnel) Sauvegarde de l'image
    wc.to_file("wordcloud.png")
    # 1. On exclut les colonnes qui ne sont pas des mots
    cols_to_drop = ['scientific_context', 'scientific_reference', 'scientific_claim', 'science_related', 'tweet_id']
    df_words = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2. On additionne les occurrences de chaque colonne/mot sur tout le dataset
    word_counts = df_words.sum().to_dict()
    
    # 3. Création du word cloud
        # 3. Création du word cloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        prefer_horizontal=0.7,      # Autorise 30% des mots à être placés à la verticale (imbrication)
        relative_scaling=0.5,       # Accentue la différence de taille entre les mots fréquents et rares
        max_words=200,              # Limite le nombre de mots pour éviter l'effet "bloc"
        min_font_size=10,           # Taille minimum
        max_font_size=150           # Taille maximum très grande pour les gros mots
    )
    wc.generate_from_frequencies(word_counts) # Génération basée sur le dictionnaire de fréquences
    
    # 4. Affichage
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    # (Optionnel) Sauvegarde de l'image
    wc.to_file("wordcloud.png")

def data_2d_3d_visu(df):

    column_principal = "science_related"

    # Colonnes numériques utilisées pour l'ACP
    cols_for_pca = [
        "scientific_context","scientific_reference","scientific_claim"
    ]

    X_pca = df[cols_for_pca].values

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # ACP sur 3 composantes
    pca = PCA(n_components=3)
    X_pca_3d = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PC1": X_pca_3d[:, 0],
        "PC2": X_pca_3d[:, 1],
        "PC3": X_pca_3d[:, 2],
        "science_related": df[column_principal]
    })

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=(
            "Projection 2D (PC1, PC2)",
            "Projection 3D (PC1, PC2, PC3)"
        )
    )

    # Scatter 2D à gauche
    fig.add_trace(
        go.Scatter(
            x=df_pca["PC1"],
            y=df_pca["PC2"],
            mode="markers",
            marker=dict(
                color=df_pca[column_principal],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=column_principal)
            ),
            name="2D"
        ),
        row=1,
        col=1
    )

    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)

    # Scatter 3D à droite
    fig.add_trace(
        go.Scatter3d(
            x=df_pca["PC1"],
            y=df_pca["PC2"],
            z=df_pca["PC3"],
            mode="markers",
            marker=dict(
                size=4,
                color=df_pca[column_principal],
                colorscale="Viridis",
                showscale=False
            ),
            name="3D"
        ),
        row=1,
        col=2
    )

    fig.update_scenes(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
        row=1,
        col=2
    )

    fig.update_layout(
        title="ACP des données de Tweet (2D et 3D)",
        width=900,
        height=450
    )

    fig.show()

#df = pd.read_csv('../scitweets_export.tsv', sep='\t')
#plot_pca_2d_3d(df,df["science_related"])

"""
###############################################
#### Interprétation des résultats 
###############################################
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

print(f"Matrice de confusion\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification report\n{classification_report(y_test, y_pred, zero_division=0)}")


# moyenne et écart type en tableau matplotlib
fig = plt.figure()
fig.suptitle("Comparaison des algorithmes")
ax = fig.add_subplot(111)
plt.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.show()
"""