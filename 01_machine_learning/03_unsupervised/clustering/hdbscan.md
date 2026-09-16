# HDBSCAN, le clustering hiérarchique fondé sur la densité

> **Résumé en une phrase** : HDBSCAN généralise DBSCAN en faisant varier implicitement le seuil de densité, construit une hiérarchie de groupes puis sélectionne les clusters les plus stables, tout en identifiant les points atypiques et les appartenances incertaines.

## Table des Matières

0. [Avant de commencer](#0-avant-de-commencer)
1. [Introduction, le problème](#1-introduction-le-problème)
2. [Rappels, densité et DBSCAN](#2-rappels-densité-et-dbscan)
3. [De DBSCAN à HDBSCAN](#3-de-dbscan-à-hdbscan)
4. [Le pipeline algorithmique complet](#4-le-pipeline-algorithmique-complet)
5. [Comprendre la sortie du modèle](#5-comprendre-la-sortie-du-modèle)
6. [Paramètres et stratégie de réglage](#6-paramètres-et-stratégie-de-réglage)
7. [Implémentation Python](#7-implémentation-python)
8. [Validation et interprétation](#8-validation-et-interprétation)
9. [Cas d'usage guidé, segmenter des clients](#9-cas-dusage-guidé-segmenter-des-clients)
10. [Guide de choix et comparaisons](#10-guide-de-choix-et-comparaisons)
11. [Pièges Courants](#11-pièges-courants)
12. [Exercices d'auto-évaluation](#12-exercices-dauto-évaluation)
13. [Sources et Références](#13-sources-et-références)
14. [Liens connexes](#14-liens-connexes)

---

## 0. Avant de commencer

### Prérequis

| Prérequis | Niveau attendu |
|-----------|----------------|
| Distances entre observations | Savoir calculer ou interpréter une distance euclidienne et comprendre qu'une métrique dépend du type de données |
| Statistiques descriptives | Comprendre moyenne, quantiles, dispersion et valeurs atypiques |
| Python scientifique | Savoir utiliser NumPy, Pandas et les bases de scikit-learn |
| Apprentissage non supervisé | Savoir distinguer clustering, réduction de dimensionnalité et détection d'anomalies |
| DBSCAN | Une connaissance préalable est utile, mais les notions nécessaires sont rappelées dans ce cours |

### Objectifs d'apprentissage

| Niveau Bloom | Objectif |
|-------------|----------|
| **Comprendre** | Expliquer pourquoi un algorithme fondé sur la densité est utile lorsque les groupes ne sont ni sphériques ni de taille comparable |
| **Comprendre** | Décrire le rôle de la distance de cœur, de la distance de portée mutuelle, de l'arbre couvrant minimal et de la stabilité |
| **Appliquer** | Entraîner HDBSCAN avec la bibliothèque `hdbscan` ou avec `sklearn.cluster.HDBSCAN` et interpréter ses sorties principales |
| **Analyser** | Diagnostiquer l'effet de `min_cluster_size`, `min_samples`, de la métrique et du choix `eom` ou `leaf` |
| **Évaluer** | Choisir HDBSCAN plutôt que K-means, DBSCAN ou un clustering hiérarchique selon la forme, la densité et la présence d'outliers |

### TL;DR en 3 phrases

> HDBSCAN cherche des régions de forte densité sans imposer un nombre de clusters et sans utiliser un unique seuil de distance pour tous les groupes. Il transforme une hiérarchie de voisinages en un arbre condensé, puis retient les groupes qui restent cohérents sur une large plage de niveaux de densité. En pratique, on commence généralement par choisir `min_cluster_size`, on vérifie la métrique et la mise à l'échelle, puis on examine les labels, les probabilités d'appartenance et la proportion de bruit.

> **Note de vocabulaire** : la littérature distingue souvent **HDBSCAN***, la formulation théorique fondée sur DBSCAN* et une hiérarchie de densité, de la bibliothèque Python appelée `hdbscan`, qui implémente cette famille d'algorithmes avec des extensions pratiques.

---

## 1. Introduction, le problème

### 1.1 Analogie concrète, repérer des quartiers dans une ville

Imaginez une ville observée depuis le ciel. Les habitants ne sont pas répartis en cercles parfaits : certains quartiers sont allongés, d'autres courbés, certains sont très peuplés et d'autres plus résidentiels. Quelques personnes se trouvent entre deux quartiers ou dans des zones isolées.

Une méthode utile doit donc répondre à quatre questions :

1. Où les observations sont-elles suffisamment concentrées pour former un groupe ?
2. Les groupes peuvent-ils avoir des formes quelconques ?
3. Peut-on accepter que certains points ne soient rattachés à aucun groupe ?
4. Que faire si un quartier dense et un quartier peu dense coexistent ?

K-means répond mal à ces questions, car il cherche des groupes autour de centroïdes et demande le nombre de groupes à l'avance. DBSCAN répond mieux aux formes arbitraires et aux points de bruit, mais son seuil global `eps` peut être trop strict pour un groupe peu dense et trop permissif pour un groupe très dense. HDBSCAN conserve l'idée de densité, mais observe l'organisation des points sur une gamme de seuils au lieu de miser sur une seule valeur.

### 1.2 Un exemple géométrique minimal

Considérons quatre zones de données :

- un amas très dense autour de `(0, 0)` ;
- un amas moins dense autour de `(5, 5)` ;
- une forme en croissant ;
- quelques points isolés.

Un bon algorithme devrait produire quelque chose de proche de ceci :

```
Y
^                 ·   ·       bruit
|          ████
|       ████████              groupe peu dense
|             ███
|   forme en croissant
|  ◜████████
| ◜██████████       groupe très dense
+------------------------------------> X
```

Le but n'est pas de rendre la projection jolie. Le but est de définir une règle de voisinage qui respecte la géométrie des données et de quantifier l'incertitude lorsque cette règle ne permet pas une décision nette.

### 1.3 Formalisation du problème

On observe un ensemble de points :

$$X = \{x_1, x_2, \ldots, x_n\}, \qquad x_i \in \mathbb{R}^d$$

Ici, `n` est le nombre d'observations et `d` le nombre de variables. HDBSCAN cherche une partition partielle :

$$\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$$

avec trois possibilités pour chaque observation :

- appartenir à un cluster `C_k` ;
- être classée comme bruit, généralement avec le label `-1` ;
- avoir une appartenance faible ou ambiguë, même si elle reçoit un label de cluster.

La particularité importante est que `K` n'est pas imposé par l'utilisateur. Il résulte de la structure de densité retenue par l'algorithme et de ses paramètres de granularité.

> 📌 **Points clés, Section 1**
> - HDBSCAN vise des groupes de forme arbitraire, avec des densités potentiellement différentes.
> - Le bruit n'est pas une erreur de l'algorithme, c'est une sortie possible et souvent utile.
> - L'algorithme ne demande pas directement le nombre de clusters.

---

## 2. Rappels, densité et DBSCAN

### 2.1 Intuition, la densité comme nombre de voisins proches

Pour savoir si un point est situé dans une zone dense, on peut compter combien de voisins se trouvent dans un rayon donné. Un point entouré de nombreuses observations est probablement au cœur d'un groupe. Un point isolé est plutôt du bruit ou une observation frontière.

DBSCAN formalise cette intuition avec deux paramètres :

- `eps`, le rayon de voisinage ;
- `min_samples`, le nombre minimal de points nécessaires pour considérer une zone comme dense.

### 2.2 Les trois rôles d'un point dans DBSCAN

Pour un point `x`, on définit son voisinage :

$`N_{\varepsilon}(x) = \{y \in X \mid d(x,y) \leq \varepsilon\}`$

`d(x, y)` est la distance choisie et `eps` est le rayon du voisinage.

Un point est :

| Type | Condition intuitive | Conséquence |
|------|---------------------|-------------|
| **Core** | Son voisinage contient au moins `min_samples` points | Il peut étendre un cluster |
| **Border** | Il est proche d'un core, mais n'est pas suffisamment dense lui-même | Il peut être rattaché au cluster |
| **Noise** | Il n'est pas relié à une région suffisamment dense | Il reçoit généralement le label `-1` |

Dans une notation simplifiée, un point est core lorsque :

$`|N_{\varepsilon}(x)| \geq \text{min\_samples}`$

Le seuil `eps` est ici le point faible de DBSCAN. Si deux clusters ont des densités très différentes, une seule valeur peut être inadaptée aux deux.

### 2.3 Pourquoi un `eps` unique est difficile à choisir

Supposons un cluster A très dense et un cluster B cinq fois moins dense. Une petite valeur de `eps` permet de séparer A proprement mais peut fragmenter B. Une grande valeur relie correctement B mais risque de fusionner A avec d'autres points.

| Choix de `eps` | Cluster dense | Cluster peu dense | Risque |
|----------------|---------------|-------------------|--------|
| Petit | Bien détecté | Fragmenté ou perdu | Trop de bruit |
| Grand | Fusion possible | Mieux détecté | Trop de connexions |
| Intermédiaire | Dépend fortement des données | Dépend fortement des données | Réglage fragile |

HDBSCAN ne supprime pas la notion de densité. Il construit une hiérarchie qui représente les résultats possibles pour différents niveaux de densité, puis sélectionne des groupes persistants.

> 📌 **Points clés, Section 2**
> - DBSCAN définit la densité à partir d'un rayon `eps` et d'un nombre minimal de voisins.
> - Le rôle de `eps` devient problématique quand les clusters ont des densités très différentes.
> - HDBSCAN remplace le choix d'un unique niveau par l'étude d'une hiérarchie de niveaux.

---

## 3. De DBSCAN à HDBSCAN

### 3.1 Intuition, inverser le point de vue avec la lambda densité

Au lieu de demander « quels points sont à moins de `eps` ? », HDBSCAN peut être compris comme une exploration progressive :

1. on commence avec une contrainte de densité faible, donc beaucoup de points sont connectés ;
2. on augmente progressivement la contrainte de densité ;
3. les groupes se séparent lorsque les connexions entre eux disparaissent ;
4. les groupes qui persistent longtemps sont considérés comme plus stables.

On remplace souvent `eps` par une échelle de densité :

$$\lambda = \frac{1}{\varepsilon}$$

Une petite distance `eps` correspond à une grande valeur de `lambda`, donc à une exigence de densité plus forte.

### 3.2 La distance de cœur

Pour chaque point `x`, on regarde la distance vers son `k`-ième voisin :

$$\\mathop{\text{core}}_k(x) = d(x, x_{(k)})$$

où `x_(k)` désigne le `k`-ième plus proche voisin de `x`, et `k` est lié à `min_samples`.

Cette distance mesure la taille de la boule qu'il faut autour de `x` pour obtenir suffisamment de voisins. Elle est grande dans une région peu dense et petite dans une région très dense.

### 3.3 La distance de portée mutuelle

Une distance brute peut relier deux points proches alors que l'un des deux se trouve dans une zone très peu dense. HDBSCAN utilise une distance qui tient compte à la fois de la distance entre les points et de leur densité locale :

$$d_{\mathrm{mreach}-k}(a,b) = \max\left(\\mathop{\text{core}}_k(a),\\mathop{\text{core}}_k(b),d(a,b)\right)$$

Les termes signifient :

- `d(a, b)` : distance directe entre `a` et `b` ;
- `core_k(a)` : distance de cœur de `a` ;
- `core_k(b)` : distance de cœur de `b` ;
- `max` : la connexion est pénalisée par le point le moins dense des deux.

Cette transformation rend la notion de proximité plus robuste aux variations locales de densité.

### 3.4 Le graphe de voisinage et l'arbre couvrant minimal

HDBSCAN n'a pas besoin de conserver toutes les connexions possibles entre les `n` points. Il peut exploiter un graphe de voisinage pondéré par la distance de portée mutuelle, puis calculer un **minimum spanning tree**, ou arbre couvrant minimal.

```
Points et distances de portée mutuelle
        │
        ▼
Graphe pondéré des voisinages
        │
        ▼
Arbre couvrant minimal
        │
        ▼
Suppression progressive des arêtes lourdes
        │
        ▼
Hiérarchie des composantes connexes
```

L'arbre contient suffisamment d'information pour suivre la séparation des composantes lorsque le niveau de densité augmente. Dans une intuition de « ponts », les arêtes lourdes sont les ponts les moins fiables entre régions denses. Les retirer révèle la structure hiérarchique.

### 3.5 De l'arbre à la hiérarchie de clusters

On peut représenter la hiérarchie sous la forme d'un dendrogramme :

```
Faible densité, lambda faible
└── Tous les points
    ├── Groupe A
    │   ├── A1
    │   └── A2
    └── Groupe B
        ├── B1
        └── B2
Forte densité, lambda élevé
```

Cette hiérarchie contient plus de clusters que la sortie finale. HDBSCAN doit donc choisir une coupe ou une sélection de branches pertinente.

> 📌 **Points clés, Section 3**
> - La distance de cœur décrit la densité locale autour d'un point.
> - La distance de portée mutuelle évite de considérer deux régions comme proches uniquement parce que deux points le sont.
> - L'arbre couvrant minimal permet de représenter efficacement les séparations de la hiérarchie.
> - HDBSCAN ne renvoie pas toute la hiérarchie par défaut, il sélectionne des groupes à partir de leur stabilité.

---

## 4. Le pipeline algorithmique complet

### 4.1 Vue d'ensemble

Le fonctionnement peut être résumé ainsi :

```
Données X
  │
  ├── Choix de la métrique
  │
  ├── Estimation des distances de cœur
  │
  ├── Distances de portée mutuelle
  │
  ├── Arbre couvrant minimal
  │
  ├── Hiérarchie des composantes
  │
  ├── Condensation avec min_cluster_size
  │
  ├── Calcul de la stabilité
  │
  └── Sélection EOM ou leaf
          │
          ├── labels_
          ├── probabilities_
          └── outlier_scores_
```

### 4.2 Condenser l'arbre

La hiérarchie brute peut contenir des branches minuscules et très instables. Le paramètre `min_cluster_size` sert à supprimer les branches qui ne contiennent pas assez d'observations pour être considérées comme des clusters autonomes.

Lorsqu'une branche descend sous cette taille minimale, les points qui la composent sont considérés comme quittant le cluster parent à ce niveau, mais la branche n'est pas conservée comme cluster final.

La sortie est appelée **condensed tree**. Elle conserve les séparations importantes tout en éliminant une partie des divisions trop fines.

### 4.3 La stabilité d'un cluster

Un cluster est stable lorsqu'il conserve beaucoup de points sur une large plage de niveaux de densité. Une formulation intuitive de sa stabilité est :

$$S(C) = \sum_{p \in C} \left(\lambda_{\mathrm{death}}(p) - \lambda_{\mathrm{birth}}(C)\right)$$

Cette écriture simplifiée signifie :

- `C` : le cluster étudié ;
- `p` : un point qui appartient à `C` ;
- `lambda_birth(C)` : niveau auquel le cluster apparaît ;
- `lambda_death(p)` : niveau auquel le point quitte cette branche ;
- `S(C)` : quantité de persistance accumulée par le cluster.

La bibliothèque calcule cette notion avec des détails d'arbre et de sélection qui dépassent cette formule pédagogique. Il faut retenir l'idée : un cluster qui existe longtemps et conserve beaucoup de points obtient une stabilité élevée.

### 4.4 Sélection Excess of Mass, ou `eom`

La méthode par défaut est généralement appelée **Excess of Mass**, abrégée `eom`. Elle cherche un ensemble de clusters persistants qui maximise une notion de stabilité sans sélectionner simultanément tous les descendants incompatibles.

L'intérêt de `eom` est d'obtenir une partition souvent compacte et robuste :

- des branches très stables peuvent être conservées ;
- des sous-branches moins utiles peuvent être absorbées par leur parent ;
- le résultat n'est pas nécessairement le plus fin possible.

### 4.5 Sélection `leaf`

La sélection `leaf` prend plutôt les feuilles de l'arbre condensé. Elle produit souvent des clusters plus fins, utiles lorsque l'on souhaite distinguer des sous-populations proches.

| Méthode | Granularité | Usage typique | Risque |
|---------|-------------|---------------|--------|
| `eom` | Modérée, adaptative | Segmentation robuste, exploration générale | Fusionner des sous-groupes utiles |
| `leaf` | Fine | Recherche de sous-structures | Sur-segmentation, nombreux petits clusters |

### 4.6 Bruit et point frontière

Un point peut être considéré comme du bruit s'il n'appartient de manière suffisamment persistante à aucune branche sélectionnée. Cela ne signifie pas que l'observation est fausse. Cela signifie que la structure de densité observée ne justifie pas son rattachement à un cluster.

Cette propriété est importante dans des cas comme :

- transactions atypiques ;
- documents très spécialisés ;
- capteurs défaillants ou situations rares ;
- comportements situés entre plusieurs segments.

> 📌 **Points clés, Section 4**
> - `min_cluster_size` contrôle la taille minimale des groupes que l'on accepte de conserver.
> - La stabilité mesure la persistance d'une branche de la hiérarchie.
> - `eom` cherche des groupes persistants sans forcer une granularité excessive.
> - `leaf` est utile pour explorer les sous-groupes, mais peut produire une segmentation plus fragmentée.

---

## 5. Comprendre la sortie du modèle

### 5.1 `labels_`, une partition avec bruit

Après entraînement, `labels_` contient un entier par observation :

- `0`, `1`, `2`, etc. pour les clusters ;
- `-1` pour le bruit.

Exemple :

```python
# labels_ contient un label par ligne de X
labels = clusterer.labels_

# Le label -1 représente les observations non affectées à un cluster
is_noise = labels == -1

# Le nombre de clusters exclut le bruit
n_clusters = len(set(labels)) - int(is_noise.any())
```

Attention : les numéros de cluster sont des identifiants arbitraires. Le cluster `0` n'est pas « meilleur » que le cluster `1` et les numéros peuvent changer lorsque les paramètres ou l'ordre des données changent.

### 5.2 `probabilities_`, une confiance locale

La bibliothèque `hdbscan` fournit une probabilité d'appartenance comprise entre 0 et 1 pour les points affectés à un cluster. Elle exprime la force de l'appartenance du point à son cluster dans la hiérarchie retenue.

```python
# Probabilité d'appartenance à son cluster attribué
membership_strength = clusterer.probabilities_

# Points faiblement rattachés à leur cluster
uncertain = membership_strength < 0.5
```

Cette valeur ne doit pas être interprétée comme une probabilité supervisée calibrée au sens d'un classifieur. C'est un score de force d'appartenance issu de la persistance du point dans son cluster.

### 5.3 `outlier_scores_`, détecter les observations atypiques

La bibliothèque `hdbscan` propose aussi un score d'outlier fondé sur la position du point dans la hiérarchie de densité. Les valeurs élevées indiquent des points plus atypiques relativement à leur environnement.

```python
# Score d'outlier fourni par la bibliothèque hdbscan
outlier_scores = clusterer.outlier_scores_

# Exemple de seuil exploratoire, à justifier sur le contexte métier
threshold = np.quantile(outlier_scores, 0.95)
outliers = outlier_scores >= threshold
```

Le quantile est préférable à un seuil universel, car l'échelle des scores dépend des données et de la configuration. Dans un usage métier, le seuil doit aussi être comparé à des règles opérationnelles et à une revue humaine.

### 5.4 La hiérarchie condensée

Avec la bibliothèque `hdbscan`, on peut inspecter l'arbre condensé :

```python
import matplotlib.pyplot as plt

# Affiche les branches retenues et les séparations de la hiérarchie
clusterer.condensed_tree_.plot(select_clusters=True)
plt.show()
```

Il est aussi possible de produire un tableau exploitable :

```python
# Transforme l'arbre condensé en DataFrame pour analyse ou export
condensed = clusterer.condensed_tree_.to_pandas()

# Les colonnes exactes peuvent évoluer selon la version du package
print(condensed.head())
```

### 5.5 Visualiser les clusters sans confondre projection et vérité

Une projection en deux dimensions peut aider à communiquer, mais elle peut aussi déformer les distances. Si les données sont déjà en deux dimensions, la visualisation est directe. Pour des embeddings ou des données de grande dimension, il faut distinguer :

| Étape | Rôle | Risque |
|------|------|--------|
| Standardisation | Mettre les variables sur des échelles comparables | Modifier la notion métier de distance |
| HDBSCAN sur l'espace de travail | Trouver les groupes | Les dimensions inutiles peuvent perturber la densité |
| UMAP ou PCA pour visualiser | Afficher en 2D ou 3D | La projection peut créer ou rapprocher visuellement des points |
| Analyse des profils | Donner un sens aux clusters | Confondre corrélation et interprétation métier |

Une règle prudente consiste à faire le clustering dans l'espace préparé pour la tâche, puis à utiliser une projection uniquement comme support de diagnostic et de communication.

> 📌 **Points clés, Section 5**
> - `labels_` donne la partition, avec `-1` pour le bruit.
> - `probabilities_` mesure la force d'appartenance, pas une probabilité supervisée calibrée.
> - Les scores d'outlier servent au classement des points atypiques, pas à remplacer une analyse métier.
> - Une visualisation 2D ne prouve pas que les groupes existent dans l'espace original.

---

## 6. Paramètres et stratégie de réglage

### 6.1 `min_cluster_size`, le paramètre de granularité principal

`min_cluster_size` indique la taille minimale d'un groupe que l'on souhaite distinguer. C'est le meilleur point de départ, car il correspond directement à une décision métier ou analytique : « quelle est la plus petite population que je considère comme un segment ? »

Effets typiques :

| Valeur de `min_cluster_size` | Effet probable |
|------------------------------|----------------|
| Petite | Plus de petits clusters, davantage de granularité, sensibilité aux micro-structures |
| Grande | Moins de clusters, groupes plus larges, petites structures absorbées ou classées comme bruit |

Il ne faut pas choisir ce paramètre uniquement pour obtenir un nombre de clusters agréable. Il doit refléter la taille minimale exploitable pour la décision.

### 6.2 `min_samples`, la conservativité face au bruit

`min_samples` contrôle la quantité de voisinage utilisée pour estimer la densité locale. Une valeur élevée rend la définition des zones denses plus exigeante et tend à produire davantage de bruit ou des clusters plus conservateurs.

Si `min_samples` est omis dans la bibliothèque `hdbscan`, il est généralement lié à `min_cluster_size`. Pour une analyse de sensibilité, il est souvent utile de fixer explicitement les deux paramètres afin de distinguer leurs effets.

| `min_samples` | Comportement typique |
|---------------|----------------------|
| Faible | Moins conservateur, davantage de points rattachés, risque de structures fragiles |
| Élevé | Plus conservateur, davantage de bruit, groupes qui doivent être mieux soutenus localement |

### 6.3 `metric`, la notion de proximité

La métrique est souvent plus importante que le choix exact d'un hyperparamètre. Quelques exemples :

| Données | Métriques candidates | Attention |
|---------|----------------------|-----------|
| Variables numériques standardisées | Euclidienne, Manhattan | La standardisation influence fortement la distance |
| Données avec outliers en coordonnées | Manhattan | Peut être plus robuste, mais dépend des distributions |
| Texte ou embeddings | Cosine | Vérifier la normalisation et le sens de similarité |
| Variables binaires | Jaccard, Hamming | Une distance euclidienne peut être peu naturelle |
| Coordonnées géographiques | Haversine | Les coordonnées doivent être en radians selon l'implémentation |

La bonne métrique est celle qui rend « proches » les observations considérées comme similaires dans le contexte de la tâche.

### 6.4 `cluster_selection_method`, `eom` ou `leaf`

Comme vu précédemment, `eom` favorise des groupes persistants et relativement robustes. `leaf` favorise une lecture plus fine de l'arbre.

Une bonne pratique est de comparer les deux sorties en examinant :

- le nombre de clusters ;
- leur taille ;
- la part de bruit ;
- la stabilité des segments ;
- la lisibilité métier ;
- la sensibilité à une petite variation des paramètres.

### 6.5 `cluster_selection_epsilon`, à utiliser avec prudence

Ce paramètre peut imposer une échelle de séparation minimale et permettre de rapprocher le comportement d'une extraction DBSCAN* à un niveau donné. Il est rarement le premier paramètre à régler.

Utilisez-le si :

- vous avez une distance métier interprétable ;
- vous devez contrôler une granularité minimale ;
- vous souhaitez explorer un niveau particulier de la hiérarchie.

### 6.6 Une procédure de réglage reproductible

1. Définir le grain d'une observation et le but du clustering.
2. Choisir la métrique avant de chercher à optimiser le nombre de clusters.
3. Préparer les variables, en documentant imputation, standardisation et réduction éventuelle.
4. Fixer `min_cluster_size` à partir de la plus petite taille de segment exploitable.
5. Tester quelques valeurs de `min_samples`, sans multiplier les essais au hasard.
6. Comparer `eom` et `leaf` si la granularité est incertaine.
7. Examiner la proportion de bruit et les probabilités d'appartenance.
8. Profiler les clusters sur les variables d'origine.
9. Vérifier la stabilité sur des sous-échantillons ou des périodes différentes.
10. Documenter les paramètres, la version logicielle et la date d'entraînement.

> 📌 **Points clés, Section 6**
> - Commencer par la granularité métier via `min_cluster_size`.
> - Fixer explicitement `min_samples` lorsque l'on veut analyser séparément robustesse et taille minimale.
> - Ne pas utiliser la métrique par défaut sans vérifier qu'elle correspond au sens métier de la proximité.
> - Une bonne segmentation doit être stable, interprétable et actionnable, pas seulement visuellement séduisante.

---

## 7. Implémentation Python

### 7.1 Installation et choix de l'API

Deux APIs sont courantes :

| API | Avantages | Limites ou différences |
|-----|-----------|------------------------|
| `hdbscan` | API historique, visualisations et sorties riches comme `outlier_scores_` et `condensed_tree_` | Dépendance séparée de scikit-learn |
| `sklearn.cluster.HDBSCAN` | Intégration directe dans scikit-learn, pipelines et outils familiers | Certaines fonctionnalités de la bibliothèque historique peuvent manquer ou différer |

Installation de la bibliothèque historique :

```bash
# Installation de l'implémentation scikit-learn-contrib
python -m pip install hdbscan
```

Dans scikit-learn, `HDBSCAN` est disponible dans les versions récentes. Vérifiez la version installée avant d'utiliser un exemple :

```python
import sklearn

# Permet de vérifier que l'API HDBSCAN est disponible dans l'environnement
print(sklearn.__version__)
```

### 7.2 Exemple complet avec `hdbscan`

```python
import numpy as np
import pandas as pd
import hdbscan
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Crée un jeu non linéaire pour illustrer les formes non sphériques
X, _ = make_moons(n_samples=600, noise=0.08, random_state=42)

# Met les variables sur une échelle comparable avant de calculer les distances
X_scaled = StandardScaler().fit_transform(X)

# Crée le clusterer avec une taille minimale de groupe interprétable
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,       # Ignore les branches contenant moins de 30 points
    min_samples=10,            # Rend l'estimation de densité modérément conservatrice
    metric="euclidean",        # Distance adaptée à ces variables numériques
    cluster_selection_method="eom",  # Sélectionne les groupes les plus stables
    prediction_data=True       # Prépare les données utiles pour prédire de nouveaux points
)

# Construit la hiérarchie et sélectionne les clusters stables
clusterer.fit(X_scaled)

# Récupère un label par observation, -1 signifiant bruit
labels = clusterer.labels_

# Récupère la force d'appartenance de chaque observation à son cluster
probabilities = clusterer.probabilities_

# Compte les clusters hors bruit
n_clusters = len(set(labels)) - int(-1 in labels)

# Calcule la proportion d'observations classées comme bruit
noise_rate = np.mean(labels == -1)

print(f"Nombre de clusters : {n_clusters}")
print(f"Part de bruit : {noise_rate:.1%}")
print(f"Taille moyenne d'appartenance : {probabilities[labels != -1].mean():.3f}")
```

### 7.3 Visualiser le résultat

```python
import matplotlib.pyplot as plt

# Construit une palette simple, une couleur par label de cluster
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
color_by_label = dict(zip([label for label in unique_labels if label != -1], colors))

# Utilise du gris pour le bruit et la couleur du cluster pour les autres points
point_colors = [
    "lightgray" if label == -1 else color_by_label[label]
    for label in labels
]

# Dessine les points avec une transparence qui facilite la lecture des zones denses
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=point_colors, s=18, alpha=0.75)
plt.title("Clusters HDBSCAN, bruit en gris")
plt.xlabel("Variable 1 standardisée")
plt.ylabel("Variable 2 standardisée")
plt.show()
```

### 7.4 Intégrer HDBSCAN à un DataFrame

```python
# Copie le tableau d'origine pour conserver les variables explicatives intactes
result = pd.DataFrame(X, columns=["x1", "x2"])

# Ajoute le label de cluster à chaque ligne dans le même ordre que X
result["cluster"] = labels

# Ajoute la force d'appartenance utile pour filtrer les cas ambigus
result["membership_strength"] = probabilities

# Marque explicitement les observations de bruit
result["is_noise"] = result["cluster"].eq(-1)

# Produit un profil de taille et de confiance par cluster
profile = (
    result[result["cluster"] != -1]
    .groupby("cluster")
    .agg(
        n_points=("cluster", "size"),
        mean_membership=("membership_strength", "mean"),
        median_x1=("x1", "median"),
        median_x2=("x2", "median"),
    )
    .sort_values("n_points", ascending=False)
)

print(profile)
```

### 7.5 Utiliser un pipeline de préparation

HDBSCAN est sensible à la géométrie des variables. Une préparation explicite est donc préférable :

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Exemple de colonnes numériques et catégorielles d'un jeu de données client
numeric_features = ["recency_days", "frequency", "monetary_value"]
categorical_features = ["country"]

# Standardise les variables numériques et encode les catégories en indicateurs binaires
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Transforme les variables sans utiliser de cible supervisée
X_prepared = preprocessor.fit_transform(customer_data)

# HDBSCAN accepte les matrices clairsemées seulement dans certaines configurations,
# donc une conversion dense peut être nécessaire si la dimension reste raisonnable
X_prepared_dense = X_prepared.toarray()

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=15,
    metric="euclidean"
)
labels = clusterer.fit_predict(X_prepared_dense)
```

Dans de grandes dimensions, convertir une matrice très creuse en matrice dense peut dépasser la mémoire disponible. Il faut alors revoir l'encodage, réduire la dimension, choisir une métrique compatible ou utiliser une implémentation adaptée.

### 7.6 Prédire le cluster de nouveaux points

Le clustering est non supervisé, mais la bibliothèque `hdbscan` peut estimer l'appartenance de nouveaux points si `prediction_data=True` a été utilisé :

```python
from hdbscan.prediction import approximate_predict

# Applique exactement le même prétraitement aux nouveaux points
X_new_scaled = scaler.transform(X_new)

# Approxime le cluster et la force d'appartenance sans reconstruire toute la hiérarchie
new_labels, new_probabilities = approximate_predict(
    clusterer,
    X_new_scaled
)
```

Cette prédiction est une approximation basée sur la structure apprise. Elle ne signifie pas que le modèle devient un classifieur supervisé classique. Il faut surveiller les nouveaux points éloignés de la distribution d'entraînement et prévoir une stratégie de réentraînement.

> 📌 **Points clés, Section 7**
> - L'API `hdbscan` apporte des sorties riches et des outils dédiés à l'analyse de la hiérarchie.
> - Les données doivent être préparées selon une métrique cohérente avec le problème.
> - Il faut conserver les labels, les scores d'appartenance, les paramètres et le prétraitement.
> - La prédiction de nouveaux points est approximative et doit être accompagnée d'un suivi de dérive.

---

## 8. Validation et interprétation

### 8.1 Pourquoi un score unique ne suffit pas

Un algorithme de clustering peut produire des groupes mathématiquement cohérents mais inutiles. La validation doit combiner plusieurs angles :

1. cohérence interne de la géométrie ;
2. stabilité lorsque les données ou les paramètres changent légèrement ;
3. interprétabilité des profils ;
4. actionnabilité métier ;
5. respect des contraintes éthiques et opérationnelles.

### 8.2 Mesures internes avec prudence

La silhouette compare la cohésion d'un point dans son cluster à sa séparation d'avec les autres clusters :

$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$$

où :

- `a(i)` est la distance moyenne entre le point `i` et les points de son cluster ;
- `b(i)` est la plus petite distance moyenne entre `i` et un autre cluster ;
- une valeur proche de 1 indique une bonne séparation selon cette géométrie.

Pour HDBSCAN, il faut décider comment traiter le bruit. Exclure les points `-1` peut simplifier le calcul mais biaiser l'évaluation vers les points les plus faciles. Les scores internes ne doivent donc pas être utilisés seuls.

### 8.3 Stabilité par rééchantillonnage

Une procédure utile consiste à réentraîner le modèle sur plusieurs sous-échantillons ou avec de petites perturbations :

```python
from sklearn.utils import resample

# Stocke les distributions de tailles pour plusieurs réplications
replication_profiles = []

for seed in range(10):
    # Rééchantillonne avec remise pour tester la robustesse de la structure
    X_bootstrap = resample(X_scaled, random_state=seed)

    # Recrée un modèle identique pour éviter de réutiliser un état interne
    model = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean"
    )

    # Apprend la structure sur le rééchantillon
    bootstrap_labels = model.fit_predict(X_bootstrap)

    # Conserve le nombre de clusters et la part de bruit comme indicateurs simples
    n_clusters = len(set(bootstrap_labels)) - int(-1 in bootstrap_labels)
    noise_rate = np.mean(bootstrap_labels == -1)
    replication_profiles.append((n_clusters, noise_rate))

stability_summary = pd.DataFrame(
    replication_profiles,
    columns=["n_clusters", "noise_rate"]
)
print(stability_summary.describe())
```

Cette expérience ne suffit pas à aligner les labels entre réplications, car le numéro d'un cluster est arbitraire. Pour comparer finement les partitions, utilisez une mesure comme l'Adjusted Rand Index sur des observations communes ou une méthode de mise en correspondance des clusters.

### 8.4 Profiler les clusters dans les variables d'origine

Le profilage transforme un label abstrait en description utilisable :

```python
# Ajoute les labels aux variables métier d'origine
profile_data = customer_data.copy()
profile_data["cluster"] = labels

# Exclut le bruit pour commencer le profilage des segments structurés
cluster_profile = (
    profile_data[profile_data["cluster"] != -1]
    .groupby("cluster")
    .agg(
        n_clients=("cluster", "size"),
        median_recency=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_value=("monetary_value", "median"),
    )
    .sort_values("n_clients", ascending=False)
)
print(cluster_profile)
```

Il est important de comparer les profils à la population globale. Une valeur élevée n'est informative que si elle est réellement différente de la référence et si elle peut conduire à une action.

### 8.5 Cohérence temporelle et dérive

Un clustering peut être stable sur un échantillon historique et se dégrader lorsque le comportement change. Suivez au minimum :

| Indicateur | Question |
|------------|----------|
| Taille des clusters | Les populations restent-elles suffisamment représentées ? |
| Taux de bruit | Davantage de points deviennent-ils atypiques ? |
| Probabilité moyenne | Les nouvelles observations sont-elles moins bien rattachées ? |
| Profils métier | Les caractéristiques des segments se déplacent-elles ? |
| Actions générées | Les segments restent-ils utiles à la décision ? |

> 📌 **Points clés, Section 8**
> - La validation d'un clustering est multidimensionnelle.
> - Les scores internes décrivent une géométrie, pas une valeur métier.
> - La stabilité doit être testée avec des rééchantillonnages, des périodes et des paramètres voisins.
> - Les clusters doivent être profilés dans les variables d'origine et suivis dans le temps.

---

## 9. Cas d'usage guidé, segmenter des clients

### 9.1 Cadrage

Objectif : segmenter des clients selon trois variables :

- `recency_days` : nombre de jours depuis la dernière commande ;
- `frequency` : nombre de commandes sur une période ;
- `monetary_value` : montant total dépensé.

La question n'est pas « combien de clusters puis-je obtenir ? ». Elle est : « existe-t-il des groupes de clients suffisamment cohérents et persistants pour adapter une action commerciale ? »

### 9.2 Préparation

Les variables ont des unités différentes et peuvent être asymétriques. Une procédure plausible est :

1. vérifier les valeurs impossibles ;
2. traiter les valeurs manquantes selon leur signification ;
3. appliquer éventuellement `log1p` aux montants ou fréquences très asymétriques ;
4. standardiser les variables ;
5. documenter chaque transformation.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sélectionne les variables correspondant à la définition du segment
features = ["recency_days", "frequency", "monetary_value"]
X_customer = customer_data[features].copy()

# Réduit l'effet des distributions très asymétriques sans modifier les zéros
X_customer["frequency"] = np.log1p(X_customer["frequency"])
X_customer["monetary_value"] = np.log1p(X_customer["monetary_value"])

# Standardise les colonnes afin que chaque variable contribue à la distance
scaler = StandardScaler()
X_customer_scaled = scaler.fit_transform(X_customer)
```

### 9.3 Recherche de paramètres raisonnables

On peut comparer quelques configurations choisies à partir de la taille minimale exploitable :

```python
configurations = [
    {"min_cluster_size": 50, "min_samples": 10},
    {"min_cluster_size": 50, "min_samples": 25},
    {"min_cluster_size": 100, "min_samples": 25},
]

rows = []
for config in configurations:
    # Entraîne un modèle séparé pour chaque hypothèse de granularité
    model = hdbscan.HDBSCAN(**config, metric="euclidean")
    model.fit(X_customer_scaled)

    # Calcule des indicateurs descriptifs comparables entre configurations
    labels_config = model.labels_
    noise_mask = labels_config == -1
    n_clusters = len(set(labels_config)) - int(noise_mask.any())
    assigned = ~noise_mask
    mean_probability = (
        model.probabilities_[assigned].mean() if assigned.any() else np.nan
    )

    rows.append({
        **config,
        "n_clusters": n_clusters,
        "noise_rate": noise_mask.mean(),
        "mean_probability": mean_probability,
    })

comparison = pd.DataFrame(rows)
print(comparison)
```

Cette table ne choisit pas automatiquement la meilleure configuration. Elle permet de discuter des compromis avec les parties prenantes : une configuration qui produit 40 % de bruit n'est pas forcément mauvaise, mais elle demande une justification précise.

### 9.4 Interprétation et action

Supposons que l'analyse mette en évidence :

| Segment | Profil possible | Action à tester |
|---------|-----------------|-----------------|
| A | Clients récents, fréquents et à forte valeur | Fidélisation et accès anticipé |
| B | Clients anciens, peu fréquents, valeur moyenne | Réactivation avec offre ciblée |
| C | Clients rares, forte valeur ponctuelle | Service personnalisé, mais échantillon à surveiller |
| Bruit | Profils ambigus ou très atypiques | Ne pas forcer une campagne automatique |

Ces intitulés ne doivent pas être déduits du numéro de cluster. Ils doivent être construits après l'analyse des variables et validés avec les équipes métier.

> 📌 **Points clés, Section 9**
> - Un cas de clustering commence par une décision à améliorer, pas par un algorithme.
> - Les transformations et la métrique définissent ce que signifie « client similaire ».
> - Le bruit peut être une population à traiter séparément, pas une population à supprimer silencieusement.
> - Un segment ne devient utile qu'après profilage, validation métier et définition d'une action.

---

## 10. Guide de choix et comparaisons

### 10.1 HDBSCAN face aux autres méthodes

| Critère | K-means | DBSCAN | HDBSCAN | Clustering hiérarchique agglomératif |
|---------|---------|--------|---------|--------------------------------------|
| Nombre de clusters | À fournir | Non requis directement | Non requis directement | À choisir lors de la coupe |
| Formes arbitraires | Faible | Bonne | Bonne | Dépend de la liaison |
| Densités différentes | Faible | Fragile avec un `eps` unique | Meilleure gestion pratique | Variable |
| Bruit explicite | Non natif | Oui | Oui | Non natif |
| Granularité hiérarchique | Non | Non | Oui | Oui |
| Probabilité ou force d'appartenance | Non | Limitée | Oui, selon l'implémentation | Non standard |
| Coût et scalabilité | Très bon | Bon à variable | Bon à variable | Peut devenir coûteux |
| Paramètre central | `n_clusters` | `eps`, `min_samples` | `min_cluster_size`, `min_samples` | Méthode de liaison, coupe |

### 10.2 Arbre de décision

```
Les groupes sont-ils approximativement sphériques et le nombre de groupes connu ?
    ├── Oui → Commencer par K-means ou un modèle de mélange
    └── Non
        │
        Les points atypiques doivent-ils être identifiés explicitement ?
            ├── Non → Comparer clustering hiérarchique, GMM ou méthodes adaptées
            └── Oui
                │
                Les densités semblent-elles varier selon les groupes ?
                    ├── Oui → Essayer HDBSCAN
                    └── Non → Comparer DBSCAN et HDBSCAN
```

### 10.3 Quand préférer DBSCAN

DBSCAN peut être préférable lorsque :

- une distance `eps` a une interprétation métier claire ;
- les densités sont relativement comparables ;
- on souhaite contrôler explicitement un seul niveau de densité ;
- le jeu de données est assez simple pour un réglage direct.

### 10.4 Quand HDBSCAN n'est pas le bon outil

HDBSCAN n'est pas automatiquement préférable lorsque :

- les variables ne possèdent pas de métrique de distance pertinente ;
- les clusters sont définis par une relation supervisée ou une cible connue ;
- chaque observation doit obligatoirement recevoir un segment ;
- la dimension est très élevée et la notion de voisinage devient peu informative ;
- la structure est temporelle, séquentielle ou relationnelle et exige un modèle spécifique.

> 📌 **Points clés, Section 10**
> - HDBSCAN est particulièrement intéressant pour des formes arbitraires, du bruit et des densités variables.
> - K-means reste une excellente baseline lorsque ses hypothèses sont raisonnables.
> - Le choix doit partir de la géométrie, du besoin de bruit explicite et de la décision métier.

---

## 11. Pièges Courants

**Piège 1 : standardiser sans réfléchir à la métrique**

Mettre toutes les colonnes sur une même échelle est souvent utile, mais cela change la définition de la proximité. Une variable métier importante peut être sous-pondérée ou sur-pondérée. **Documentez les transformations et vérifiez que la distance obtenue correspond à votre notion de similarité.**

**Piège 2 : chercher un nombre de clusters précis**

HDBSCAN ne doit pas être forcé à produire « exactement cinq segments » uniquement parce qu'un tableau de bord en prévoit cinq. **Si un nombre cible est une contrainte forte, comparez avec une méthode conçue pour contrôler directement ce nombre et expliquez le compromis.**

**Piège 3 : interpréter `probabilities_` comme une probabilité calibrée**

La force d'appartenance est issue de la persistance dans la hiérarchie. Elle n'est pas équivalente à la probabilité qu'un client appartienne réellement à un segment latent. **Utilisez-la comme score de confiance relatif, puis validez les cas ambigus.**

**Piège 4 : supprimer le bruit avant l'analyse**

Retirer toutes les lignes `-1` peut faire disparaître les comportements rares, les erreurs de saisie ou les populations importantes. **Analysez séparément la taille, le profil et la valeur métier du bruit.**

**Piège 5 : faire le clustering sur une projection UMAP ou t-SNE sans justification**

Une projection 2D est optimisée pour la visualisation et peut modifier les voisinages. **Faites le clustering dans l'espace de travail approprié, puis utilisez la projection comme outil de diagnostic.**

**Piège 6 : modifier `min_cluster_size` et `min_samples` ensemble sans analyser leurs rôles**

Le résultat peut changer fortement, mais vous ne saurez pas si la variation vient de la taille minimale des groupes ou de la conservativité de la densité. **Fixez un paramètre pendant que vous étudiez l'autre.**

**Piège 7 : comparer les numéros de clusters entre deux entraînements**

Le label `0` d'un modèle n'a aucune obligation de correspondre au label `0` d'un autre modèle. **Comparez les partitions, les profils ou les recouvrements, pas les entiers bruts.**

**Piège 8 : utiliser une métrique inadaptée aux embeddings**

La distance euclidienne peut être moins pertinente que la distance cosinus pour des représentations vectorielles normalisées. **Choisissez la métrique à partir de la construction de l'espace et vérifiez sa sensibilité.**

**Piège 9 : conclure à partir d'une seule exécution**

Un clustering peut refléter une variation d'échantillonnage ou une perturbation mineure. **Testez la stabilité sur plusieurs échantillons, périodes et configurations voisines.**

**Piège 10 : confondre cluster mathématique et segment opérationnel**

Un groupe peut être statistiquement distinct mais impossible à cibler, trop petit, instable ou sans action associée. **La validation finale doit inclure les utilisateurs de l'analyse et les contraintes de mise en œuvre.**

---

## 12. Exercices d'auto-évaluation

### Questions de révision

1. Pourquoi DBSCAN peut-il échouer lorsqu'il existe des clusters de densités différentes ?
2. Quelle différence conceptuelle existe entre `min_cluster_size` et `min_samples` ?
3. À quoi sert la distance de portée mutuelle ?
4. Pourquoi HDBSCAN construit-il une hiérarchie avant de sélectionner les clusters finaux ?
5. Que signifie le label `-1` et pourquoi ne faut-il pas le supprimer automatiquement ?
6. Dans quel cas choisir `leaf` plutôt que `eom` ?
7. Pourquoi les scores internes de clustering ne suffisent-ils pas à valider un segment métier ?

### Exercice applicatif

On dispose de 1 000 clients et de trois variables préparées. Le besoin métier est de créer des segments exploitables d'au moins 40 clients. Après trois essais, on obtient :

| Configuration | Nombre de clusters | Part de bruit | Appartenance moyenne |
|---------------|--------------------|---------------|----------------------|
| A, `min_cluster_size=10`, `min_samples=5` | 12 | 4 % | 0,82 |
| B, `min_cluster_size=40`, `min_samples=10` | 4 | 18 % | 0,91 |
| C, `min_cluster_size=100`, `min_samples=50` | 2 | 43 % | 0,96 |

1. Quelle configuration testeriez-vous en premier pour le besoin formulé ?
2. Quels contrôles supplémentaires réaliseriez-vous avant de la retenir ?
3. Que pourriez-vous faire si le segment métier exige six groupes, mais que HDBSCAN ne trouve que quatre groupes stables ?
4. Écrivez le squelette de code qui entraîne la configuration B et ajoute les labels à un DataFrame.

*Solution attendue* : la configuration B est le point de départ le plus cohérent avec la taille minimale de 40 clients. Il faut ensuite vérifier les profils, la stabilité, la distribution du bruit, la qualité des actions et la sensibilité à `min_samples` et à la métrique. Si six groupes sont obligatoires, il faut comparer une sélection `leaf`, une autre méthode de clustering ou une contrainte métier explicite, sans prétendre que six groupes sont naturellement présents.

### Réponses

<details>
<summary>Réponses aux questions de révision</summary>

1. DBSCAN utilise un seuil `eps` unique. Un seuil adapté à un cluster dense peut fragmenter un cluster peu dense, tandis qu'un seuil adapté au cluster peu dense peut fusionner ou élargir excessivement le cluster dense.
2. `min_cluster_size` contrôle la taille minimale d'une branche conservée comme groupe. `min_samples` intervient dans l'estimation locale de la densité et rend le modèle plus ou moins conservateur face aux points isolés.
3. La distance de portée mutuelle combine la distance directe et les distances de cœur des deux points. Elle pénalise une connexion qui traverse une région peu dense.
4. La hiérarchie permet d'observer plusieurs niveaux de densité et de sélectionner des groupes persistants plutôt que de dépendre d'un seuil arbitraire unique.
5. `-1` désigne une observation qui n'a pas été rattachée à un cluster sélectionné. Elle peut être un outlier, un point frontière ou un comportement rare, et doit être analysée avant toute suppression.
6. `leaf` est utile lorsque l'on cherche les sous-groupes les plus fins dans l'arbre, au prix d'un risque plus élevé de sur-segmentation.
7. Un score interne mesure une cohérence géométrique selon une métrique donnée. Il ne mesure ni la stabilité temporelle, ni la valeur métier, ni la possibilité d'agir sur les segments.

</details>

<details>
<summary>Correction indicative de l'exercice applicatif</summary>

```python
# Entraîne HDBSCAN avec une taille minimale alignée sur le besoin métier
model = hdbscan.HDBSCAN(
    min_cluster_size=40,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
)

# Apprend la hiérarchie et renvoie un label par client
customer_data["cluster"] = model.fit_predict(X_customer_scaled)

# Conserve la force d'appartenance pour repérer les cas ambigus
customer_data["cluster_probability"] = model.probabilities_
```

Une analyse complète doit aussi contrôler la taille de chaque cluster, le profil des variables, le taux de bruit, la robustesse sur plusieurs échantillons et l'utilité des actions associées.

</details>

---

## 13. Sources et Références

### Papers fondateurs

- **DBSCAN** : Martin Ester, Hans-Peter Kriegel, Jörg Sander et Xiaowei Xu, 1996, *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*, [PDF KDD](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf).
- **Hierarchical density estimates** : Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek et Jörg Sander, 2015, *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*, [DOI ACM 10.1145/2733381](https://dl.acm.org/doi/10.1145/2733381).
- **HDBSCAN accéléré** : Leland McInnes et John Healy, 2017, *Accelerated Hierarchical Density Based Clustering*, [arXiv:1705.07321](https://arxiv.org/abs/1705.07321).
- **Bibliothèque hdbscan** : Leland McInnes, John Healy et Steve Astels, 2017, *hdbscan: Hierarchical density based clustering*, [JOSS, DOI 10.21105/joss.00205](https://joss.theoj.org/papers/10.21105/joss.00205).

### Documentation technique

- [Documentation officielle de la bibliothèque hdbscan](https://hdbscan.readthedocs.io/)
- [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), dérivation par distance de portée mutuelle, arbre couvrant minimal et arbre condensé.
- [Basic Usage of HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html), API, métriques et sorties courantes.
- [Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html), sélection de `min_cluster_size`, `min_samples` et `leaf`.
- [API `sklearn.cluster.HDBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html), intégration dans scikit-learn.
- [Guide scikit-learn sur le clustering](https://scikit-learn.org/stable/modules/clustering.html), comparaison avec DBSCAN et autres méthodes.

### Pour approfondir

- [Extraction de DBSCAN* depuis HDBSCAN](https://hdbscan.readthedocs.io/en/latest/dbscan_from_hdbscan.html), utile pour comprendre le lien entre hiérarchie et niveau de densité fixé.
- [Dépôt GitHub scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan), code source, exemples et informations de citation.
- [Documentation NVIDIA cuML](https://docs.nvidia.com/cuml/), piste d'exploration pour les traitements accélérés sur GPU, en vérifiant la couverture exacte de HDBSCAN dans la version utilisée.

---

## 14. Liens connexes

- [Prétraitement des données](../../01_fundamentals/preprocessing_data.md), nettoyage, encodage et mise à l'échelle.
- [t-SNE](../dimension_reduction/tsne.md), visualisation non linéaire à ne pas confondre avec un algorithme de clustering.
- [UMAP](../dimension_reduction/umap.md), réduction de dimensionnalité pouvant servir à l'exploration et à la visualisation.
- [Statistiques descriptives](../../../00_statistics_foundations/measures_dispersion.md), dispersion, quantiles et valeurs atypiques.
- [Principes de visualisation](../../../00_statistics_foundations/data_visualization_principles.md), présentation prudente des groupes et des incertitudes.

---

*Dernière mise à jour : Septembre 2026*
