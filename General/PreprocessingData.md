# Preprocessing des Données pour les Modèles de Machine Learning

## Table des Matières
1. [Introduction](#introduction)
2. [Phase 1 : Compréhension et Exploration](#phase-1)
3. [Phase 2 : Nettoyage des Données](#phase-2)
4. [Phase 3 : Gestion des Valeurs Manquantes](#phase-3)
5. [Phase 4 : Détection et Traitement des Outliers](#phase-4)
6. [Phase 5 : Encodage des Variables](#phase-5)
7. [Phase 6 : Normalisation et Standardisation](#phase-6)
8. [Phase 7 : Feature Engineering](#phase-7)
9. [Phase 8 : Séparation des Données](#phase-8)
10. [Phase 9 : Gestion du Déséquilibre des Classes](#phase-9)
11. [Phase 10 : Prévention du Data Leakage](#phase-10)
12. [Checklist et Ordre des Opérations](#checklist)
13. [Références](#references)

---

<br>

## 1. Introduction {#introduction}

Le preprocessing (ou prétraitement) des données représente **80% du temps** d'un projet de Machine Learning :cite[ekx]. La qualité des données d'entrée détermine directement la performance du modèle final selon le principe **"Garbage In, Garbage Out"**.

**Pourquoi le preprocessing est-il critique ?**

- Les données brutes sont vulnérables au **bruit, à la corruption, aux valeurs manquantes et aux incohérences** :cite[ekx]
- Un prétraitement inadéquat peut conduire à de **fausses prédictions** et affecter l'**accuracy** du modèle
- Certains algorithmes sont **sensibles à l'échelle des features** (KNN, SVM, réseaux de neurones) tandis que d'autres sont **invariants** (arbres de décision) :cite[bs0]

Ce cours couvre l'ensemble des étapes de preprocessing dans l'ordre optimal, en expliquant le **pourquoi** de chaque décision.

---

<br>

## 2. Phase 1 : Compréhension et Exploration des Données {#phase-1}

### 2.1 Objectif

Avant toute manipulation, il est essentiel de **comprendre la structure, la distribution et les caractéristiques** des données.

### 2.2 Fonctions Clés

```python
import pandas as pd
import numpy as np

# Chargement
df = pd.read_csv('data.csv')

# Inspection de base
df.head()          # Premières lignes
df.tail()          # Dernières lignes
df.shape           # Dimensions (lignes, colonnes)
df.info()          # Types de données et valeurs non-nulles
df.describe()      # Statistiques descriptives (mean, std, min, max, quartiles)
df.dtypes          # Types de chaque colonne
df.columns         # Noms des colonnes
```

### 2.3 Analyse Exploratoire (EDA)

**Objectifs de l'EDA :**
- Identifier les **variables cibles** (labels) et **variables explicatives** (features)
- Détecter les **patterns**, **corrélations** et **anomalies**
- Comprendre la **distribution** des variables (normale, skewed, bimodale)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution des variables numériques
df.hist(bins=30, figsize=(15,10))

# Matrice de corrélation
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Boxplots pour détecter visuellement les outliers
df.boxplot(figsize=(15,10))
```

**Pourquoi cette étape ?**
- Permet d'identifier **rapidement** les problèmes (valeurs manquantes, outliers, déséquilibre)
- Guide les **choix de preprocessing** adaptés au type de données
- Évite les transformations **inutiles ou inappropriées**

---

<br>

## 3. Phase 2 : Nettoyage des Données {#phase-2}

### 3.1 Gestion des Doublons

**Pourquoi supprimer les doublons ?**
- Créent un **biais** dans l'apprentissage (certains exemples sont sur-représentés)
- Peuvent conduire à du **overfitting**
- Faussent les **statistiques** (mean, variance)

```python
# Détecter les doublons
df.duplicated().sum()

# Supprimer les doublons
df.drop_duplicates(inplace=True)
```

### 3.2 Correction des Incohérences

**Types d'incohérences fréquentes :**
- **Formats de date** non standardisés
- **Unités** différentes (km vs miles, °C vs °F)
- **Casse** (majuscules/minuscules)
- **Espaces** parasites

```python
# Conversion de dates
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Nettoyage des chaînes de caractères
df['nom'] = df['nom'].str.strip().str.lower()

# Unification des unités (exemple)
df['temperature_celsius'] = df['temperature_fahrenheit'].apply(lambda x: (x - 32) * 5/9)
```

---

<br>

## 4. Phase 3 : Gestion des Valeurs Manquantes {#phase-3}

### 4.1 Types de Données Manquantes

Selon Little & Rubin (2002), il existe **3 types de mécanismes** de données manquantes :cite[bpe] :

1. **MCAR (Missing Completely At Random)** : La probabilité de valeur manquante est identique pour tous les cas
2. **MAR (Missing At Random)** : La probabilité dépend des valeurs observées mais pas de la valeur manquante elle-même
3. **NMAR (Not Missing At Random)** : La probabilité dépend de la valeur manquante elle-même

### 4.2 Détection

```python
# Nombre de valeurs manquantes par colonne
df.isnull().sum()

# Pourcentage de valeurs manquantes
(df.isnull().sum() / len(df)) * 100

# Visualisation
import missingno as msno
msno.matrix(df)
msno.heatmap(df)
```

### 4.3 Méthodes d'Imputation

#### 4.3.1 Suppression

**Quand l'utiliser ?**
- Moins de **5% de valeurs manquantes**
- Données de type **MCAR**
- Dataset **suffisamment large**

```python
# Suppression des lignes avec valeurs manquantes
df.dropna(inplace=True)

# Suppression des colonnes avec plus de X% de NaN
threshold = 0.5
df = df.loc[:, df.isnull().mean() < threshold]
```

#### 4.3.2 Imputation Simple

**Mean/Median Imputation** :cite[bpe]

- **Mean** : Sensible aux outliers, assume distribution normale
- **Median** : Plus robuste aux outliers

**Pourquoi le median est préférable ?**
- Ne fausse pas la distribution en présence d'**outliers**
- Plus stable pour les distributions **skewed**

```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer_mean = SimpleImputer(strategy='mean')
df['colonne'] = imputer_mean.fit_transform(df[['colonne']])

# Median imputation
imputer_median = SimpleImputer(strategy='median')
df['colonne'] = imputer_median.fit_transform(df[['colonne']])

# Mode pour variables catégorielles
imputer_mode = SimpleImputer(strategy='most_frequent')
```

#### 4.3.3 KNN Imputation

**Principe** : Utilise les **K voisins les plus proches** pour estimer la valeur manquante :cite[bpe].

**Avantages** :
- Capture les **relations entre variables**
- Plus précis que mean/median

**Inconvénients** :
- Coûteux en **calcul** (O(n²))
- Nécessite de **choisir K**

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='uniform')
df_imputed = imputer.fit_transform(df)
```

#### 4.3.4 MissForest (Recommandé)

**Principe** : Utilise un **Random Forest** pour prédire les valeurs manquantes de manière itérative :cite[bpe].

**Pourquoi MissForest est-il supérieur ?**
- Capture les **relations non-linéaires complexes** entre variables
- Fonctionne avec des **données mixtes** (numériques + catégorielles)
- **Performances supérieures** selon de multiples études :cite[bpe]

```python
from missingpy import MissForest

imputer = MissForest(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df)
```

#### 4.3.5 MICE (Multiple Imputation by Chained Equations)

**Principe** : Génère **plusieurs datasets imputés** en modélisant chaque variable avec valeurs manquantes :cite[bpe].

**Avantages** :
- Capture l'**incertitude** de l'imputation
- Performances très proches de MissForest :cite[bpe]

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df)
```

### 4.4 Recommandations

**Selon l'étude comparative de 2025** :cite[bpe] :

| Méthode | RMSE (Breast Cancer) | Cas d'usage |
|---------|---------------------|-------------|
| **MissForest** | **Le plus faible** | Données complexes, mixtes |
| **MICE** | Très proche de MissForest | Haute incertitude, MAR/MCAR |
| **KNN** | Moyen | Datasets moyens, relations spatiales |
| **Mean/Median** | Élevé | Petits datasets, faible % de NaN |
| **LOCF** | **Le plus élevé** | **À éviter** |

**Règle générale** :cite[bpe] :
- **MissForest** : Pour datasets avec variables numériques ET catégorielles, interactions complexes
- **MICE** : Pour datasets avec haute incertitude ou besoin de capturer la variabilité
- **Mean/Median** : Seulement si < 5% de valeurs manquantes et données simples

---

<br>

## 5. Phase 4 : Détection et Traitement des Outliers {#phase-4}

### 5.1 Qu'est-ce qu'un Outlier ?

Un **outlier** (valeur aberrante) est une observation qui s'écarte significativement des autres valeurs du dataset.

**Pourquoi détecter les outliers ?**
- Peuvent **fausser** les statistiques (mean, variance)
- Affectent les modèles **sensibles** (régression linéaire, KNN)
- Peuvent être des **erreurs de mesure** ou des **cas rares mais réels**

### 5.2 Méthodes de Détection

#### 5.2.1 Z-Score

**Principe** : Mesure l'écart par rapport à la moyenne en unités d'**écart-type**.

$$Z = \frac{x - \mu}{\sigma}$$

**Règle** : Valeur est un outlier si $$|Z| > 3$$ :cite[bl4].

**Pourquoi 3 ?**
- Selon la **loi normale**, 99.7% des valeurs sont dans $$[\mu - 3\sigma, \mu + 3\sigma]$$

```python
from scipy import stats

# Calcul du Z-score
z_scores = np.abs(stats.zscore(df['colonne']))

# Détection des outliers
outliers = df[z_scores > 3]

# Suppression
df_no_outliers = df[z_scores <= 3]
```

**Limitations** :
- **Assume une distribution normale**
- Sensible aux outliers extrêmes (ils affectent la mean et std)

#### 5.2.2 IQR (Interquartile Range)

**Principe** : Plus **robuste** car utilise la **médiane** :cite[b2w].

$$IQR = Q3 - Q1$$

**Règle** :
- Outlier inférieur : $$x < Q1 - 1.5 \times IQR$$
- Outlier supérieur : $$x > Q3 + 1.5 \times IQR$$

**Pourquoi 1.5 ?**
- Convention statistique de **Tukey** (compromis entre sensibilité et spécificité)

```python
Q1 = df['colonne'].quantile(0.25)
Q3 = df['colonne'].quantile(0.75)
IQR = Q3 - Q1

# Définition des bornes
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrage
df_no_outliers = df[(df['colonne'] >= lower_bound) & (df['colonne'] <= upper_bound)]
```

**Avantages** :
- **Robuste** aux distributions non-normales
- Ne nécessite **aucune hypothèse** sur la distribution

#### 5.2.3 Isolation Forest

**Principe** : Algorithme d'**apprentissage automatique** qui isole les outliers en utilisant des arbres de décision aléatoires :cite[ej2].

**Pourquoi Isolation Forest ?**
- Détecte les outliers dans des **espaces multidimensionnels**
- Ne fait **aucune hypothèse** sur la distribution
- Efficace pour les **datasets complexes**

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(df[['col1', 'col2', 'col3']])

# -1 pour outliers, 1 pour inliers
df_no_outliers = df[outliers == 1]
```

### 5.3 Traitement des Outliers

**Options** :

1. **Suppression** : Si erreur de mesure confirmée
2. **Transformation** : Log, Box-Cox pour réduire l'impact
3. **Winsorization** : Remplacer par une valeur seuil (percentile 1% et 99%)
4. **Garder** : Si les outliers sont informatifs (fraude, maladies rares)

```python
# Winsorization
from scipy.stats.mstats import winsorize

df['colonne_winsorized'] = winsorize(df['colonne'], limits=[0.01, 0.01])
```

**Règle de décision** :
- **Supprimer** : Si < 1% du dataset et clairement des erreurs
- **Transformer** : Si distribution skewed (log, sqrt)
- **Garder** : Si outliers = information (détection fraude, diagnostic médical)

---

<br>

## 6. Phase 5 : Encodage des Variables Catégorielles {#phase-5}

### 6.1 Pourquoi Encoder ?

La plupart des algorithmes ML ne peuvent traiter que des **valeurs numériques**. Les variables catégorielles doivent être converties.

### 6.2 Label Encoding

**Principe** : Attribue un **entier unique** à chaque catégorie.

**Quand l'utiliser ?**
- Variables **ordinales** (ordre logique : petit, moyen, grand)
- **Arbres de décision** (invariants à l'échelle)

**Pourquoi éviter pour les variables nominales ?**
- Introduit un **ordre artificiel** (1 < 2 < 3) qui n'existe pas
- Peut **biaiser** les modèles basés sur la distance (KNN, régression)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['categorie_encoded'] = le.fit_transform(df['categorie'])
```

### 6.3 One-Hot Encoding

**Principe** : Crée une **colonne binaire** pour chaque catégorie.

**Quand l'utiliser ?**
- Variables **nominales** (pas d'ordre : couleur, pays)
- Modèles **linéaires** (régression, SVM)

**Pourquoi ?**
- Ne crée **aucun ordre artificiel**
- Chaque catégorie est **indépendante**

```python
df_encoded = pd.get_dummies(df, columns=['categorie'], drop_first=True)

# Avec sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['categorie']])
```

**Option `drop_first=True`** :
- Évite la **multicolinéarité parfaite** (dummy variable trap)
- K-1 colonnes suffisent pour K catégories

### 6.4 Target Encoding

**Principe** : Remplace chaque catégorie par la **moyenne de la variable cible** pour cette catégorie.

**Pourquoi l'utiliser ?**
- Très efficace pour les **arbres de décision** et **boosting**
- Réduit la **dimensionnalité** (1 colonne au lieu de K)

**Risque** :
- **Data leakage** si mal implémenté (voir Phase 10)

```python
# Calculer la moyenne par catégorie (sur train uniquement)
target_mean = df_train.groupby('categorie')['target'].mean()

# Appliquer au train et test
df_train['categorie_encoded'] = df_train['categorie'].map(target_mean)
df_test['categorie_encoded'] = df_test['categorie'].map(target_mean)
```

### 6.5 Comparaison

| Méthode | Avantages | Inconvénients | Cas d'usage |
|---------|-----------|---------------|-------------|
| **Label Encoding** | Simple, compact | Ordre artificiel | Variables ordinales, arbres |
| **One-Hot** | Pas d'ordre | Haute dimensionnalité | Régressions, SVM, NN |
| **Target Encoding** | Compact, informatif | Risque de leakage | Arbres, boosting |

---

<br>

## 7. Phase 6 : Normalisation et Standardisation {#phase-6}

### 7.1 Pourquoi Scaler les Données ?

**Certains algorithmes sont sensibles à l'échelle** des features :cite[bs0] :

- **Sensibles** : KNN, SVM, Régression (avec régularisation), PCA, Réseaux de neurones
- **Invariants** : Arbres de décision, Random Forest, XGBoost

**Pourquoi cette sensibilité ?**
- Les algorithmes basés sur la **distance** (KNN, SVM) donnent plus de poids aux features avec de grandes valeurs
- La **descente de gradient** converge plus rapidement avec des features à la même échelle :cite[bs0]
- Les **poids** dans les réseaux de neurones se mettent à jour de manière **inégale** si les features ont des échelles différentes

### 7.2 Standardisation (Z-score Normalization)

**Formule** :

$$z = \frac{x - \mu}{\sigma}$$

**Résultat** : Moyenne = 0, Écart-type = 1

**Quand l'utiliser ?**
- Algorithmes assumant une **distribution normale** ou centrée (PCA, LDA)
- Réseaux de neurones avec activation **tanh** ou **sigmoid**
- **SVM** avec kernel RBF

**Pourquoi ?**
- Les poids sont initialisés autour de **0**
- La descente de gradient est plus **stable**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['col1', 'col2']])
```

### 7.3 Normalisation (Min-Max Scaling)

**Formule** :

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Résultat** : Valeurs dans l'intervalle **[0, 1]**

**Quand l'utiliser ?**
- Réseaux de neurones avec activation **ReLU**
- Algorithmes nécessitant des valeurs **bornées** (images)
- Quand on connaît les **bornes** des features

**Pourquoi ?**
- Préserve la **forme de la distribution**
- Idéal pour les données déjà **bornées** (pixels, probabilités)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[['col1', 'col2']])
```

### 7.4 Comparaison et Règles de Décision

| Critère | Standardisation | Normalisation |
|---------|----------------|---------------|
| **Distribution** | Assume normalité | Aucune hypothèse |
| **Outliers** | Robuste | Sensible |
| **Intervalle** | Non borné | [0, 1] |
| **Cas d'usage** | PCA, SVM, NN (tanh) | Images, NN (ReLU) |

**Règle générale** :cite[bs0] :
> "Quand on hésite, **standardiser** est le choix le plus sûr"

### 7.5 Robust Scaling

**Principe** : Utilise la **médiane** et l'**IQR** au lieu de mean/std.

$$x_{robust} = \frac{x - Q2}{IQR}$$

**Quand l'utiliser ?**
- Présence d'**outliers importants**
- Distribution **fortement skewed**

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_robust = scaler.fit_transform(df[['col1', 'col2']])
```

---

<br>

## 8. Phase 7 : Feature Engineering {#phase-7}

### 8.1 Définition

Le **Feature Engineering** consiste à créer de nouvelles variables (**features**) à partir des données existantes pour améliorer les performances du modèle.

**Pourquoi est-ce crucial ?**
- Un **bon feature** peut avoir plus d'impact que le choix de l'algorithme
- Permet de **capturer** des informations latentes

### 8.2 Techniques Classiques

#### 8.2.1 Extraction de Features Temporelles

```python
# Extraction de composants temporels
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

**Pourquoi ?**
- Capture la **saisonnalité** (mois, jour de la semaine)
- Détecte les **patterns cycliques**

#### 8.2.2 Interactions entre Features

```python
# Multiplication
df['feature_interaction'] = df['feature1'] * df['feature2']

# Ratios
df['ratio'] = df['revenu'] / (df['depenses'] + 1)  # +1 pour éviter division par 0

# Polynomiales
df['feature_squared'] = df['feature'] ** 2
```

**Pourquoi ?**
- Capture les **relations non-linéaires**
- Améliore les modèles **linéaires**

#### 8.2.3 Binning (Discrétisation)

```python
# Découpage en bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                         labels=['enfant', 'jeune', 'adulte', 'senior'])

# Quantile-based binning
df['revenu_quartile'] = pd.qcut(df['revenu'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Pourquoi ?**
- Réduit l'impact des **outliers**
- Capture des **seuils** métier importants

#### 8.2.4 Transformations Mathématiques

```python
# Log transform (pour distributions skewed)
df['log_revenu'] = np.log1p(df['revenu'])  # log1p = log(1 + x) pour éviter log(0)

# Box-Cox transform
from scipy.stats import boxcox
df['transformed'], lambda_param = boxcox(df['colonne'] + 1)
```

**Pourquoi log transform ?**
- Réduit le **skewness** (asymétrie)
- Rend la distribution plus **normale**
- Stabilise la **variance**

### 8.3 Sélection de Features

**Pourquoi sélectionner ?**
- Réduire la **dimensionnalité** (curse of dimensionality)
- Améliorer la **vitesse** d'entraînement
- Réduire le **overfitting**

#### 8.3.1 Filter Methods

```python
# Corrélation avec la target
correlation = df.corr()['target'].abs().sort_values(ascending=False)

# Chi2 test pour variables catégorielles
from sklearn.feature_selection import chi2, SelectKBest
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
```

#### 8.3.2 Wrapper Methods

```python
# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

#### 8.3.3 Embedded Methods

```python
# Feature importance avec Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_

# Lasso (L1 regularization)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
selected = X_train.columns[lasso.coef_ != 0]
```

**Ordre optimal selon l'étude de 2025** :cite[bpe] :
> **Toujours réaliser l'imputation AVANT la sélection de features**

**Pourquoi ?**
- Les valeurs manquantes **faussent** les corrélations et importances
- La sélection sur des données incomplètes est **biaisée**

---

<br>

## 9. Phase 8 : Séparation des Données (Train/Validation/Test) {#phase-8}

### 9.1 Pourquoi Séparer ?

- **Train set** : Entraînement du modèle
- **Validation set** : Tuning des hyperparamètres, early stopping
- **Test set** : Évaluation finale **non biaisée**

**Pourquoi 3 sets ?**
- Éviter le **overfitting** sur le test set
- Le validation set permet d'**optimiser** sans toucher au test

### 9.2 Proportions Standards

- **Train** : 60-80%
- **Validation** : 10-20%
- **Test** : 10-20%

```python
from sklearn.model_selection import train_test_split

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)
```

**Option `stratify=y`** :
- Préserve la **distribution des classes** dans chaque split
- **Essentiel** pour les datasets déséquilibrés

### 9.3 Cross-Validation

**Principe** : Diviser le train en **K folds**, entraîner K fois en utilisant K-1 folds pour le train et 1 fold pour la validation.

**Pourquoi ?**
- **Réduit la variance** de l'estimation de performance
- **Utilise toutes les données** pour l'entraînement

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 9.4 Time Series Split

**Pour les données temporelles** :

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

**Pourquoi un split spécial ?**
- Les données futures ne doivent **jamais** être dans le train
- Préserve l'**ordre temporel**

---

<br>

## 10. Phase 9 : Gestion du Déséquilibre des Classes {#phase-9}

### 10.1 Détection

```python
# Distribution des classes
y.value_counts()
y.value_counts(normalize=True)

# Visualisation
sns.countplot(x=y)
```

**Règle** : Déséquilibre si ratio < **1:10** (ou < **1:5** selon les auteurs)

### 10.2 Pourquoi est-ce un Problème ?

- Le modèle apprend à **prédire la classe majoritaire**
- **Accuracy** élevée mais **recall faible** sur la classe minoritaire
- Performance **catastrophique** sur les cas rares (pourtant souvent les plus importants : fraude, maladies)

### 10.3 Techniques de Rééquilibrage

#### 10.3.1 Undersampling

**Principe** : Supprimer des exemples de la **classe majoritaire**.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Avantages** :
- Réduit le **temps d'entraînement**
- Simple

**Inconvénients** :
- **Perte d'information**
- Risque de **underfitting**

#### 10.3.2 Oversampling Simple

**Principe** : Dupliquer des exemples de la **classe minoritaire**.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**Inconvénient** :
- **Overfitting** (duplication exacte)

#### 10.3.3 SMOTE (Synthetic Minority Over-sampling Technique)

**Principe** : Génère des exemples **synthétiques** en interpolant entre les voisins de la classe minoritaire :cite[bn1].

**Pourquoi SMOTE est supérieur ?**
- Pas de duplication exacte → réduit le **overfitting**
- Génère des exemples **plausibles**
- **Performances supérieures** dans de nombreuses études :cite[bn1]

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Variantes** :
- **ADASYN** : Génère plus d'exemples dans les régions difficiles
- **Borderline-SMOTE** : Se concentre sur les exemples proches de la frontière

#### 10.3.4 Class Weights

**Principe** : Pénaliser davantage les **erreurs sur la classe minoritaire**.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Ou poids personnalisés
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
```

**Pourquoi cette approche ?**
- **Aucune modification des données**
- Moins de risque de **overfitting**
- Compatible avec tous les algorithmes sklearn

### 10.4 Comparaison et Recommandations

| Méthode | Avantages | Inconvénients | Quand l'utiliser |
|---------|-----------|---------------|------------------|
| **Undersampling** | Rapide | Perte d'info | Très large dataset |
| **SMOTE** | Pas de duplication | Génère du bruit | Dataset moyen |
| **Class Weights** | Aucune modif data | Nécessite support algo | Par défaut |
| **Ensemble** | Robuste | Complexe | Compétitions |

**Règle générale** :cite[bn1] :
1. **Toujours essayer Class Weights d'abord**
2. Si insuffisant, combiner avec **SMOTE**
3. Utiliser des **métriques adaptées** : F1-score, AUC-ROC, Precision-Recall

---

<br>

## 11. Phase 10 : Prévention du Data Leakage {#phase-10}

### 11.1 Qu'est-ce que le Data Leakage ?

Le **data leakage** survient quand des informations du **test set** "fuient" dans le **train set**, conduisant à des performances **artificiellement gonflées** :cite[csf].

**Pourquoi est-ce critique ?**
- Le modèle ne **généralisera pas** en production
- Les performances reportées sont **fausses**

### 11.2 Types de Leakage

#### 11.2.1 Target Leakage

**Définition** : Une feature contient des informations sur la target qui ne seront **pas disponibles en production**.

**Exemple** :
- Prédire le défaut de paiement en utilisant la feature "compte_bloque" (qui est une conséquence du défaut)

#### 11.2.2 Train-Test Contamination

**Définition** : Appliquer des transformations (scaling, imputation) sur l'**ensemble du dataset** avant le split :cite[ad7].

**Exemple erroné** :

```python
# MAUVAIS - Data leakage !
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Utilise les stats du test set
X_train, X_test = train_test_split(X_scaled)
```

**Exemple correct** :

```python
# BON - Pas de leakage
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Stats sur train uniquement
X_test_scaled = scaler.transform(X_test)        # Applique les stats du train
```

**Pourquoi ?**
- Le `fit` calcule des **statistiques** (mean, std, min, max)
- Ces statistiques doivent provenir **uniquement du train set**
- Le test set doit **simuler des données futures**

### 11.3 Règles de Prévention

**Règle d'or** :cite[csf] :cite[ad7] :
> **Toujours splitter AVANT toute transformation qui "apprend" des données**

**Ordre correct** :
1. **Split** train/test
2. **Imputation** (fit sur train, transform sur test)
3. **Scaling** (fit sur train, transform sur test)
4. **Encoding** (fit sur train, transform sur test)

### 11.4 Pipeline scikit-learn (Solution Recommandée)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Le pipeline garantit l'absence de leakage
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Split AVANT le pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit sur train (chaque step fit uniquement sur train)
pipeline.fit(X_train, y_train)

# Predict sur test (chaque step transform avec stats du train)
y_pred = pipeline.predict(X_test)
```

**Pourquoi les pipelines ?**
- **Automatisent** le fit/transform correct
- **Garantissent** l'absence de leakage
- **Simplifient** le déploiement

---

<br>

## 12. Checklist et Ordre des Opérations {#checklist}

### 12.1 Ordre Optimal des Étapes


- [ ] Exploration (df.info(), df.describe(), visualisations)
- [ ] Suppression des doublons
- [ ] Correction des incohérences (dates, unités, casse)
- [ ] Détection des outliers (visualisation)
- [ ] SPLIT TRAIN/TEST (étape critique !)
- [ ] Imputation des valeurs manquantes (fit sur train)
- [ ] Traitement des outliers (décision basée sur le contexte)
- [ ] Feature Engineering (création de nouvelles features)
- [ ] Encodage des variables catégorielles (fit sur train)
- [ ] Normalisation/Standardisation (fit sur train)
- [ ] Sélection de features (sur train uniquement)
- [ ] Gestion du déséquilibre (SMOTE sur train uniquement)
- [ ] Entraînement du modèle

### 12.2 Points Clés à Retenir

**Imputation** :cite[bpe] :
- Préférer **MissForest** ou **MICE**
- Toujours **avant** la sélection de features

**Scaling** :cite[bs0] :
- **Obligatoire** pour : KNN, SVM, PCA, Réseaux de neurones
- **Inutile** pour : Arbres de décision, Random Forest
- En cas de doute : **Standardiser**

**Encodage** :
- **One-Hot** pour régressions et SVM
- **Label** pour variables ordinales
- **Target** pour arbres et boosting

**Split** :cite[csf] :cite[ad7] :
- **TOUJOURS avant** toute transformation
- Utiliser `stratify=y` pour préserver la distribution

**Déséquilibre** :cite[bn1] :
- Essayer **Class Weights** en premier
- Combiner avec **SMOTE** si nécessaire
- Utiliser **F1-score** et **AUC-ROC** (pas Accuracy)

### 12.3 Fonctions Clés par Étape

| Étape | Fonctions essentielles |
|-------|------------------------|
| **Exploration** | `df.info()`, `df.describe()`, `df.isnull().sum()` |
| **Nettoyage** | `df.drop_duplicates()`, `pd.to_datetime()` |
| **Imputation** | `MissForest`, `IterativeImputer` (MICE), `KNNImputer` |
| **Outliers** | `scipy.stats.zscore()`, `IQR`, `IsolationForest` |
| **Encodage** | `pd.get_dummies()`, `LabelEncoder`, Target Encoding |
| **Scaling** | `StandardScaler`, `MinMaxScaler`, `RobustScaler` |
| **Split** | `train_test_split(stratify=y)`, `StratifiedKFold` |
| **Déséquilibre** | `SMOTE`, `class_weight='balanced'` |
| **Pipeline** | `Pipeline`, `ColumnTransformer` |

---

<br>

## 13. Références {#references}

### Articles Scientifiques

:cite[ekx] Maharana, K., Mondal, S., Nemade, B. (2022). "A review: Data pre-processing and data augmentation techniques". *Global Transitions Proceedings*, 3(1), 91-99.

:cite[bpe] Joel, L.O. et al. (2025). "A comparative study of imputation techniques for missing values in healthcare diagnostic datasets". *International Journal of Data Science and Analytics*, 20, 6357-6373. [https://doi.org/10.1007/s41060-025-00825-9](https://doi.org/10.1007/s41060-025-00825-9)

:cite[bs0] Raschka, S. "When should I apply data normalization/standardization?". [https://sebastianraschka.com/faq/docs/when-to-standardize.html](https://sebastianraschka.com/faq/docs/when-to-standardize.html)

:cite[bl4] "Outlier Detection and Treatment: Z-score, IQR, and Robust Methods". *Medium*. [https://medium.com/@aakash013/outlier-detection-treatment-z-score-iqr-and-robust-methods](https://medium.com/@aakash013/outlier-detection-treatment-z-score-iqr-and-robust-methods)

:cite[b2w] "3 Simple Statistical Methods for Outlier Detection". *Towards Data Science*.

:cite[ej2] "Outlier Detection (with examples)". *Hex Templates*.

:cite[bn1] "A comparative study in class imbalance mitigation when working...". *PMC*, 2025.

:cite[csf] Chawla, A. "Prevent Data Leakage in ML Pipelines". *Daily Dose of DS*.

:cite[ad7] "Data Leakage in Machine Learning: Why You Must Split Before Preprocessing". *Towards AI*.

:cite[a18] "Preventing Data Leakage in Machine Learning: A Guide". *Medium*.

### Ressources Complémentaires

- Stekhoven, D.J., Bühlmann, P. (2012). "MissForest—non-parametric missing value imputation for mixed-type data". *Bioinformatics*, 28(1), 112-118.

- Chawla, N.V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique". *Journal of Artificial Intelligence Research*, 16, 321-357.

- Little, R.J.A., Rubin, D.B. (2002). *Statistical Analysis with Missing Data*. John Wiley & Sons.

---

**Note finale** : Ce cours constitue un guide exhaustif du preprocessing. Pour chaque projet ML, adapter ces étapes en fonction du contexte métier, du type de données et de l'algorithme cible. L'ordre des opérations et la prévention du data leakage sont les aspects les plus critiques pour garantir des modèles robustes en production.
