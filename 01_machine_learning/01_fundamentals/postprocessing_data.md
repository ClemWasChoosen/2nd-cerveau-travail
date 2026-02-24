# Post-Processing des Modèles Machine Learning et Deep Learning

> **Guide complet pour valider, analyser et décider du déploiement de modèles ML/DL**
> 
> *Dernière mise à jour : Février 2026*

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Métriques d'Évaluation](#2-métriques-dévaluation)
   - 2.1 [Classification](#21-classification)
   - 2.2 [Régression](#22-régression)
3. [Diagnostic du Modèle](#3-diagnostic-du-modèle)
   - 3.1 [Overfitting et Underfitting](#31-overfitting-et-underfitting)
   - 3.2 [Learning Curves](#32-learning-curves)
   - 3.3 [Bias-Variance Tradeoff](#33-bias-variance-tradeoff)
4. [Analyse des Erreurs](#4-analyse-des-erreurs)
   - 4.1 [Analyse des Résidus (Régression)](#41-analyse-des-résidus-régression)
   - 4.2 [Matrice de Confusion (Classification)](#42-matrice-de-confusion-classification)
5. [Tests Statistiques](#5-tests-statistiques)
6. [Calibration des Modèles](#6-calibration-des-modèles)
7. [Interprétabilité et Explicabilité](#7-interprétabilité-et-explicabilité)
8. [Robustesse et Tests Adverses](#8-robustesse-et-tests-adverses)
9. [Analyse de Biais et Fairness](#9-analyse-de-biais-et-fairness)
10. [Checklist de Déploiement](#10-checklist-de-déploiement)
11. [Références](#11-références)

---

## 1. Introduction

### Pourquoi le post-processing est-il crucial ?

Le post-processing est l'étape **la plus critique** pour déterminer si un modèle est réellement utilisable en production. Un modèle avec d'excellentes métriques sur le jeu de test peut être totalement inadapté dans la réalité pour plusieurs raisons :

- **Overfitting** : Le modèle a mémorisé les données d'entraînement
- **Biais cachés** : Le modèle discrimine certaines populations
- **Mauvaise calibration** : Les probabilités prédites ne reflètent pas la réalité
- **Fragilité** : Le modèle est sensible aux perturbations
- **Non-interprétabilité** : Impossible de comprendre les décisions

### Objectifs du post-processing

1. ✅ **Valider** que le modèle généralise bien
2. 📊 **Quantifier** la performance de manière objective
3. 🔍 **Diagnostiquer** les problèmes (overfitting, biais, etc.)
4. 🎯 **Optimiser** les seuils de décision
5. 🛡️ **Garantir** la robustesse et l'équité
6. 🚀 **Décider** si le modèle est prêt pour la production

---

## 2. Métriques d'Évaluation

### 2.1 Classification

#### 2.1.1 Accuracy

**Formule mathématique :**

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

**Interprétation :**

Pourcentage de prédictions correctes sur l'ensemble des prédictions.

**Pourquoi l'utiliser ?**

- Très intuitive et facile à expliquer
- Utile quand les classes sont équilibrées

**Quand NE PAS l'utiliser ?**

- ⚠️ **Datasets déséquilibrés** : Si 95% de vos données sont de classe 0, un modèle qui prédit toujours 0 aura 95% d'accuracy !

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Exemple : dataset déséquilibré (95% de classe 0)
y_true = np.array([0]*95 + [1]*5)
y_pred_naive = np.array([0]*100)  # Modèle qui prédit toujours 0
y_pred_smart = np.array([0]*90 + [1]*10)  # Modèle "intelligent"

print(f"Accuracy modèle naïf: {accuracy_score(y_true, y_pred_naive):.2f}")  # 0.95
print(f"Accuracy modèle intelligent: {accuracy_score(y_true, y_pred_smart):.2f}")  # 0.90

# Le modèle naïf a une meilleure accuracy mais est inutile !
```

**Sources :**
- :cite[n1k] *A review of model evaluation metrics for ML in genetics and genomics* (PMC, 2024)
- :cite[ekx] *Evaluation metrics and statistical tests for ML* (Nature, 2024)

---

#### 2.1.2 Precision

**Formule mathématique :**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Interprétation :**

Parmi toutes les prédictions positives, quelle proportion est correcte ?

**Pourquoi l'utiliser ?**

- Quand les **faux positifs sont coûteux**
- Exemple : Détection de spam (on préfère laisser passer un spam que bloquer un email important)

```python
from sklearn.metrics import precision_score, recall_score

# Exemple : Détection de maladie rare
y_true = np.array([0]*90 + [1]*10)  # 10% de malades
y_pred_conservative = np.array([0]*80 + [1]*20)  # Modèle qui prédit beaucoup de malades

# Calcul
precision = precision_score(y_true, y_pred_conservative)
recall = recall_score(y_true, y_pred_conservative)

print(f"Precision: {precision:.2f}")  # Faible si beaucoup de FP
print(f"Recall: {recall:.2f}")  # Élevé car on détecte beaucoup de vrais malades
```

---

#### 2.1.3 Recall (Sensibilité)

**Formule mathématique :**

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Interprétation :**

Parmi tous les vrais positifs, quelle proportion a été détectée ?

**Pourquoi l'utiliser ?**

- Quand les **faux négatifs sont coûteux**
- Exemple : Détection de cancer (on préfère un faux positif qu'un faux négatif)

**Pourquoi cette différence Precision vs Recall ?**

- **Precision** : "Quand je dis OUI, ai-je raison ?"
- **Recall** : "Est-ce que je détecte tous les OUI ?"

---

#### 2.1.4 F1-Score

**Formule mathématique :**

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interprétation :**

Moyenne harmonique entre Precision et Recall.

**Pourquoi l'utiliser ?**

- Quand on veut un **équilibre** entre Precision et Recall
- Mieux que l'accuracy pour les datasets déséquilibrés
- ⚠️ Mais ignore les True Negatives !

```python
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Exemple : Comparaison de modèles sur dataset déséquilibré
X = np.random.randn(1000, 10)
y = np.array([0]*900 + [1]*100)  # 10% de classe 1

# Modèle
clf = RandomForestClassifier(random_state=42)

# Évaluation avec différentes métriques
acc_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')

print(f"Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
print(f"F1-Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
```

---

#### 2.1.5 ROC-AUC (Area Under the ROC Curve)

**Formule mathématique :**

La courbe ROC trace le **True Positive Rate (TPR)** contre le **False Positive Rate (FPR)** pour différents seuils de décision :

$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

**Interprétation :**

- **AUC = 0.5** : Modèle aléatoire
- **AUC = 1.0** : Modèle parfait
- **AUC > 0.8** : Généralement considéré comme bon

**Pourquoi l'utiliser ?**

- **Indépendant du seuil** de décision
- Permet de comparer des modèles objectivement
- Visualise le trade-off TPR vs FPR

**Quand NE PAS l'utiliser ?**

- Datasets **très déséquilibrés** → Préférer PR-AUC (Precision-Recall AUC)
- Quand les **coûts FP et FN sont très différents**

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Générer des prédictions
y_true = np.array([0]*80 + [1]*20)
y_scores = np.random.rand(100)

# Calculer ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Visualisation
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

#### 2.1.6 Matthews Correlation Coefficient (MCC)

**Formule mathématique :**

$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

**Interprétation :**

- **MCC = +1** : Prédiction parfaite
- **MCC = 0** : Prédiction aléatoire
- **MCC = -1** : Désaccord total

**Pourquoi l'utiliser ?**

- ⭐ **Meilleure métrique pour datasets déséquilibrés** que F1 ou Accuracy :cite[n1k]
- Prend en compte **tous les éléments** de la matrice de confusion
- Symétrique (traite les classes de manière égale)

**Sources :**
- :cite[n1k] Chicco & Jurman (2020) - *"MCC should replace ROC-AUC for binary classification"*

---

### 2.2 Régression

#### 2.2.1 Mean Absolute Error (MAE)

**Formule mathématique :**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interprétation :**

Erreur moyenne absolue en unités de la variable cible.

**Pourquoi l'utiliser ?**

- **Facile à interpréter** (même unité que la cible)
- **Robuste aux outliers** (pas de carré)
- Donne le même poids à toutes les erreurs

```python
from sklearn.metrics import mean_absolute_error

# Exemple : Prédiction du prix d'une maison
y_true = np.array([100000, 200000, 300000, 150000, 250000])  # Prix réels
y_pred = np.array([95000, 210000, 280000, 160000, 240000])   # Prix prédits

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: ${mae:,.0f}")  # Erreur moyenne en dollars
```

---

#### 2.2.2 Root Mean Squared Error (RMSE)

**Formule mathématique :**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interprétation :**

Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**

- Pénalise davantage les **grandes erreurs** (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

```python
import numpy as np

# Exemple de prédictions
y_true = np.array([100, 100, 100, 100, 100])
y_pred1 = np.array([105, 105, 105, 105, 105])  # Erreurs constantes
y_pred2 = np.array([100, 100, 100, 100, 125])  # Une grosse erreur

mae1 = np.mean(np.abs(y_true - y_pred1))
rmse1 = np.sqrt(np.mean((y_true - y_pred1)**2))
print(f"Cas 1 - MAE: {mae1:.2f}, RMSE: {rmse1:.2f}")  # MAE: 5.00, RMSE: 5.00

mae2 = np.mean(np.abs(y_true - y_pred2))
rmse2 = np.sqrt(np.mean((y_true - y_pred2)**2))
print(f"Cas 2 - MAE: {mae2:.2f}, RMSE: {rmse2:.2f}")  # MAE: 5.00, RMSE: 11.18

# Même MAE mais RMSE plus élevé car une grosse erreur
```

**Quand préférer RMSE à MAE ?**

- Quand les grandes erreurs sont **inacceptables**
- Quand on veut être **conservateur**

**Quand préférer MAE à RMSE ?**

- Quand on veut une métrique **robuste aux outliers**
- Quand toutes les erreurs ont le même coût

---

#### 2.2.3 R² (Coefficient de détermination)

**Formule mathématique :**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Où $$\bar{y}$$ est la moyenne de $$y$$.

**Interprétation :**

- **R² = 1** : Modèle parfait (explique 100% de la variance)
- **R² = 0** : Modèle aussi bon qu'une moyenne
- **R² < 0** : Modèle pire qu'une moyenne (!)

**Pourquoi l'utiliser ?**

- **Sans unité** → Facilite la comparaison entre datasets
- Mesure la **proportion de variance expliquée**

**Quand NE PAS l'utiliser ?**

- ⚠️ Augmente artificiellement quand on ajoute des features (même inutiles) → Utiliser **R² ajusté** :

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Où $$n$$ = nombre d'observations, $$p$$ = nombre de features.

```python
from sklearn.metrics import r2_score

y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 210, 240, 290])

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Calcul de R² ajusté
n = len(y_true)
p = 5  # Nombre de features
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R² ajusté: {r2_adj:.3f}")
```

---

## 3. Diagnostic du Modèle

### 3.1 Overfitting et Underfitting

#### Définitions

**Overfitting (sur-apprentissage) :**

Le modèle **mémorise** les données d'entraînement au lieu d'apprendre des patterns généralisables.

- Performance train : ⭐⭐⭐⭐⭐
- Performance test : ⭐⭐

**Underfitting (sous-apprentissage) :**

Le modèle est **trop simple** pour capturer les patterns dans les données.

- Performance train : ⭐⭐
- Performance test : ⭐⭐

#### Pourquoi ça arrive ?

**Causes de l'overfitting :**

1. **Modèle trop complexe** pour les données (ex: ResNet-152 sur MNIST)
2. **Pas assez de données**
3. **Pas de régularisation**
4. **Entraînement trop long**

**Causes de l'underfitting :**

1. **Modèle trop simple** (ex: régression linéaire sur relation non-linéaire)
2. **Features insuffisantes**
3. **Régularisation trop forte**
4. **Pas assez d'entraînement**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Générer des données non-linéaires
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test de différents degrés polynomiaux
degrees = [1, 4, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees, 1):
    # Transformation polynomiale
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Entraînement
    model = Ridge(alpha=0.001)
    model.fit(X_train_poly, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Scores
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Visualisation
    plt.subplot(1, 3, i)
    plt.scatter(X_train, y_train, color='blue', s=30, label='Train')
    plt.scatter(X_test, y_test, color='red', s=30, label='Test')
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, color='green', linewidth=2, label='Model')
    plt.title(f'Degree {degree}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
    plt.legend()

plt.tight_layout()
plt.show()

# Degree 1: Underfitting (train et test MSE élevés)
# Degree 4: Good fit (train et test MSE faibles et similaires)
# Degree 15: Overfitting (train MSE très faible, test MSE élevé)
```

---

### 3.2 Learning Curves

**Définition :**

Les learning curves tracent la performance (loss ou score) sur les données d'entraînement et de validation en fonction de la taille du dataset ou du nombre d'époques.

**Pourquoi les utiliser ?**

- **Diagnostiquer overfitting vs underfitting** de manière visuelle
- **Décider** si on a besoin de plus de données
- **Optimiser** le nombre d'époques d'entraînement

#### 3.2.1 Good Fit

**Caractéristiques :**

- Training loss **décroît progressivement**
- Validation loss **décroît progressivement**
- Les deux courbes **convergent** vers une valeur stable
- **Gap faible** entre train et validation

```python
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Générer des données
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)

# Modèle
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

# Calculer learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Moyennes et écarts-types
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Good Fit')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

---

#### 3.2.2 Overfitting

**Caractéristiques :**

- Training loss **très faible et continue de baisser**
- Validation loss **se stabilise ou augmente**
- **Large gap** entre train et validation
- ⚠️ **Signe qu'il faut plus de données ou de régularisation**

```python
# Modèle overfit (trop de profondeur, pas de régularisation)
model_overfit = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                       min_samples_split=2, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    model_overfit, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Overfitting')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Gap important entre train (très haut) et validation (plus bas)
```

---

#### 3.2.3 Underfitting

**Caractéristiques :**

- Training loss **reste élevée**
- Validation loss **reste élevée**
- **Faible gap** entre train et validation (mais les deux sont mauvaises)
- ⚠️ **Signe qu'il faut un modèle plus complexe**

**Source :**
- :cite[g8q] *Learning Curve to identify Overfitting and Underfitting* (Towards Data Science, 2021)

---

### 3.3 Bias-Variance Tradeoff

**Définition :**

L'erreur totale d'un modèle se décompose en :

$$\text{Erreur totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit irréductible}$$

- **Biais** : Erreur due à des hypothèses simplificatrices (underfitting)
- **Variance** : Erreur due à la sensibilité aux fluctuations des données (overfitting)

**Pourquoi c'est important ?**

Comprendre cette décomposition permet de savoir **quelle direction prendre** pour améliorer le modèle :

- **Biais élevé** → Modèle trop simple → Augmenter la complexité
- **Variance élevée** → Modèle trop complexe → Régulariser ou ajouter des données

```python
# Simulation du bias-variance tradeoff
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Tester différents modèles
models = {
    'Low Complexity (Bias élevé)': DecisionTreeRegressor(max_depth=2),
    'Medium Complexity': DecisionTreeRegressor(max_depth=5),
    'High Complexity (Variance élevée)': DecisionTreeRegressor(max_depth=None)
}

for name, model in models.items():
    # Entraîner sur plusieurs échantillons bootstrap
    mse_list = []
    for _ in range(50):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X)
        mse_list.append(mean_squared_error(y, y_pred))
    
    print(f"{name}:")
    print(f"  MSE moyen: {np.mean(mse_list):.4f}")
    print(f"  Variance MSE: {np.var(mse_list):.4f}")
    print()
```

**Source :**
- :cite[dba] *Diagnosing Bias and Variance in ML Models* (Medium, 2024)
- :cite[cx6] *Bias-Variance Tradeoff* (IBM)

---

## 4. Analyse des Erreurs

### 4.1 Analyse des Résidus (Régression)

**Définition :**

Les résidus sont la différence entre les valeurs réelles et prédites :

$$\text{Résidu}_i = y_i - \hat{y}_i$$

**Pourquoi analyser les résidus ?**

- Vérifier les **hypothèses du modèle** (linéarité, homoscédasticité, normalité)
- Détecter les **outliers**
- Identifier des **patterns non capturés**

#### 4.1.1 Residual Plot (Résidus vs Prédictions)

**Bon modèle :**
- Résidus **distribués aléatoirement** autour de 0
- Pas de pattern visible

**Problèmes détectables :**
- **Pattern en courbe** → Relation non-linéaire non capturée
- **Pattern en entonnoir** → Hétéroscédasticité (variance non constante)
- **Points éloignés** → Outliers

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Générer des données avec relation non-linéaire
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.ravel() + X.ravel()**2 + np.random.randn(100) * 5

# Modèle linéaire (inadapté)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calcul des résidus
residuals = y - y_pred

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Résidus vs Prédictions
axes[0].scatter(y_pred, residuals, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot - Pattern visible (non-linéarité)')

# QQ-plot (normalité des résidus)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

**Source :**
- :cite[bl4] *Understanding Residual Analysis in Regression* (Medium, 2024)
- :cite[b2w] *Diagnostic Plots for Linear Regression* (UVA Library, 2024)

---

### 4.2 Matrice de Confusion (Classification)

**Définition :**

Tableau qui résume les prédictions d'un modèle de classification :

|                     | **Prédit Positif** | **Prédit Négatif** |
|---------------------|--------------------|--------------------|
| **Réel Positif**    | TP (True Positive) | FN (False Negative)|
| **Réel Négatif**    | FP (False Positive)| TN (True Negative) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Données
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print("\nAnalyse de la matrice:")
print(f"True Positives (TP): {cm[1,1]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Negatives (FN): {cm[1,0]}")
```

**Pourquoi c'est crucial ?**

La matrice de confusion révèle **où** le modèle se trompe :

- **FN élevés** → Problème de Recall → Modèle manque des cas positifs
- **FP élevés** → Problème de Precision → Modèle sur-détecte des positifs

---

## 5. Tests Statistiques

### Pourquoi des tests statistiques ?

Pour savoir si la différence de performance entre deux modèles est **statistiquement significative** ou due au **hasard**.

### 5.1 Paired t-test

**Quand l'utiliser ?**

- Comparer **deux modèles** sur le **même dataset**
- Assumer que les différences suivent une **distribution normale**

⚠️ **Limitations :**
- Sensible aux **outliers**
- Inadapté pour données **non-normales**
- Ne pas utiliser avec **resampling** (ex: CV répété)

```python
from scipy.stats import ttest_rel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X, y = make_classification(n_samples=500, random_state=42)

# Deux modèles
model1 = RandomForestClassifier(random_state=42)
model2 = GradientBoostingClassifier(random_state=42)

# Scores sur 10 folds
scores1 = cross_val_score(model1, X, y, cv=10, scoring='accuracy')
scores2 = cross_val_score(model2, X, y, cv=10, scoring='accuracy')

# t-test
statistic, p_value = ttest_rel(scores1, scores2)

print(f"Model 1 accuracy: {scores1.mean():.3f} ± {scores1.std():.3f}")
print(f"Model 2 accuracy: {scores2.mean():.3f} ± {scores2.std():.3f}")
print(f"t-statistic: {statistic:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Différence significative (p < 0.05)")
else:
    print("❌ Pas de différence significative")
```

---

### 5.2 Wilcoxon Signed-Rank Test

**Pourquoi préférer Wilcoxon au t-test ?**

- **Non-paramétrique** → Pas d'hypothèse de normalité
- **Plus robuste** aux outliers
- ⭐ **Recommandé par la littérature** pour comparer des modèles ML :cite[ekx]

```python
from scipy.stats import wilcoxon

# Même exemple que précédemment
statistic, p_value = wilcoxon(scores1, scores2)

print(f"Wilcoxon statistic: {statistic:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Différence significative (p < 0.05)")
else:
    print("❌ Pas de différence significative")
```

---

### 5.3 McNemar's Test

**Quand l'utiliser ?**

- Comparer **deux classificateurs** sur un **seul test set**
- Teste si les **patterns d'erreurs** sont différents

```python
from statsmodels.stats.contingency_tables import mcnemar

# Prédictions de deux modèles sur le même test set
y_pred1 = clf.predict(X_test)
y_pred2 = model2.fit(X_train, y_train).predict(X_test)

# Construire la table de contingence
n00 = np.sum((y_pred1 == y_test) & (y_pred2 == y_test))  # Les deux corrects
n01 = np.sum((y_pred1 == y_test) & (y_pred2 != y_test))  # Seul model1 correct
n10 = np.sum((y_pred1 != y_test) & (y_pred2 == y_test))  # Seul model2 correct
n11 = np.sum((y_pred1 != y_test) & (y_pred2 != y_test))  # Les deux incorrects

table = [[n00, n01], [n10, n11]]
result = mcnemar(table, exact=True)

print(f"McNemar statistic: {result.statistic:.3f}, p-value: {result.pvalue:.4f}")
```

**Source :**
- :cite[bn1] *Statistical Significance Tests for ML Algorithms* (Machine Learning Mastery, 2024)
- :cite[ekx] *Evaluation metrics and statistical tests* (Nature, 2024)

---

## 6. Calibration des Modèles

### Qu'est-ce que la calibration ?

Un modèle **bien calibré** est un modèle dont les probabilités prédites reflètent les vraies probabilités.

**Exemple :**
- Si le modèle prédit 80% de probabilité de pluie sur 100 jours
- → Il **devrait pleuvoir** environ **80 jours**

**Pourquoi c'est important ?**

- Décisions médicales (seuils de risque)
- Systèmes de scoring de crédit
- Toute application où on utilise `predict_proba`

### 6.1 Calibration Curve

```python
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Données
X, y = make_classification(n_samples=10000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Deux modèles
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42)

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Calibration curves
fig, ax = plt.subplots(figsize=(10, 6))
CalibrationDisplay.from_estimator(rf, X_test, y_test, n_bins=10, name='Random Forest', ax=ax)
CalibrationDisplay.from_estimator(lr, X_test, y_test, n_bins=10, name='Logistic Regression', ax=ax)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax.set_title('Calibration Curves')
ax.legend()
plt.show()
```

**Interprétation :**

- **Sur la diagonale** → Modèle bien calibré
- **Sous la diagonale** → Modèle sur-confiant (prédit des probas trop hautes)
- **Au-dessus de la diagonale** → Modèle sous-confiant

---

### 6.2 Calibration avec CalibratedClassifierCV

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrer un Random Forest
rf_calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
rf_calibrated.fit(X_train, y_train)

# Comparer avant/après
fig, ax = plt.subplots(figsize=(10, 6))
CalibrationDisplay.from_estimator(rf, X_test, y_test, n_bins=10, name='RF (non calibré)', ax=ax)
CalibrationDisplay.from_estimator(rf_calibrated, X_test, y_test, n_bins=10, name='RF (calibré)', ax=ax)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax.legend()
plt.show()
```

**Méthodes de calibration :**

1. **Sigmoid (Platt Scaling)** : Ajuste une régression logistique sur les probabilités
2. **Isotonic Regression** : Ajuste une fonction monotone non-paramétrique

**Pourquoi calibrer ?**

- Random Forests ont tendance à être **sous-confiants** (probas trop proches de 0.5)
- SVMs sans calibration donnent des sorties **non calibrées**

**Source :**
- :cite[eks] *Probability Calibration* (Scikit-learn, 2024)
- :cite[bpe] *Probability Calibration Tutorial* (Kaggle, 2024)

---

## 7. Interprétabilité et Explicabilité

### Pourquoi l'interprétabilité est cruciale ?

- **Confiance** : Comprendre pourquoi le modèle prédit X
- **Débogage** : Identifier les features problématiques
- **Réglementaire** : RGPD, HIPAA exigent l'explicabilité
- **Fairness** : Détecter les biais

### 7.1 Feature Importance (Global)

```python
from sklearn.inspection import permutation_importance

# Modèle
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature importance (built-in)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance (Gini)')
plt.show()

# Permutation importance (plus fiable)
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.boxplot([perm_importance.importances[i] for i in sorted_idx], vert=False)
plt.yticks(range(1, len(sorted_idx) + 1), sorted_idx)
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.show()
```

---

### 7.2 SHAP (SHapley Additive exPlanations)

**Principe :**

SHAP attribue à chaque feature une **valeur de contribution** pour une prédiction donnée, basée sur la théorie des jeux coopératifs.

```python
import shap

# Expliquer les prédictions avec SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:100])

# Visualisation globale
shap.summary_plot(shap_values[1], X_test[:100], plot_type="bar")

# Visualisation pour une prédiction
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0])
```

**Pourquoi SHAP ?**

- ⭐ **Théoriquement fondé** (propriétés de Shapley)
- **Local et global** : Explique une prédiction ET le modèle
- **Applicable** à tous les modèles (même deep learning)

**Source :**
- :cite[sil] *Explainable AI with LIME & SHAP* (DataCamp, 2024)
- :cite[aqj] *Mastering Explainable AI* (Medium, 2024)

---

### 7.3 LIME (Local Interpretable Model-agnostic Explanations)

**Principe :**

LIME explique une **prédiction locale** en ajustant un modèle simple (ex: linéaire) autour de l'instance à expliquer.

```python
from lime import lime_tabular

# Créer un explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names=[f'Feature {i}' for i in range(X_train.shape[1])],
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Expliquer une prédiction
exp = explainer.explain_instance(X_test[0], rf.predict_proba, num_features=10)
exp.show_in_notebook()
```

**LIME vs SHAP :**

| Critère | LIME | SHAP |
|---------|------|------|
| **Théorie** | Heuristique | Fondé (Shapley) |
| **Vitesse** | Rapide | Plus lent |
| **Cohérence** | Parfois instable | Stable |
| **Usage** | Exploration rapide | Analyse approfondie |

---

## 8. Robustesse et Tests Adverses

### 8.1 Qu'est-ce que la robustesse ?

Un modèle **robuste** maintient sa performance face à :

- **Perturbations** des données (bruit, outliers)
- **Attaques adversariales** (modifications malveillantes)
- **Distribution shift** (données différentes de l'entraînement)

### 8.2 Adversarial Examples

```python
# Exemple simple d'attaque FGSM (Fast Gradient Sign Method)
import torch
import torch.nn as nn
import torch.optim as optim

# Modèle simple
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 2)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()

# Générer un exemple adversarial
X_torch = torch.tensor(X_test[:1], dtype=torch.float32, requires_grad=True)
y_torch = torch.tensor(y_test[:1], dtype=torch.long)

# Prédiction originale
output = model(X_torch)
pred_original = output.argmax().item()

# Calcul du gradient
loss = criterion(output, y_torch)
loss.backward()

# Perturbation adversariale (epsilon = 0.1)
epsilon = 0.1
perturbation = epsilon * X_torch.grad.sign()
X_adv = X_torch + perturbation

# Prédiction adversariale
output_adv = model(X_adv)
pred_adv = output_adv.argmax().item()

print(f"Prédiction originale: {pred_original}")
print(f"Prédiction adversariale: {pred_adv}")
print(f"Perturbation max: {perturbation.abs().max().item():.4f}")
```

**Source :**
- :cite[drk] *Robustness and Distribution Shift* (Data102, 2020)
- :cite[bs0] *Adversarial Examples Discussion* (Distill, 2019)

---

## 9. Analyse de Biais et Fairness

### 9.1 Qu'est-ce que le biais algorithmique ?

Un modèle est **biaisé** si ses performances ou décisions varient de manière injuste entre différents groupes démographiques (genre, ethnie, âge, etc.).

### 9.2 Métriques de Fairness

```python
from sklearn.metrics import confusion_matrix

# Supposons que nous avons un attribut sensible (ex: genre)
# 0 = Femme, 1 = Homme
sensitive_attr = np.random.binomial(1, 0.5, len(y_test))

# Prédictions
y_pred = clf.predict(X_test)

# Métriques par groupe
for group in [0, 1]:
    mask = sensitive_attr == group
    y_true_group = y_test[mask]
    y_pred_group = y_pred[mask]
    
    cm = confusion_matrix(y_true_group, y_pred_group)
    tpr = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    fpr = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
    
    print(f"Groupe {group}:")
    print(f"  True Positive Rate (TPR): {tpr:.3f}")
    print(f"  False Positive Rate (FPR): {fpr:.3f}")
    print()
```

**Métriques de Fairness :**

1. **Demographic Parity** : P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
2. **Equalized Odds** : TPR et FPR égaux entre groupes
3. **Equal Opportunity** : TPR égaux entre groupes

**Source :**
- :cite[ctt] *Algorithmic Bias Examples and Tools* (Arize, 2024)
- :cite[ekh] *Fairness and Bias in ML* (Lumenova, 2024)

---

## 10. Checklist de Déploiement

### 10.1 Performance

- [ ] **Métriques** calculées sur un test set **réellement indépendant**
- [ ] **Comparaison** avec baseline (modèle simple, règle métier)
- [ ] **Tests statistiques** pour confirmer la supériorité du modèle
- [ ] **Calibration** vérifiée (si probabilités utilisées)

### 10.2 Robustesse

- [ ] **Learning curves** analysées (pas d'overfitting)
- [ ] **Validation croisée** avec plusieurs splits
- [ ] **Tests sur données OOD** (Out-of-Distribution)
- [ ] **Sensibilité aux outliers** testée

### 10.3 Interprétabilité

- [ ] **Feature importance** analysée
- [ ] **SHAP/LIME** pour comprendre les prédictions
- [ ] **Documentation** des décisions du modèle

### 10.4 Fairness

- [ ] **Métriques de fairness** calculées par groupe
- [ ] **Biais démographiques** identifiés et documentés
- [ ] **Stratégies de mitigation** implémentées si nécessaire

### 10.5 Production

- [ ] **Latence** acceptable pour le use case
- [ ] **Taille du modèle** compatible avec l'infrastructure
- [ ] **Monitoring** des performances en production défini
- [ ] **Stratégie de retraining** établie

**Source :**
- :cite[csf] *ML Production Readiness Checklist* (Medium, 2024)
- :cite[a18] *Rubric for ML Production Readiness* (Google Research, PDF)

---

## 11. Références

### Articles Scientifiques

- :cite[ekx] Rainio et al. (2024) - *Evaluation metrics and statistical tests for machine learning*. Nature Scientific Reports.
- :cite[n1k] Miller et al. (2024) - *A review of model evaluation metrics for ML in genetics and genomics*. PMC.
- :cite[eks] Scikit-learn (2024) - *Probability calibration*. Official Documentation.

### Ressources Pratiques

- :cite[g8q] Muralidhar (2021) - *Learning Curve to identify Overfitting and Underfitting*. Towards Data Science.
- :cite[sil] DataCamp (2024) - *Explainable AI with LIME and SHAP*.
- :cite[bn1] Machine Learning Mastery (2024) - *Statistical Significance Tests for ML*.

### Livres et Cours

- Andrew Ng - *Machine Learning Specialization* (Coursera)
- Hastie, Tibshirani & Friedman - *The Elements of Statistical Learning*
- Molnar, Christoph - *Interpretable Machine Learning*

---

## Conclusion

Le post-processing n'est **pas une étape optionnelle**, c'est **la garantie** que votre modèle est :

1. ✅ **Fiable** : Généralise bien, pas d'overfitting
2. 📊 **Performant** : Métriques adaptées au problème
3. 🔍 **Compréhensible** : Interprétable et explicable
4. 🛡️ **Équitable** : Pas de biais discriminatoires
5. 🚀 **Déployable** : Robuste et prêt pour la production

**Dernier conseil :**

> *"Un bon data scientist passe 20% de son temps à entraîner des modèles, et 80% à les valider, analyser et comprendre."*

Ne te précipite jamais vers le déploiement. Prends le temps d'analyser **en profondeur** chaque aspect de ton modèle. C'est cette rigueur qui fait la différence entre un modèle "qui marche sur Kaggle" et un modèle "prêt pour la production".

---

**Note finale :** Ce cours est un guide complet. N'hésite pas à revenir régulièrement sur les sections pertinentes pour ton cas d'usage. Chaque projet est unique et nécessite une adaptation de ces méthodes.
