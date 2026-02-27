# Régression Logistique

> **Résumé en une phrase** : La régression logistique est un algorithme de classification supervisée qui modélise la probabilité d'appartenance à une classe via une fonction sigmoïde, avec support de la régularisation L1/L2 et extension multi-classes.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-02-27 |
| **Dernière mise à jour** | 2026-02-27 |
| **Domaine** | Machine Learning / Classification Supervisée |
| **Niveau** | Intermédiaire à Avancé |
| **Durée de lecture** | ~35 minutes |
| **Fichier** | `logistic_regression.md` |
| **Emplacement** | `/01_machine_learning/02_supervised/` |
| **Tags** | `#machine-learning` `#supervised-learning` `#classification` `#logistic-regression` |

### Prérequis

- [ ] Notions de base en algèbre linéaire (vecteurs, matrices, produit scalaire)
- [ ] Notions de probabilités (concepts de base)
- [ ] Python et NumPy (manipulation de tableaux)

### Cours connexes (Liens Zettelkasten)

- **Même niveau** : [[svm]], [[xgboost]], [[lightgbm]]
- **Fondamentaux** : [[preprocessing_data]], [[postprocessing_data]]

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Ce cours couvre la régression logistique de manière exhaustive : classification binaire et multi-classes, optimisation par gradient descent, régularisation L1/L2 pour éviter l'overfitting, et implémentation pratique. Vous comprendrez les mathématiques sous-jacentes, les choix de design, et maîtriserez l'implémentation with scikit-learn.

### Objectifs d'apprentissage

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Expliquer la fonction sigmoïde, la log-loss, et le rôle de la régularisation
2. **Appliquer** : Implémenter et utiliser la régression logistique pour des problèmes binaires et multi-classes
3. **Analyser** : Interpréter les coefficients et diagnostiquer l'overfitting
4. **Évaluer** : Optimiser les hyperparamètres (C, penalty, solver) et ajuster le seuil de décision

---

## 🔍 Contexte et Motivation

### Pourquoi la régression logistique ?

1. **Simplicité & Interprétabilité** : Coefficients directement explicables (crucial en médecine, finance, réglementation)
2. **Performance robuste** : Excellent compromis sur données linéairement séparables
3. **Fondation conceptuelle** : Base du deep learning (neurone = régression logistique)
4. **Rapidité** : Entraînement et inférence très efficaces

### Problème résolu

Prédire une classe binaire (0/1) ou multi-classes avec une **probabilité** associée, basé sur des features continues ou discrètes.

**Exemple** : Diagnostic médical (sain/malade), détection de spam, scoring de crédit, classification d'images (MNIST).

---

## 📚 Fondamentaux Théoriques

### 1. Classification Binaire

#### 1.1 Rappel : Régression Linéaire

La régression linéaire modélise $$y = \mathbf{w}^T \mathbf{x} + b$$ avec sortie $$\in (-\infty, +\infty)$$.

**Problème** : Pour classifier, nous voulons une probabilité $$\in [0, 1]$$.

#### 1.2 Fonction Sigmoïde

**Définition** :

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Propriétés clés** :
- Sortie : $$[0, 1]$$ (interprétable comme probabilité)
- $$\sigma(0) = 0.5$$ (point d'équilibre)
- $$\sigma(-z) = 1 - \sigma(z)$$ (symétrie)
- Dérivée : $$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$ (simplifie le gradient)

**Visualisation** :

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

z = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid(z), linewidth=2, label='σ(z)')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Seuil 0.5')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='z = 0')
plt.xlabel('z'); plt.ylabel('σ(z)'); plt.title('Fonction Sigmoïde')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

#### 1.3 Modèle de Régression Logistique Binaire

**Formulation** :

$$P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

**Où** :
- $$\mathbf{x}$$ : vecteur des features
- $$\mathbf{w}$$ : poids (coefficients à apprendre)
- $$b$$ : biais (intercept)
- $$z = \mathbf{w}^T \mathbf{x} + b$$ : score linéaire (logit)

**Règle de décision** :

$$\hat{y} = \begin{cases} 1 & \text{si } \sigma(z) \geq 0.5 \Leftrightarrow z \geq 0 \\ 0 & \text{sinon} \end{cases}$$

**Interprétation géométrique** : $$\mathbf{w}^T \mathbf{x} + b = 0$$ définit un hyperplan de décision.

#### 1.4 Fonction de Coût : Binary Cross-Entropy (Log-Loss)

**Formule** :

Pour un exemple $$(x, y)$$ :

$$\mathcal{L}(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

Pour le dataset complet ($$m$$ exemples) :

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Intuition** :
- Si $$y=1$$ et $$\hat{y}=1$$ → coût = 0 (parfait)
- Si $$y=1$$ et $$\hat{y}=0$$ → coût = $$\infty$$ (très mauvais)
- Pénalité exponentielle pour les erreurs confiantes

**Pourquoi cette fonction ?**
1. Dérive du **maximum de vraisemblance** (MLE)
2. **Fonction convexe** → un seul minimum global
3. Gradient simple à calculer

#### 1.5 Optimisation : Gradient Descent

**Algorithme** :

Initialiser $$\mathbf{w}, b$$

Répéter jusqu'à convergence :

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

**Gradients (formules simplifiées)** :

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Dérivation (pour comprendre)** :

En appliquant la règle de la chaîne et en utilisant $$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$, on obtient miraculeusement $$\hat{y} - y$$ !

**Variantes pratiques** :
- **Batch GD** : Gradient sur tout le dataset (lent mais stable)
- **SGD** : Gradient sur 1 exemple (rapide mais bruité)
- **Mini-Batch GD** : Compromis optimal (utilisé par défaut)

**Sources** :
- [Pattern Recognition and Machine Learning - Bishop (2006)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Chapitre 4.3
- [Gradient Descent - Andrew Ng](https://www.coursera.org/learn/machine-learning)

### 2. Régularisation : L1, L2, Elastic Net

#### 2.1 Problème : Overfitting

**Observation** : Avec beaucoup de features ou peu de données, le modèle peut **sur-apprendre** (overfitting) : performance parfaite sur le train, médiocre sur le test.

**Cause** : Poids $$\mathbf{w}$$ très grands, modèle trop sensible aux variations des features.

#### 2.2 Régularisation L2 (Ridge)

**Fonction de coût modifiée** :

$$J_{\text{L2}}(\mathbf{w}, b) = J(\mathbf{w}, b) + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

**Où** :
- $$\lambda$$ : force de régularisation (hyperparamètre)
- $$\sum w_j^2$$ : pénalité L2 (norme euclidienne au carré)

**Effet** :
- Pénalise les **poids grands** → shrinkage (réduction)
- Préfère des poids petits et distribués
- **Ne crée pas de sparsité** (tous les poids restent $$\neq 0$$)

**Quand utiliser** :
- Toutes les features sont potentiellement utiles
- Prévention de l'overfitting par défaut

#### 2.3 Régularisation L1 (Lasso)

**Fonction de coût modifiée** :

$$J_{\text{L1}}(\mathbf{w}, b) = J(\mathbf{w}, b) + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

**Où** :
- $$\sum |w_j|$$ : pénalité L1 (norme Manhattan)

**Effet** :
- Pénalise les poids en valeur absolue
- **Crée de la sparsité** : certains poids deviennent exactement 0
- **Feature selection automatique**

**Quand utiliser** :
- Beaucoup de features dont certaines sont inutiles
- Besoin d'interprétabilité (identifier les features importantes)

#### 2.4 Elastic Net

**Fonction de coût modifiée** :

$$J_{\text{ElasticNet}}(\mathbf{w}, b) = J(\mathbf{w}, b) + \lambda \left[ \frac{1-\rho}{2m} \sum w_j^2 + \frac{\rho}{m} \sum |w_j| \right]$$

**Où** :
- $$\rho \in [0, 1]$$ : ratio L1/L2 ($$\rho=0$$ → L2 pur, $$\rho=1$$ → L1 pur)

**Effet** : Combine avantages de L1 (sparsité) et L2 (stabilité)

**Quand utiliser** :
- Features corrélées
- Compromis entre feature selection et stabilité

#### 2.5 Hyperparamètre C dans scikit-learn

**Notation scikit-learn** : $$C = \frac{1}{\lambda}$$

- **C grand** (ex: 100) → peu de régularisation → risque d'overfitting
- **C petit** (ex: 0.01) → beaucoup de régularisation → risque d'underfitting

**Trouver le bon C** : Grid Search avec validation croisée

**Comparaison visuelle** :

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Données avec beaucoup de features (risque d'overfitting)
X, y = make_classification(n_samples=200, n_features=20, n_informative=5, 
                           n_redundant=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test de différentes valeurs de C
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores_l1, test_scores_l1 = [], []
train_scores_l2, test_scores_l2 = [], []

for C in C_values:
    # L1
    model_l1 = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=1000)
    model_l1.fit(X_train_scaled, y_train)
    train_scores_l1.append(model_l1.score(X_train_scaled, y_train))
    test_scores_l1.append(model_l1.score(X_test_scaled, y_test))
    
    # L2
    model_l2 = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
    model_l2.fit(X_train_scaled, y_train)
    train_scores_l2.append(model_l2.score(X_train_scaled, y_train))
    test_scores_l2.append(model_l2.score(X_test_scaled, y_test))

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# L2
ax1.plot(C_values, train_scores_l2, 'o-', label='Train', linewidth=2)
ax1.plot(C_values, test_scores_l2, 's-', label='Test', linewidth=2)
ax1.set_xscale('log')
ax1.set_xlabel('C (inverse de λ)', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Régularisation L2 (Ridge)', fontsize=13, fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

# L1
ax2.plot(C_values, train_scores_l1, 'o-', label='Train', linewidth=2, color='green')
ax2.plot(C_values, test_scores_l1, 's-', label='Test', linewidth=2, color='orange')
ax2.set_xscale('log')
ax2.set_xlabel('C (inverse de λ)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Régularisation L1 (Lasso)', fontsize=13, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

**Observation** : Gap entre train et test diminue avec régularisation → moins d'overfitting.

**Sources** :
- [Regularization - Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Chapitre 3
- [L1 and L2 Regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

### 3. Classification Multi-Classes

#### 3.1 Extension : One-vs-Rest (OvR)

**Principe** : Pour $$K$$ classes, entraîner $$K$$ classifieurs binaires :
- Classifieur 1 : Classe 1 vs (Classe 2, 3, ..., K)
- Classifieur 2 : Classe 2 vs (Classe 1, 3, ..., K)
- ...
- Classifieur K : Classe K vs (Classe 1, 2, ..., K-1)

**Prédiction** : Choisir la classe avec la **plus haute probabilité** :

$$\hat{y} = \arg\max_{k} P(y=k | \mathbf{x})$$

**Implémentation scikit-learn** : Automatique avec `multi_class='ovr'` (par défaut)

**Avantages** :
- Simple, efficace
- Fonctionne avec tout classifieur binaire

**Inconvénients** :
- $$K$$ modèles à entraîner (coût computationnel)
- Probabilités non calibrées entre classes

#### 3.2 Softmax Regression (Multinomial Logistic Regression)

**Principe** : Généralisation directe pour $$K$$ classes avec une seule optimisation.

**Formulation** :

Pour chaque classe $$k$$, on calcule un score linéaire :

$$z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$

La probabilité de la classe $$k$$ est donnée par la **fonction softmax** :

$$P(y=k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Propriétés softmax** :
- Sortie : $$[0, 1]$$ pour chaque classe
- $$\sum_{k=1}^{K} P(y=k | \mathbf{x}) = 1$$ (distribution de probabilité valide)
- Généralisation de la sigmoïde (pour $$K=2$$, softmax = sigmoïde)

**Fonction de coût : Cross-Entropy Multi-Classes**

$$J(\mathbf{W}, \mathbf{b}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \mathbb{1}_{y^{(i)}=k} \log(P(y^{(i)}=k | \mathbf{x}^{(i)}))$$

**Où** :
- $$\mathbb{1}_{y^{(i)}=k}$$ : indicatrice (1 si vrai label = $$k$$, 0 sinon)
- $$\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, ..., \mathbf{w}_K]$$ : matrice des poids

**Implémentation scikit-learn** : `multi_class='multinomial'` + solver compatible (lbfgs, saga)

**Avantages** :
- **Un seul modèle** (plus efficace)
- Probabilités mieux calibrées
- Interprétation plus directe

**Inconvénients** :
- Plus complexe mathématiquement
- Nécessite un solver spécifique

**Comparaison visuelle (MNIST simplifié)** :

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement MNIST simplifié (8x8 images de chiffres 0-9)
digits = load_digits()
X, y = digits.data, digits.target

# Split et normalisation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement avec Softmax (multinomial)
model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model_softmax.fit(X_train_scaled, y_train)

# Prédictions
y_pred = model_softmax.predict(X_test_scaled)

print("Classification Report (Softmax):")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Classe prédite', fontsize=12)
plt.ylabel('Classe vraie', fontsize=12)
plt.title('Matrice de Confusion - MNIST (10 classes)', fontsize=14, fontweight='bold')
plt.show()
```

**Tableau comparatif** :

| Critère | One-vs-Rest (OvR) | Softmax (Multinomial) |
|---------|-------------------|------------------------|
| **Nombre de modèles** | $$K$$ modèles binaires | 1 modèle multi-classes |
| **Temps d'entraînement** | $$\sim K \times$$ temps binaire | Plus rapide (1 optimisation) |
| **Calibration probabilités** | Moins bonne | Meilleure |
| **Solver requis** | Tous | lbfgs, saga, newton-cg |
| **Interprétabilité** | $$K$$ frontières binaires | 1 espace de décision |
| **Recommandation** | Petit $$K$$ (<5) ou solver limité | $$K$$ moyen/grand (≥5) |

**Sources** :
- [Softmax Regression - Deep Learning Book](https://www.deeplearningbook.org/) - Chapitre 6.2.2.3
- [Multi-class Classification - scikit-learn](https://scikit-learn.org/stable/modules/multiclass.html)

---

## 💻 Implémentation Pratique

### 1. Implémentation Binaire from Scratch

```python
"""
Régression Logistique Binaire from Scratch
Objectif : Comprendre chaque étape de l'algorithme
"""

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_=1.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.w = None
        self.b = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_loss(self, y_true, y_pred):
        m = len(y_true)
        eps = 1e-10
        loss = -1/m * np.sum(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        
        # Ajout régularisation
        if self.regularization == 'l2':
            loss += (self.lambda_ / (2*m)) * np.sum(self.w**2)
        elif self.regularization == 'l1':
            loss += (self.lambda_ / m) * np.sum(np.abs(self.w))
        
        return loss
    
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        
        for i in range(self.n_iter):
            # Forward
            z = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(z)
            
            # Gradients
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            # Ajout gradient régularisation
            if self.regularization == 'l2':
                dw += (self.lambda_ / m) * self.w
            elif self.regularization == 'l1':
                dw += (self.lambda_ / m) * np.sign(self.w)
            
            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 2. Utilisation avec scikit-learn (Production)

```python
"""
Pipeline complet : Classification binaire avec régularisation
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Données
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Grid Search pour trouver les meilleurs hyperparamètres
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear'],  # Compatible L1 et L2
    'clf__class_weight': [None, 'balanced']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Meilleurs paramètres:", grid.best_params_)
print(f"Meilleur ROC AUC (CV): {grid.best_score_:.4f}")

# Évaluation finale
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + classification_report(y_test, y_pred))
print(f"ROC AUC (Test): {roc_auc_score(y_test, y_proba):.4f}")

# Analyse des coefficients (après normalisation)
coefficients = best_model.named_steps['clf'].coef_[0]
print("\nTop 5 features les plus importantes (valeur absolue):")
top_indices = np.argsort(np.abs(coefficients))[-5:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {coefficients[idx]:+.4f}")
```

### 3. Classification Multi-Classes (MNIST)

```python
"""
Classification multi-classes avec Softmax Regression
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Données MNIST (8x8)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Softmax Regression (multinomial)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                          C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n" + classification_report(y_test, y_pred))

# Validation croisée pour robustesse
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualisation : exemples bien/mal classés
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Bien classés
correct_idx = np.where(y_pred == y_test)[0][:5]
for i, idx in enumerate(correct_idx):
    axes[0, i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    axes[0, i].set_title(f'Prédit: {y_pred[idx]}\nVrai: {y_test[idx]}', color='green')
    axes[0, i].axis('off')

# Mal classés
wrong_idx = np.where(y_pred != y_test)[0][:5]
for i, idx in enumerate(wrong_idx):
    axes[1, i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    axes[1, i].set_title(f'Prédit: {y_pred[idx]}\nVrai: {y_test[idx]}', color='red')
    axes[1, i].axis('off')

plt.suptitle('Exemples de prédictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 🔬 Exemple Complet : Diagnostic Médical

```python
"""
Cas réel : Prédiction du diabète avec gestion complète
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Simulation de données inspirées du dataset Pima Indians Diabetes
np.random.seed(42)
n_samples = 768

X = np.column_stack([
    np.random.randint(0, 20, n_samples),      # Grossesses
    np.random.normal(120, 30, n_samples),     # Glucose
    np.random.normal(70, 12, n_samples),      # Pression
    np.random.normal(20, 15, n_samples),      # Épaisseur peau
    np.random.normal(80, 115, n_samples),     # Insuline
    np.random.normal(32, 7, n_samples),       # IMC
    np.random.normal(0.5, 0.3, n_samples),    # Pedigree diabète
    np.random.randint(21, 81, n_samples),     # Âge
])

y = ((X[:, 1] > 130) | (X[:, 5] > 35)).astype(int)

feature_names = ['Grossesses', 'Glucose', 'Pression', 'EpaisseurPeau', 
                 'Insuline', 'IMC', 'Pedigree', 'Age']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)

# Pipeline avec Grid Search
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear'],
    'clf__class_weight': ['balanced']  # Important pour données médicales
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("=" * 70)
print("ÉVALUATION MODÈLE")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Sain', 'Diabétique']))
print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Importance des features
clf = best_model.named_steps['clf']
importance = list(zip(feature_names, clf.coef_[0]))
importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n" + "=" * 70)
print("IMPORTANCE DES FEATURES")
print("=" * 70)
for feature, coef in importance:
    direction = "↑ risque" if coef > 0 else "↓ risque"
    print(f"{feature:15s} : {coef:+.4f}  {direction}")

# Ajustement du seuil (contexte médical : minimiser faux négatifs)
thresholds = np.linspace(0.1, 0.9, 80)
recalls, precisions = [], []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
    fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
    fn = np.sum((y_pred_thresh == 0) & (y_test == 1))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    recalls.append(recall)
    precisions.append(precision)

# Seuil optimal (maximiser recall tout en gardant précision raisonnable)
f1_scores = 2 * (np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls) + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n{'=' * 70}")
print(f"SEUIL OPTIMAL : {optimal_threshold:.3f}")
print(f"Recall à ce seuil : {recalls[optimal_idx]:.2%} (détection de {recalls[optimal_idx]*100:.0f}% des diabétiques)")
print(f"Précision à ce seuil : {precisions[optimal_idx]:.2%}")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc_score(y_test, y_proba):.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('Courbe ROC', fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

# Précision vs Recall
ax2.plot(thresholds, precisions, label='Précision', linewidth=2)
ax2.plot(thresholds, recalls, label='Recall', linewidth=2)
ax2.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
ax2.set_xlabel('Seuil de décision'); ax2.set_ylabel('Score')
ax2.set_title('Impact du Seuil', fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreurs fréquentes

1. **Oublier la normalisation** → Convergence lente/impossible
   - **Solution** : Toujours utiliser `StandardScaler` dans un `Pipeline`

2. **Ignorer le déséquilibre des classes** → Biais vers classe majoritaire
   - **Solution** : `class_weight='balanced'` ou SMOTE

3. **Se fier uniquement à l'accuracy** → Métriques trompeuses si classes déséquilibrées
   - **Solution** : Analyser précision, recall, F1, ROC AUC

4. **Appliquer sur données non-linéaires** → Performance médiocre
   - **Solution** : Feature engineering (polynômes) ou autre algorithme (SVM RBF, RF, XGBoost)

5. **Interpréter coefficients sans normalisation** → Conclusions erronées
   - **Solution** : Normaliser AVANT d'interpréter

### ✅ Bonnes pratiques

1. **Utiliser un Pipeline** → Évite data leakage
2. **Validation croisée** → Estimation robuste de la performance
3. **Grid Search** → Optimisation automatique des hyperparamètres
4. **Analyser les probabilités** → Ajuster le seuil selon le contexte métier
5. **Feature selection avec L1** → Améliore interprétabilité et généralisation

**Sources** :
- [scikit-learn Best Practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

## 📝 Résumé Rapide

### Concepts Clés

| Concept | Formule | Usage |
|---------|---------|-------|
| **Sigmoïde** | $$\sigma(z) = \frac{1}{1 + e^{-z}}$$ | Transformation en probabilité |
| **Modèle binaire** | $$P(y=1) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$ | Classification 0/1 |
| **Log-Loss** | $$J = -\frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$ | Fonction de coût |
| **Régularisation L2** | $$+ \frac{\lambda}{2m} \sum w_j^2$$ | Shrinkage des poids |
| **Régularisation L1** | $$+ \frac{\lambda}{m} \sum \mid w_j \mid$$ | Sparsité (feature selection) |
| **Softmax** | $$P(y=k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$ | Multi-classes |

### Code Minimal

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Pipeline + Grid Search (best practice)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# Meilleur modèle
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
```

### Hyperparamètres clés

| Paramètre | Valeurs | Effet |
|-----------|---------|-------|
| **C** | 0.01 → 100 | ↓C = + régularisation |
| **penalty** | 'l1', 'l2', 'elasticnet' | L1→sparsité, L2→shrinkage |
| **solver** | 'lbfgs', 'liblinear', 'saga' | Algorithme d'optimisation |
| **class_weight** | None, 'balanced' | Gestion classes déséquilibrées |
| **multi_class** | 'ovr', 'multinomial' | Stratégie multi-classes |

### Quand utiliser

```
Régression Logistique si :
├─ Données linéairement séparables
├─ Besoin d'interprétabilité
├─ Classification binaire/multi-classes
├─ Rapidité cruciale
└─ Dataset petit/moyen (< 1M exemples)

Alternatives :
├─ Non-linéaire → SVM RBF, Random Forest, XGBoost
├─ Texte/NLP → Naive Bayes, BERT
└─ Images → CNN (Deep Learning)
```

---

## 🔗 Intégration Repository

### Fichiers à mettre à jour

1. **`01_machine_learning/02_supervised/README.md`**
   ```markdown
   ### Classification
   - [[logistic_regression]] - Régression Logistique (binaire, multi-classes, régularisation)
   - [[svm]] - Support Vector Machines
   - [[xgboost]] - Gradient Boosting
   ```

2. **Références croisées dans `svm.md`, `xgboost.md`, `lightgbm.md`**
   ```markdown
   ## Voir aussi
   - [[logistic_regression]] - Algorithme linéaire plus simple et interprétable
   ```

### Nom du fichier

- **Fichier** : `logistic_regression.md`
- **Emplacement** : `/01_machine_learning/02_supervised/`

---

## 🚀 Pour Aller Plus Loin

### Papers Académiques

1. **The Origins of Logistic Regression** - David Cox (1958)
   - [Journal of the Royal Statistical Society](https://www.jstor.org/stable/2983890)
   - 📌 Paper fondateur avec bases mathématiques rigoureuses

2. **Logistic Regression in Rare Events Data** - King & Zeng (2001)
   - [PDF](https://gking.harvard.edu/files/0s.pdf)
   - 📌 Biais sur classes déséquilibrées et corrections

3. **A Comparison of Numerical Optimizers** - Komarek & Moore (2005)
   - [CMU Technical Report](https://www.cs.cmu.edu/~awm/papers/logreg.pdf)
   - 📌 Comparaison détaillée des solveurs (lbfgs, liblinear, etc.)

### Ressources Pratiques

- **Machine Learning (Coursera)** - Andrew Ng : [Cours](https://www.coursera.org/learn/machine-learning) - Semaine 3
- **StatQuest: Logistic Regression** : [Video](https://www.youtube.com/watch?v=yIYKR4sgzI8) - Visualisations excellentes
- **scikit-learn Documentation** : [Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### Outils

- **scikit-learn** : `pip install scikit-learn`
- **imbalanced-learn** : `pip install imbalanced-learn` (SMOTE)
- **SHAP** : `pip install shap` (interprétation avancée)

---

*Dernière mise à jour : 2026-02-27*  
*Cours optimisé selon Cognitive Load Theory et Multimedia Learning*
