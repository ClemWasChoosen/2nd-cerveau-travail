# XGBoost : Guide Complet et Détaillé

## Table des matières
1. [Introduction et Contexte](#introduction)
2. [Concepts Préliminaires](#concepts-préliminaires)
3. [Fonctionnement Mathématique de XGBoost](#fonctionnement-mathématique)
4. [Hyperparamètres Fondamentaux](#hyperparamètres)
5. [Exemples Pratiques](#exemples-pratiques)
6. [Production et Entreprise](#production)
7. [Ressources et Références](#ressources)

---

## 1. Introduction et Contexte {#introduction}

### Qu'est-ce que XGBoost ?

**XGBoost** (eXtreme Gradient Boosting) est un algorithme de machine learning développé par Tianqi Chen en 2014, basé sur le **gradient boosting** pour les arbres de décision. C'est l'un des algorithmes les plus performants et populaires en ML, particulièrement pour les données tabulaires.

### Pourquoi XGBoost est-il si populaire ?

- **Performances exceptionnelles** : Gagne régulièrement des compétitions Kaggle
- **Rapidité** : Optimisations algorithmiques avancées (parallélisation, cache-aware computing)
- **Régularisation** : Intégrée nativement pour éviter le surapprentissage
- **Flexibilité** : Supporte classification, régression, ranking
- **Gestion des données manquantes** : Nativement supportée
- **Interprétabilité** : Feature importance, SHAP values

### Positionnement par rapport aux autres algorithmes

| Algorithme | Avantages | Inconvénients |
|------------|-----------|---------------|
| **XGBoost** | Performance, régularisation, rapidité | Complexité hyperparamètres, sensible au tuning |
| **LightGBM** | Plus rapide, meilleur pour grands datasets | Parfois moins robuste sur petits datasets |
| **CatBoost** | Excellent pour variables catégorielles | Moins de contrôle fin |
| **Random Forest** | Simple, robuste, parallélisable | Moins performant, plus lourd en mémoire |

---

## 2. Concepts Préliminaires {#concepts-préliminaires}

### 2.1 Les Arbres de Décision (CART)

Un **arbre de décision** partitionne l'espace des features de manière récursive :

**Prédiction pour un arbre** :
$$f(x) = w_{q(x)}$$

où :
- $$q(x)$$ : fonction qui assigne $$x$$ à une feuille
- $$w_j$$ : score de la feuille $$j$$
- $$T$$ : nombre de feuilles

### 2.2 Le Boosting : Principe Général

Le boosting construit un modèle fort en combinant plusieurs modèles faibles **séquentiellement** :

$$\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)$$

où :
- $$K$$ : nombre d'arbres (itérations)
- $$f_k$$ : le $$k$$-ème arbre
- $$\hat{y}_i$$ : prédiction finale pour l'observation $$i$$

**Principe itératif** :
- **Itération 0** : Prédiction initiale (souvent la moyenne)
- **Itération 1** : Apprend sur les erreurs (résidus) de l'itération 0
- **Itération 2** : Apprend sur les erreurs de l'itération 1
- ...

**Exemple concret** : Prédire le salaire d'un employé
Salaire réel : 50k€
Itération 0 : Prédiction = 40k€ → Erreur = +10k€
Itération 1 : Apprend à corriger +10k€ → Prédiction = 40k + 8k = 48k€ → Erreur = +2k€
Itération 2 : Apprend à corriger +2k€ → Prédiction = 48k + 1.5k = 49.5k€ → Erreur = +0.5k€

### 2.3 Gradient Boosting vs XGBoost

**Gradient Boosting classique** :
- Optimise une fonction de perte $$L$$ en ajoutant des arbres qui suivent le **gradient négatif**
- À chaque itération : $$f_k(x) \approx -\nabla L$$

**XGBoost** ajoute :
- **Approximation du second ordre** (Newton-Raphson) : utilise le **gradient ET la hessienne**
- **Régularisation** de la complexité des arbres
- **Optimisations algorithmiques** massives

---

## 3. Fonctionnement Mathématique de XGBoost {#fonctionnement-mathématique}

### 3.1 Fonction Objective

XGBoost minimise une fonction objective composée de **deux termes** :

$$\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

**Décomposition** :

1. **Loss Function** $$\sum_{i=1}^{n} l(y_i, \hat{y}_i)$$ : Mesure l'erreur de prédiction
   - Régression : MSE = $$(y_i - \hat{y}_i)^2$$
   - Classification binaire : Log Loss = $$y_i \log(p_i) + (1-y_i) \log(1-p_i)$$

2. **Regularization Term** $$\sum_{k=1}^{K} \Omega(f_k)$$ : Pénalise la complexité
   $$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
   
   où :
   - $$T$$ : nombre de feuilles
   - $$w_j$$ : score de la feuille $$j$$
   - $$\gamma$$ : pénalité par feuille (contrôle $$T$$)
   - $$\lambda$$ : régularisation L2 sur les poids (contrôle $$w_j$$)

**Intuition** : On veut un modèle précis (minimiser $$l$$) mais pas trop complexe (minimiser $$\Omega$$).

### 3.2 Apprentissage Itératif (Additive Training)

À l'itération $$t$$, on a déjà $$t-1$$ arbres. On cherche le $$t$$-ème arbre $$f_t$$ :

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

La fonction objective devient :

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + \text{constante}$$

### 3.3 Approximation de Taylor au Second Ordre

**C'est ici que XGBoost innove !**

On approxime la loss avec un développement de Taylor au **second ordre** autour de $$\hat{y}_i^{(t-1)}$$ :

$$l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$$

où :
- $$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$$ : **gradient** (dérivée première)
- $$h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$ : **hessienne** (dérivée seconde)

**Pourquoi le second ordre ?**
- Le gradient donne la **direction** de descente
- La hessienne donne la **courbure** (vitesse de changement)
- Converge plus vite que le gradient seul (méthode de Newton vs descente de gradient)

**Exemple concret - MSE** :
- Loss : $$l(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$
- Gradient : $$g_i = \frac{\partial l}{\partial \hat{y}} = -(y_i - \hat{y}_i^{(t-1)}) = -\text{residual}_i$$
- Hessienne : $$h_i = \frac{\partial^2 l}{\partial \hat{y}^2} = 1$$

### 3.4 Fonction Objective Simplifiée

En retirant les constantes et en remplaçant par Taylor :

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

Pour un arbre avec $$T$$ feuilles, regroupons les observations par feuille :

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} [({\sum_{i \in I_j} g_i}) w_j + \frac{1}{2} ({\sum_{i \in I_j} h_i} + \lambda) w_j^2] + \gamma T$$

où $$I_j$$ = ensemble des observations dans la feuille $$j$$.

**Définissons** :
- $$G_j = \sum_{i \in I_j} g_i$$ : somme des gradients dans la feuille $$j$$
- $$H_j = \sum_{i \in I_j} h_i$$ : somme des hessiennes dans la feuille $$j$$

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} [G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2] + \gamma T$$

### 3.5 Poids Optimal des Feuilles

Pour minimiser $$\tilde{\mathcal{L}}^{(t)}$$ par rapport à $$w_j$$, on dérive et on annule :

$$\frac{\partial \tilde{\mathcal{L}}^{(t)}}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0$$

$$\boxed{w_j^* = -\frac{G_j}{H_j + \lambda}}$$H_j + \lambda}}$$

**Poids optimal de la feuille** $$j$$ :
- Proportionnel à la somme des gradients (erreurs)
- Inversement proportionnel à la somme des hessiennes (confiance)
- Régularisé par $$\lambda$$

### 3.6 Score de Qualité d'un Split

En remplaçant $$w_j^*$$ dans la fonction objective :

$$\tilde{\mathcal{L}}^{(t)} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

**Gain d'un split** : Amélioration quand on divise une feuille $$I$$ en $$I_L$$ (gauche) et $$I_R$$ (droite) :

$$\boxed{\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma}$$

**Interprétation** :
- Plus le gain est élevé, meilleur est le split
- $$\gamma$$ est soustrait : pénalise l'ajout d'une nouvelle feuille
- Si Gain < 0, on n'effectue pas le split (pruning)

**Exemple visuel** :



### 3.7 Algorithme de Construction de l'Arbre

**Split Finding Algorithm** :
Pour chaque arbre t:
1. Calculer g_i et h_i pour chaque observation
2. Initialiser un arbre avec une seule feuille (toutes les observations)

3. **Optimisations XGBoost** :
- **Approximate Split Finding** : Histogrammes de quantiles au lieu de tester toutes les valeurs
- **Weighted Quantile Sketch** : Prend en compte les poids (hessienne)
- **Sparsity-aware Split Finding** : Gestion des valeurs manquantes
- **Cache-aware Access** : Optimisation mémoire
- **Blocks for Out-of-core Computation** : Datasets trop grands pour la RAM

---

## 4. Hyperparamètres Fondamentaux {#hyperparamètres}

XGBoost a **beaucoup** d'hyperparamètres. Voici les plus importants classés par catégorie.

### 4.1 Paramètres des Arbres (Tree Booster)

#### **`max_depth`** (default: 6)
- **Description** : Profondeur maximale d'un arbre
- **Impact** :
  - ↑ Augmenter → modèle plus complexe, risque d'overfitting
  - ↓ Diminuer → modèle plus simple, risque d'underfitting
- **Recommandations** :
  - Petits datasets : 3-5
  - Moyens datasets : 6-8
  - Grands datasets : 8-12
- **Exemple pratique** : 
  - Dataset avec 10 features simples → `max_depth=3` suffit
  - Dataset avec 100 features et interactions complexes → `max_depth=8`

#### **`min_child_weight`** (default: 1)
- **Description** : Somme minimale des hessiennes ($$\sum h_i$$) requise dans un nœud enfant
- **Impact** : 
  - ↑ Augmenter → Plus conservateur, moins de splits, réduit overfitting
  - ↓ Diminuer → Plus de splits, modèle plus complexe
- **Intuition** : Si un nœud a peu d'observations ou observations avec petite hessienne (faible confiance), on ne split pas
- **Recommandations** :
  - Données déséquilibrées : augmenter (5-10)
  - Données équilibrées : 1-3
- **Formule** : Split autorisé si
  $$H_L \geq \text{min child weight}$$ ET $$H_R \geq \text{min child weight}$$

#### **`gamma`** (min_split_loss, default: 0)
- **Description** : Réduction minimale de loss requise pour effectuer un split (le $$\gamma$$ dans la formule du Gain)
- **Impact** :
  - ↑ Augmenter → Moins de splits, arbres plus petits, moins d'overfitting
  - 0 → Tous les splits avec Gain > 0 sont autorisés
- **Recommandations** :
  - Commencer à 0, augmenter progressivement si overfitting
  - Valeurs typiques : 0-5 pour données normalisées
- **Exemple** :
  - Gain calculé = 0.8, gamma = 0 → Split effectué
  - Gain calculé = 0.8, gamma = 1.0 → Split rejeté

#### **`subsample`** (default: 1)
- **Description** : Fraction des observations à échantillonner pour chaque arbre
- **Impact** :
  - < 1 → Stochastic Gradient Boosting, réduit overfitting, accélère training
  - = 1 → Utilise toutes les observations
- **Recommandations** :
  - Valeurs typiques : 0.5-0.9
  - Grand dataset → 0.5-0.7
  - Petit dataset → 0.8-1.0
- **Exemple** : 
  - Dataset de 10,000 lignes, subsample=0.7 → Chaque arbre apprend sur 7,000 lignes aléatoires

#### **`colsample_bytree`** (default: 1)
- **Description** : Fraction des features à échantillonner pour chaque arbre
- **Impact** : Similaire à Random Forest, réduit overfitting et corrélation entre arbres
- **Recommandations** : 0.5-1.0
- **Variantes** :
  - `colsample_bylevel` : Échantillonnage par niveau de l'arbre
  - `colsample_bynode` : Échantillonnage par nœud

#### **`max_leaves`** (default: 0 = unlimited)
- **Description** : Nombre maximum de feuilles (alternative à `max_depth`)
- **Impact** : Contrôle direct de la complexité
- **Recommandations** : 
  - Utilisé par LightGBM par défaut
  - XGBoost : préférer `max_depth` sauf cas spécifiques

### 4.2 Paramètres de Régularisation

#### **`lambda`** (reg_lambda, default: 1)
- **Description** : Régularisation L2 sur les poids des feuilles (le $$\lambda$$ dans $$w_j^* = -\frac{G_j}{H_j + \lambda}$$)
- **Impact** :
  - ↑ Augmenter → Poids plus petits, modèle plus lisse, moins d'overfitting
  - = 0 → Pas de régularisation L2
- **Recommandations** : 
  - Valeur par défaut (1) fonctionne souvent bien
  - Augmenter si overfitting : 10-100
- **Formule** : $$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

#### **`alpha`** (reg_alpha, default: 0)
- **Description** : Régularisation L1 sur les poids des feuilles
- **Impact** :
  - Favorise la **sparsité** (certains poids = 0)
  - Utile pour feature selection
- **Recommandations** :
  - Dataset avec beaucoup de features inutiles : 0.1-1
  - Sinon : garder à 0
- **Comparaison L1 vs L2** :
  - L1 : Pénalité proportionnelle à $$|w|$$ → Met certains poids à 0
  - L2 : Pénalité proportionnelle à $$w^2$$ → Réduit tous les poids

### 4.3 Paramètres d'Apprentissage

#### **`learning_rate`** (eta, default: 0.3)
- **Description** : Coefficient qui réduit le poids de chaque arbre
- **Formule** : $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$
- **Impact** :
  - ↓ Diminuer → Convergence plus lente mais modèle plus robuste
  - ↑ Augmenter → Convergence plus rapide mais risque d'overfitting
- **Recommandations** :
  - **Règle d'or** : Petit learning_rate + grand n_estimators = meilleure performance
  - Valeurs typiques : 0.01-0.3
  - Début développement : 0.1
  - Production : 0.01-0.05 avec early stopping
- **Exemple** :
  - lr=0.3, 100 arbres → Convergence rapide, risque d'overfitting
  - lr=0.01, 1000 arbres → Convergence lente, meilleure généralisation

#### **`n_estimators`** (num_boost_round, default: 100)
- **Description** : Nombre d'arbres (itérations de boosting)
- **Impact** :
  - ↑ Augmenter → Modèle plus complexe, risque d'overfitting sans early stopping
  - Trop peu → Underfitting
- **Recommandations** :
  - Utiliser avec **early stopping** : mettre une grande valeur (500-5000)
  - Sans early stopping : tuner soigneusement (50-300)
- **Relation avec learning_rate** :
  - lr élevé → moins d'arbres nécessaires
  - lr faible → plus d'arbres nécessaires

### 4.4 Paramètres Spécifiques à la Tâche

#### **`objective`**
- **Classification binaire** : 
  - `binary:logistic` : Retourne probabilités [0,1]
  - `binary:hinge` : Retourne classes binaires (SVM-like)
- **Classification multiclasse** :
  - `multi:softmax` : Retourne classes
  - `multi:softprob` : Retourne probabilités pour chaque classe
- **Régression** :
  - `reg:squarederror` : MSE (défaut)
  - `reg:squaredlogerror` : MSLE (quand cibles ont distribution log)
  - `reg:logistic` : Régression logistique
  - `reg:pseudohubererror` : Robuste aux outliers
- **Ranking** :
  - `rank:pairwise`, `rank:ndcg`

#### **`eval_metric`**
- **Classification** : `logloss`, `error`, `auc`, `aucpr`
- **Régression** : `rmse`, `mae`, `mape`, `mphe`
- **Multiclass** : `mlogloss`, `merror`
- **Ranking** : `ndcg`, `map`

#### **`scale_pos_weight`** (default: 1)
- **Description** : Poids pour la classe positive (classification déséquilibrée)
- **Formule** : `scale_pos_weight = sum(negative instances) / sum(positive instances)`
- **Exemple** :
  - 900 négatifs, 100 positifs → `scale_pos_weight=9`

### 4.5 Gestion des Données Manquantes

XGBoost gère nativement les valeurs manquantes :

#### **`missing`** (default: np.nan)
- Valeur considérée comme manquante

#### Mécanisme interne :
- Pour chaque split, teste l'assignation des valeurs manquantes à gauche ET à droite
- Choisit la direction qui maximise le gain
- Entraîne naturellement une "direction par défaut" pour les valeurs manquantes

### 4.6 Paramètres de Performance

#### **`n_jobs`** (nthread, default: -1)
- Nombre de threads CPU
- -1 = utilise tous les cœurs disponibles

#### **`tree_method`**
- `auto` : Choix automatique
- `exact` : Enumération exhaustive des splits (précis mais lent)
- `approx` : Approximate split finding (histogrammes)
- `hist` : Histogram-based algorithm (le plus rapide)
- `gpu_hist` : Version GPU (CUDA requis)

#### **`predictor`**
- `auto`, `cpu_predictor`, `gpu_predictor`

### 4.7 Early Stopping

Arrête le training quand la métrique ne s'améliore plus :

```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    early_stopping_rounds=50  # Arrête si pas d'amélioration après 50 rounds
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Meilleur nombre d'arbres: {model.best_iteration}")
```
