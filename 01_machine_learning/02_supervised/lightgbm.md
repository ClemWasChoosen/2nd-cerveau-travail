# **LightGBM : Guide Complet**

## **1. Introduction et Contexte**

### **1.1 Qu'est-ce que LightGBM ?**

**LightGBM** (Light Gradient Boosting Machine) est un framework de gradient boosting développé par Microsoft Research en 2017. Il fait partie de la famille des algorithmes de boosting basés sur les arbres de décision, au même titre que XGBoost et CatBoost.

**Paper fondateur** : *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (Ke et al., 2017) - NIPS 2017
🔗 https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

### **1.2 Pourquoi LightGBM ?**

Les implémentations classiques de gradient boosting (comme XGBoost) rencontrent des **limitations** sur des datasets volumineux :

1. **Complexité temporelle** : L'énumération de tous les splits possibles est coûteuse
2. **Complexité spatiale** : Stocker les données en mémoire devient problématique
3. **Scalabilité** : Difficile de paralléliser efficacement avec des millions d'instances

**LightGBM répond à ces problèmes** par 3 innovations majeures :
- **Histogram-based learning** : Discrétisation des features continues
- **GOSS** (Gradient-based One-Side Sampling) : Échantillonnage intelligent des instances
- **EFB** (Exclusive Feature Bundling) : Réduction de la dimensionnalité

---

## **2. Rappels : Gradient Boosting**

### **2.1 Principe général**

Le gradient boosting construit un modèle additif de manière séquentielle :

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

Où :
- $$F_m(x)$$ : modèle après $$m$$ itérations
- $$h_m(x)$$ : nouvel arbre (weak learner)
- $$\nu$$ : learning rate (shrinkage)

### **2.2 Fonction objectif**

À chaque itération, on minimise :

$$\mathcal{L}(F_m) = \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + h_m(x_i)) + \Omega(h_m)$$

Où :
- $$L$$ : fonction de perte (loss function)
- $$\Omega(h_m)$$ : terme de régularisation de l'arbre

### **2.3 Approximation de Taylor (2nd ordre)**

On approxime la perte par développement de Taylor :

$$\mathcal{L}(F_m) \approx \sum_{i=1}^{n} \left[ L(y_i, F_{m-1}(x_i)) + g_i h_m(x_i) + \frac{1}{2} h_i h_m^2(x_i) \right] + \Omega(h_m)$$

Où :
- $$g_i = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$ : **gradient de 1er ordre**
- $$h_i = \frac{\partial^2 L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2}$$ : **hessien (gradient de 2nd ordre)**

**Pourquoi le 2nd ordre ?** Il permet une convergence plus rapide et des approximations plus précises de la fonction objectif.

---

## **3. Les Innovations Clés de LightGBM**

### **3.1 Histogram-based Learning**

#### **Le problème**

Les algorithmes classiques (pré-sorted algorithm) :
1. Trient toutes les valeurs des features
2. Énumèrent tous les splits possibles : complexité $$O(n \cdot d)$$ où $$n$$ = nombre d'instances, $$d$$ = nombre de features

#### **La solution : Histogrammes**

LightGBM **discrétise** les features continues en $$k$$ bins (par défaut $$k=255$$).

**Algorithme** :
1. Construire des histogrammes de gradients pour chaque feature
2. Chercher le meilleur split sur ces bins discrets

**Complexité** : $$O(k \cdot d)$$ au lieu de $$O(n \cdot d)$$

**Gain pour le split** :

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Où :
- $$G_L = \sum_{i \in \text{left}} g_i$$ : somme des gradients à gauche
- $$H_L = \sum_{i \in \text{left}} h_i$$ : somme des hessiens à gauche
- $$\lambda$$ : régularisation L2
- $$\gamma$$ : pénalité de complexité pour ajouter une feuille

**Pourquoi ça marche ?** 
- La perte d'information due à la discrétisation est minime
- Le gain en vitesse est énorme (réduction de plusieurs ordres de grandeur)
- Consommation mémoire réduite (pas besoin de stocker toutes les valeurs triées)

#### **Optimisation supplémentaire : Histogram Subtraction**

Pour un nœud parent, si on a construit l'histogramme d'un enfant (gauche), on obtient l'autre (droit) par soustraction :

$$\text{Hist}_{\text{right}} = \text{Hist}_{\text{parent}} - \text{Hist}_{\text{left}}$$

**Gain** : Divise par 2 le coût de construction des histogrammes.

---

### **3.2 GOSS (Gradient-based One-Side Sampling)**

#### **Le problème**

Avec des millions d'instances, calculer les gradients et trouver les splits reste coûteux.

#### **L'intuition**

Toutes les instances ne contribuent pas également à l'apprentissage :
- Les instances avec **grands gradients** sont mal prédites → importantes
- Les instances avec **petits gradients** sont bien prédites → moins importantes

**GOSS propose** : Garder toutes les instances à grands gradients, mais échantillonner aléatoirement celles à petits gradients.

#### **L'algorithme GOSS**

**Paramètres** :
- $$a$$ : proportion d'instances à grand gradient à garder
- $$b$$ : proportion d'instances à petit gradient à échantillonner

**Étapes** :
1. Trier les instances par $$|g_i|$$ (valeur absolue du gradient)
2. Garder les top $$a \times 100\%$$ instances (set $$A$$)
3. Échantillonner aléatoirement $$b \times 100\%$$ des instances restantes (set $$B$$)
4. Amplifier la contribution de $$B$$ par un facteur $$\frac{1-a}{b}$$

**Calcul du gain avec GOSS** :

$$\tilde{G}_L = \sum_{i \in A_L} g_i + \frac{1-a}{b} \sum_{i \in B_L} g_i$$

$$\tilde{H}_L = \sum_{i \in A_L} h_i + \frac{1-a}{b} \sum_{i \in B_L} h_i$$

**Pourquoi l'amplification $$\frac{1-a}{b}$$ ?**
Pour compenser le sous-échantillonnage et maintenir la distribution originale des gradients.

**Complexité** : Réduite de $$O(n)$$ à $$O(a \cdot n + b \cdot n)$$ où typiquement $$a + b \ll 1$$

**Valeurs typiques** : $$a = 0.1$$, $$b = 0.1$$ → on utilise seulement 20% des données !

**Pourquoi ça marche ?**
- Préserve la distribution des gradients
- Se concentre sur les instances difficiles
- Gain de vitesse significatif avec perte de précision minime

---

### **3.3 EFB (Exclusive Feature Bundling)**

#### **Le problème**

En haute dimension (beaucoup de features), la complexité augmente linéairement avec $$d$$.

#### **L'intuition**

Dans les données réelles (surtout les données sparses comme one-hot encoding), de nombreuses features sont **mutuellement exclusives** : elles ne prennent jamais de valeurs non-nulles simultanément.

**Exemple** : 
- Feature A : [0, 1, 0, 0]
- Feature B : [1, 0, 0, 0]
- Feature C : [0, 0, 1, 1]

A et B sont mutuellement exclusives, on peut les "bundler" en une seule feature.

#### **L'algorithme EFB**

**Objectif** : Regrouper les features mutuellement exclusives en "bundles".

**Étape 1 : Construction du graphe**
- Créer un graphe où chaque feature est un nœud
- Ajouter une arête entre deux features si elles **ne sont pas** mutuellement exclusives
- Poids de l'arête = degré de conflit

**Étape 2 : Problème de coloration de graphe**
C'est un problème NP-hard, donc on utilise une heuristique gloutonne :

1. Trier les features par degré décroissant (nombre de conflits)
2. Pour chaque feature, l'assigner au bundle avec le plus petit conflit

**Étape 3 : Merge des features**
Pour merger des features dans un bundle :
- Ajouter des offsets pour séparer les bins de différentes features

**Exemple** :
- Feature A : bins [0, 1, 2] → reste [0, 1, 2]
- Feature B : bins [0, 1, 2] → devient [3, 4, 5]
- Bundle AB : [0, 1, 2, 3, 4, 5]

**Complexité** : Réduction de $$O(d)$$ à $$O(d')$$ où $$d' \ll d$$ (nombre de bundles)

**Pourquoi ça marche ?**
- Les datasets réels ont souvent beaucoup de sparsité
- One-hot encoding crée naturellement des features exclusives
- Réduction dimensionnelle sans perte d'information

---

### **3.4 Leaf-wise (Best-first) Tree Growth**

#### **Level-wise vs Leaf-wise**

**XGBoost (level-wise)** :
- Développe tous les nœuds d'un même niveau
- Plus équilibré, moins d'overfitting
- Mais peut créer des splits inutiles

**LightGBM (leaf-wise)** :
- Développe la feuille avec le **maximum gain**
- Arbres plus profonds et asymétriques
- Converge plus rapidement

**Illustration** :

**Risque** : Overfitting plus facile avec leaf-wise

**Solution** : Paramètre `max_depth` pour limiter la profondeur

**Pourquoi leaf-wise dans LightGBM ?**
- Convergence plus rapide vers de bonnes solutions
- Couplé avec GOSS et EFB, permet d'économiser du temps sans sacrifier la précision
- Mieux adapté aux datasets avec des relations complexes

---

## **4. Aspects Mathématiques Détaillés**

### **4.1 Fonction Objectif Complète**

Pour un arbre avec $$T$$ feuilles, la fonction objectif est :

$$\mathcal{L} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{j=1}^{T} \left[ \frac{1}{2} \lambda w_j^2 + \gamma \right]$$

Où :
- $$w_j$$ : poids (prédiction) de la feuille $$j$$
- $$\lambda$$ : régularisation L2 sur les poids
- $$\gamma$$ : pénalité pour le nombre de feuilles (pruning)

### **4.2 Calcul du Poids Optimal d'une Feuille**

Pour une feuille $$j$$ contenant l'ensemble d'instances $$I_j$$, le poids optimal est :

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Dérivation** : En annulant la dérivée de $$\mathcal{L}$$ par rapport à $$w_j$$.

### **4.3 Exemples de Loss Functions**

#### **Régression (MSE)** :

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

Gradients :
- $$g_i = \hat{y}_i - y_i$$
- $$h_i = 1$$

#### **Classification binaire (log-loss)** :

$$L(y, \hat{y}) = -[y \log(p) + (1-y) \log(1-p)]$$

Où $$p = \frac{1}{1 + e^{-\hat{y}}}$$ (sigmoid)

Gradients :
- $$g_i = p_i - y_i$$
- $$h_i = p_i(1 - p_i)$$

**Pourquoi le hessien $$h_i = p_i(1-p_i)$$ ?**
C'est la dérivée seconde de la log-loss. Il représente la "confiance" : 
- $$h_i$$ est maximal quand $$p_i = 0.5$$ (incertitude maximale)
- $$h_i$$ est minimal quand $$p_i \to 0$$ ou $$p_i \to 1$$ (prédictions confiantes)

---

## **5. Comparaison : XGBoost vs LightGBM vs CatBoost**

| **Aspect** | **XGBoost** | **LightGBM** | **CatBoost** |
|-----------|------------|-------------|-------------|
| **Année** | 2014 | 2017 | 2017 |
| **Split finding** | Pre-sorted / Histogram | Histogram | Oblivious trees |
| **Tree growth** | Level-wise | Leaf-wise | Symmetric (level-wise) |
| **Sampling** | Random | GOSS | MVS (Minimal Variance Sampling) |
| **Feature bundling** | ❌ | EFB | ❌ |
| **Catégorielles** | Encodage manuel | Support natif | **Excellent support natif** |
| **Vitesse** | Rapide | **Très rapide** | Moyen |
| **Mémoire** | Moyenne | **Faible** | Moyenne |
| **Overfitting** | Moyen | **Risque élevé** | **Résistant** |
| **GPU** | ✅ | ✅ | ✅ |

### **5.1 Quand utiliser LightGBM ?**

✅ **Avantages** :
- **Très gros datasets** (millions d'instances)
- Besoins de **rapidité** d'entraînement
- Données **sparses** (beaucoup de zéros)
- **Mémoire limitée**

⚠️ **Attention** :
- **Petits datasets** (< 10k instances) : risque d'overfitting
- Nécessite un tuning attentif des hyperparamètres

### **5.2 Quand utiliser XGBoost ?**

✅ **Avantages** :
- Datasets de taille **moyenne**
- Besoin de **stabilité** et robustesse
- Meilleure **documentation** et communauté
- Plus de contrôle sur la régularisation

### **5.3 Quand utiliser CatBoost ?**

✅ **Avantages** :
- Beaucoup de **features catégorielles**
- Besoin de **résistance à l'overfitting**
- Moins de tuning nécessaire (meilleurs hyperparamètres par défaut)
- Excellente gestion du **target leakage**

---

## **6. Gestion des Variables Catégorielles**

### **6.1 Le Problème du One-Hot Encoding**

One-hot encoding crée :
- **Explosion dimensionnelle** : $$k$$ catégories → $$k$$ features
- **Sparsité** : Beaucoup de zéros
- **Perte d'information** : Relations ordinales perdues

### **6.2 Solution de LightGBM**

LightGBM utilise une **approche optimale de split** pour les catégorielles.

**Algorithme** :
1. Trier les catégories par gradient moyen : $$\frac{\sum_{i} g_i}{\sum_{i} h_i}$$
2. Trouver le split optimal le long de cet ordre
3. Complexité : $$O(k \log k)$$ au lieu de $$O(2^k)$$

**Exemple** :
Catégorie = Couleur : {Rouge, Bleu, Vert, Jaune}

1. Calculer le gradient moyen par catégorie :
   - Rouge : -0.5
   - Bleu : 0.2
   - Vert : 0.8
   - Jaune : -0.1

2. Trier : Rouge (-0.5) < Jaune (-0.1) < Bleu (0.2) < Vert (0.8)

3. Tester les splits :
   - {Rouge} vs {Jaune, Bleu, Vert}
   - {Rouge, Jaune} vs {Bleu, Vert}
   - {Rouge, Jaune, Bleu} vs {Vert}

**Pourquoi trier par gradient moyen ?**
Fisher (1958) a montré que c'est l'ordre optimal pour minimiser la variance dans le cas de régression. LightGBM généralise ce résultat.

**Utilisation pratique** :
```python
# Spécifier les features catégorielles
categorical_features = ['color', 'brand', 'category']

# LightGBM les gère automatiquement
lgb.train(params, train_data, categorical_feature=categorical_features)
```
# 7. Hyperparamètres et Tuning
## 7.1 Hyperparamètres Essentiels
Contrôle de la complexité du modèle

|     Paramètre     |        Description        |   Valeurs typiques   |           Impact          |
|:-----------------:|:-------------------------:|:--------------------:|:-------------------------:|
| num_leaves        | Nombre max de feuilles    | 31 (défaut), 15-255  | ↑ capacité, ↑ overfitting |
| max_depth         | Profondeur max            | -1 (défaut), 5-15    | Limite l'overfitting      |
| min_data_in_leaf  | Instances min par feuille | 20 (défaut), 50-1000 | ↑ régularisation          |
| min_gain_to_split | Gain min pour split       | 0 (défaut), 0.01-1   | Pruning agressif          |

Règle empirique : `num_leaves` ≤2 max_depth
Pourquoi ? Si `num_leaves` est trop grand pour la profondeur, l'arbre ne pourra jamais atteindre toutes les feuilles → gaspillage.

Contrôle du learning
|       Paramètre       |   Description   | Valeurs typiques |
|:---------------------:|:---------------:|:----------------:|
| learning_rate         | Shrinkage       | 0.01-0.1         |
| num_iterations        | Nombre d'arbres | 100-10000        |
| early_stopping_rounds | Arrêt anticipé  | 50-200           |

Compromis :

- `learning_rate` ↓ et `num_iterations` ↑ → meilleure généralisation mais plus lent
- `learning_rate` ↑ → plus rapide mais risque d'overfitting

Régularisation
|     Paramètre    |      Description      | Valeurs typiques |
|:----------------:|:---------------------:|:----------------:|
| lambda_l1        | Régularisation L1     | 0-10             |
| lambda_l2        | Régularisation L2     | 0-10             |
| feature_fraction | % features par arbre  | 0.8-1.0          |
| bagging_fraction | % instances par arbre | 0.8-1.0          |
| bagging_freq     | Fréquence du bagging  | 1-10             |

Pourquoi `feature_fraction` < 1 ?

- Introduit de la randomisation → réduit l'overfitting
- Rend les arbres plus diversifiés → meilleur ensemble
- Inspiré de Random Forest

Paramètres spécifiques GOSS

|   Paramètre   |      Description     |    Valeurs typiques    |
|:-------------:|:--------------------:|:----------------------:|
| boosting_type | Type de boosting     | 'gbdt', 'goss', 'dart' |
| top_rate      | Paramètre aa de GOSS | 0.1-0.5                |
| other_rate    | Paramètre bb de GOSS | 0.1-0.5                |

7.2 Stratégie de Tuning
Phase 1 : Tuning grossier

- Fixer : learning_rate = 0.1, num_iterations = 100
- Tuner : num_leaves, max_depth
  - Commencer par num_leaves = 31, max_depth = -1
  - Augmenter progressivement en surveillant la validation

Phase 2 : Régularisation

- Tuner : min_data_in_leaf, min_gain_to_split
    - Objectif : réduire l'overfitting observé en Phase 1
- Tuner : lambda_l1, lambda_l2
    - Essayer différentes combinaisons

Phase 3 : Sampling

- Tuner : feature_fraction, bagging_fraction, bagging_freq
  - Introduire de la randomisation

Phase 4 : Learning rate

- Réduire : learning_rate (0.05, 0.01, 0.005)
- Augmenter : num_iterations en conséquence
- Utiliser : early_stopping_rounds pour éviter l'overfitting

Techniques d'optimisation
Optuna (recommandé) :

```python
import optuna
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
    }
    
    pruning_callback = LightGBMPruningCallback(trial, 'auc')
    
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[pruning_callback],
        nfold=5,
        stratified=True,
        early_stopping_rounds=50
    )
    
    return cv_results['auc-mean'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```
# 8. Pièges et Bonnes Pratiques
## 8.1 Pièges Courants
❌ Piège 1 : num_leaves trop grand pour la taille du dataset

    Symptôme : Overfitting sévère
    Solution : Règle : num_leaves < n10 où n = taille train set

❌ Piège 2 : Ne pas utiliser early_stopping

    Symptôme : Overfitting, temps perdu
    Solution : Toujours activer avec early_stopping_rounds

❌ Piège 3 : Ignorer les features catégorielles

    Symptôme : Performance sous-optimale
    Solution : Spécifier categorical_feature explicitement

❌ Piège 4 : learning_rate trop élevé

    Symptôme : Instabilité, sous-fitting
    Solution : Commencer par 0.05-0.1, puis réduire

❌ Piège 5 : Ne pas normaliser les features pour LightGBM ?

    Bonne nouvelle : Pas nécessaire ! Les arbres sont invariants aux transformations monotones

## 8.2 Bonnes Pratiques
- ✅ Toujours faire une validation croisée pour estimer la vraie performance
- ✅ Monitorer train vs valid metrics à chaque itération
- ✅ Sauvegarder les modèles :

```python
model.save_model('model.txt')
model = lgb.Booster(model_file='model.txt')
```

- ✅ Utiliser un seed fixe pour la reproductibilité :

```python
params = {
    'seed': 42,
    'feature_fraction_seed': 42,
    'bagging_seed': 42,
}
```
✅ Logger les hyperparamètres et résultats (MLflow, Weights&Biases)

# 9. Sources et Références
## 9.1 Papers Fondamentaux

1. LightGBM original paper :\
    Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017.\
    🔗 https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
2. XGBoost (pour comparaison) :\
    Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016.\
    🔗 https://arxiv.org/abs/1603.02754
3. Gradient Boosting original :\
    Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics.\
    🔗 https://www.jstor.org/stable/2699986

## 9.2 Ressources Techniques

4. Documentation officielle LightGBM :\
    🔗 https://lightgbm.readthedocs.io/
5. Comparaison détaillée des GBDT :\
    Bentéjac, C., et al. (2021). A comparative analysis of gradient boosting algorithms. Artificial Intelligence Review.
    🔗 https://link.springer.com/article/10.1007/s10462-020-09896-5
6. Tuning LightGBM :\
    🔗 https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

## 9.3 Blogs et Tutoriels

    Neptune.ai - LightGBM parameters guide :
        🔗 https://neptune.ai/blog/lightgbm-parameters-guide
    Kaggle - LightGBM tricks :
        Discussions et notebooks des compétitions
# 10. Checklist pour Démarrer un Projet avec LightGBM
Préparation des données

- [ ] Identifier les features catégorielles
- [ ] Gérer les valeurs manquantes (LightGBM les gère nativement)
- [ ] Split train/validation/test stratifié

Configuration initiale

- [ ] Définir l'objectif (objective) et la métrique (metric)
- [ ] Spécifier categorical_feature si applicable
- [ ] Fixer les seeds pour la reproductibilité

Entraînement de base

- [ ] Commencer avec paramètres par défaut
- [ ] Activer early_stopping_rounds
- [ ] Logger les métriques train et validation

Tuning

- [ ] Phase 1 : Tuner num_leaves et max_depth
- [ ] Phase 2 : Ajouter régularisation (lambda_l1, lambda_l2)
- [ ] Phase 3 : Optimiser sampling (feature_fraction, bagging_fraction)
- [ ] Phase 4 : Réduire learning_rate et augmenter num_iterations

Validation

- [ ] Cross-validation (5-fold minimum)
- [ ] Vérifier l'absence d'overfitting
- [ ] Analyser feature importance

Production

- [ ] Sauvegarder le modèle final
- [ ] Documenter les hyperparamètres optimaux
- [ ] Tester sur le test set (une seule fois !)

# Conclusion
LightGBM est un outil extrêmement puissant pour les problèmes de ML sur données tabulaires, particulièrement quand :

- La vitesse est cruciale
- Les datasets sont volumineux
- La mémoire est limitée

Ses innovations (Histogram, GOSS, EFB, Leaf-wise) en font souvent le choix par défaut pour les compétitions Kaggle et les applications industrielles.
Cependant, il nécessite :

- Un tuning attentif des hyperparamètres
- Une vigilance sur l'overfitting (surtout avec petits datasets)
- Une bonne compréhension des compromis vitesse/précision

Prochaines étapes :

1. Expérimenter sur tes données
2. Comparer avec XGBoost/CatBoost sur ton cas d'usage
3. Approfondir le tuning avec Optuna
4. Explorer les fonctionnalités avancées (custom objectives, callbacks, etc.)
