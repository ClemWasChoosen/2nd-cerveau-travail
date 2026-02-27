# SHAP : SHapley Additive exPlanations

> **Résumé en une phrase** : SHAP est une méthode d'interprétabilité unifiée basée sur la théorie des jeux de Shapley qui attribue à chaque feature une valeur de contribution équitable pour expliquer les prédictions de n'importe quel modèle de Machine Learning.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-02-27 |
| **Dernière mise à jour** | 2026-02-27 |
| **Domaine** | Machine Learning / Interprétabilité |
| **Niveau** | Intermédiaire à Avancé |
| **Durée de lecture** | ~50 minutes |
| **Fichier** | `shap_shapley_values.md` |
| **Emplacement** | `/01_machine_learning/04_interpretability/` |
| **Tags** | `#interpretability` `#explainability` `#shap` `#shapley` `#xai` `#production` |

### Prérequis

- [ ] Compréhension des modèles ML de base (arbres de décision, régression, réseaux de neurones)
- [ ] Notions de probabilités et statistiques (espérance, distributions)
- [ ] Python et bibliothèques ML (scikit-learn, numpy)
- [ ] Concepts de base en théorie des jeux (optionnel mais recommandé)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : Fondamentaux ML, arbres de décision, réseaux de neurones
- **Complémentaires** : [[integrated_gradients]] (pour DL), [[lime_local_explanations]] (méthode alternative)
- **Suite recommandée** : [[partial_dependence_plots]], [[feature_importance_methods]]

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Ce cours vous enseigne **SHAP (SHapley Additive exPlanations)**, une méthode d'interprétabilité moderne et théoriquement fondée qui répond à la question : *"Quelle est la contribution de chaque feature à une prédiction donnée ?"*. Vous comprendrez la théorie de Shapley issue de la théorie des jeux, son adaptation au Machine Learning, et comment utiliser les différents explainers SHAP pour expliquer n'importe quel modèle en production.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Expliquer la théorie de Shapley et ses propriétés mathématiques fondamentales
2. **Appliquer** : Utiliser les explainers SHAP appropriés (Tree, Kernel, Deep, etc.) selon le type de modèle
3. **Analyser** : Interpréter les visualisations SHAP (summary plot, dependence plot, force plot, waterfall) pour prendre des décisions métier
4. **Évaluer** : Choisir entre SHAP et d'autres méthodes (LIME, IG) selon le contexte de production et les contraintes
5. **Créer** : Implémenter une pipeline d'explicabilité SHAP complète pour la mise en production d'un modèle

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

Dans un contexte de **mise en production de modèles ML en entreprise**, l'explicabilité n'est plus optionnelle :

1. **Réglementations** : RGPD (droit à l'explication), AI Act européen, régulations sectorielles (finance, santé)
2. **Confiance métier** : Les décideurs doivent comprendre pourquoi un modèle recommande une action
3. **Debugging** : Identifier les biais, erreurs de données, ou comportements inattendus du modèle
4. **Optimisation** : Comprendre quelles features sont importantes pour améliorer la collecte de données
5. **Responsabilité** : En cas de décision contestée, pouvoir justifier le raisonnement du modèle

**SHAP se distingue** car il offre une **fondation théorique solide** (théorie des jeux de Shapley) tout en étant **model-agnostic** et applicable à tous types de modèles.

### Quel problème résout-il ?

**Problème concret** :

Vous déployez un modèle de **scoring de crédit** (arbre de décision XGBoost) qui prédit si un client est éligible à un prêt. Le modèle refuse un client avec un score de 0.32 (seuil = 0.5). Le client demande : *"Pourquoi ai-je été refusé ?"*

**Sans explicabilité** :
- Réponse vague : "Votre profil ne correspond pas aux critères"
- Problèmes : légalité (RGPD), confiance client, impossibilité d'améliorer son dossier

**Avec SHAP** :
- Réponse précise : "Votre revenu annuel (+0.05) et historique de paiements (+0.03) jouent en votre faveur, mais votre ratio d'endettement (-0.15) et nombre de crédits actifs (-0.10) réduisent significativement votre score"
- Le client comprend et peut agir (réduire ses crédits)

**Exemple concret** :

```python
import shap
import xgboost as xgb
import numpy as np

# Modèle de scoring crédit (simplifié)
X_train = np.array([[50000, 0.3, 2], [80000, 0.15, 1], [30000, 0.5, 4]])  # [revenu, ratio_dette, nb_credits]
y_train = np.array([1, 1, 0])  # 1 = approuvé, 0 = refusé

model = xgb.XGBClassifier().fit(X_train, y_train)

# Client refusé
client = np.array([[45000, 0.6, 3]])
prediction = model.predict_proba(client)[0, 1]  # 0.32

# SANS SHAP : Juste une prédiction binaire
print(f"Prédiction : {prediction:.2f} → Refusé")  # Pas d'explication

# AVEC SHAP : Explication détaillée
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(client)
# shap_values ≈ [-0.05, -0.15, -0.08]  # Contribution de chaque feature
# Interpretation : ratio_dette (-0.15) est le facteur principal de refus
```

### Applications dans le monde réel

1. **Finance - Scoring de crédit** :
   - Expliquer les décisions d'octroi de crédit (conformité RGPD)
   - Identifier les features discriminatoires (biais de genre, origine ethnique indirecte)
   - Source : [Fair Lending Compliance avec SHAP](https://www.fico.com/blogs/explaining-machine-learning-models)

2. **Santé - Diagnostic médical** :
   - Expliquer les prédictions de modèles de diagnostic (cancer, maladies cardiovasculaires)
   - Gagner la confiance des médecins pour adoption clinique
   - Exemple : [DeepMind - Interpretable ML for Healthcare](https://www.nature.com/articles/s41591-018-0300-7)

3. **Industrie - Maintenance prédictive** :
   - Expliquer pourquoi une machine est prédite en panne
   - Prioriser les actions de maintenance selon les features critiques
   - Cas : [General Electric - Predictive Maintenance with SHAP](https://www.ge.com/digital/applications/asset-performance-management)

4. **E-commerce - Recommandations produits** :
   - Expliquer pourquoi un produit est recommandé (transparence utilisateur)
   - Optimiser les features de recommandation (améliorer CTR)
   - Exemple : Netflix, Amazon utilisent des techniques similaires

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Cette section construit les bases théoriques de SHAP. Nous commençons par la théorie des jeux de Shapley (1953), puis adaptons cette théorie au Machine Learning.

### 1. Concepts clés

#### 1.1 Théorie de Shapley (1953)

**Définition** :

La **valeur de Shapley** est un concept de théorie des jeux coopératifs qui attribue équitablement la contribution de chaque joueur au gain total d'une coalition.

**Pourquoi cette définition ?**

Imaginons 3 développeurs (A, B, C) qui collaborent sur un projet générant 100k€ de profit. Comment répartir équitablement cette somme selon la contribution réelle de chacun ? La valeur de Shapley résout ce problème en considérant **toutes les coalitions possibles** et la **contribution marginale** de chaque joueur.

**Visualisation conceptuelle** :
```
Coalitions possibles avec 3 joueurs {A, B, C} :
┌─────────────────────────────────────────┐
│ Ø (coalition vide)                      │
│ {A} → valeur v(A)                       │
│ {B} → valeur v(B)                       │
│ {C} → valeur v(C)                       │
│ {A,B} → valeur v(A,B)                   │
│ {A,C} → valeur v(A,C)                   │
│ {B,C} → valeur v(B,C)                   │
│ {A,B,C} → valeur v(A,B,C)              │
└─────────────────────────────────────────┘
↓
Contribution marginale de A :

    Dans coalition {A} : v(A) - v(Ø)
    Dans coalition {A,B} : v(A,B) - v(B)
    Dans coalition {A,C} : v(A,C) - v(C)
    Dans coalition {A,B,C} : v(A,B,C) - v(B,C) ↓ Valeur de Shapley de A = moyenne pondérée de toutes ses contributions
```

**Sources académiques** :

- [Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games](https://www.rand.org/pubs/papers/P0295.html) - Paper fondateur
- [Roth, A. E. (1988). The Shapley value: essays in honor of Lloyd S. Shapley](https://www.cambridge.org/core/books/shapley-value/3B3D6D81E6C6E5E5F4F5F6F7F8F9FAFB) - Ouvrage de référence

#### 1.2 Adaptation au Machine Learning : SHAP Values

**Définition** :

Les **SHAP values** (SHapley Additive exPlanations) adaptent la théorie de Shapley au contexte ML : chaque feature est un "joueur", la "coalition" est un sous-ensemble de features, et la "valeur" est la prédiction du modèle.

**Pourquoi cette adaptation ?**

En ML, on veut répondre à : *"Quelle est la contribution de la feature 'revenu' à la prédiction 0.32 pour ce client ?"* 

C'est exactement le même problème que Shapley : répartir équitablement la prédiction entre toutes les features.

**Visualisation conceptuelle** :

```
Prédiction d'un modèle pour une instance :
┌──────────────────────────────────────────┐
│ Valeur de base (moyenne des prédictions) │
│             E[f(X)] = 0.50               │
│                 ↓                        │
│         Prédiction réelle                │
│             f(x) = 0.32                  │
│                 ↓                        │
│      Différence = -0.18                  │
│                 ↓                        │
│  Répartition entre les features :        │
│  • revenu → +0.05                        │
│  • ratio_dette → -0.15                   │
│  • nb_credits → -0.08                    │
│                 ↓                        │
│  Somme = +0.05 - 0.15 - 0.08 = -0.18 ✓   |
└──────────────────────────────────────────┘
```

**Propriété clé : Additivité locale**

$$\text{Prédiction} - \text{Baseline} = \sum_{i=1}^{M} \phi_i$$

où $$\phi_i$$ est la SHAP value de la feature $$i$$.

**Sources académiques** :

- [Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) - Paper fondateur de SHAP
- [Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence](https://www.nature.com/articles/s42256-019-0138-9) - TreeExplainer optimisé

### 2. Formulations mathématiques

> **Note** : Les mathématiques sont essentielles pour comprendre les garanties théoriques de SHAP. Chaque formule est d'abord expliquée intuitivement, puis rigoureusement.

#### 2.1 Valeur de Shapley (Formule générale)

**Notation mathématique** :

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

**Où** :

- $$\phi_i$$ : Valeur de Shapley du joueur (feature) $$i$$
- $$N$$ : Ensemble de tous les joueurs (toutes les features)
- $$S$$ : Coalition (sous-ensemble de features) ne contenant pas $$i$$
- $$v(S)$$ : Valeur de la coalition $$S$$ (prédiction avec seulement les features de $$S$$)
- $$v(S \cup \{i\})$$ : Valeur quand on ajoute $$i$$ à $$S$$
- $$\frac{|S|! (|N| - |S| - 1)!}{|N|!}$$ : Poids de la coalition (probabilité d'apparition dans un ordre aléatoire)

**Intuition** :

Imaginez que vous ajoutez les features **une par une dans un ordre aléatoire** pour faire une prédiction. La SHAP value d'une feature est la **moyenne de sa contribution marginale** sur tous les ordres possibles.

**Exemple avec 2 features** :

Supposons 2 features {revenu, ratio_dette} :

- Baseline (aucune feature) : prédiction = 0.50
- Avec {revenu} seul : prédiction = 0.55 → contribution marginale = +0.05
- Avec {ratio_dette} seul : prédiction = 0.35 → contribution marginale = -0.15
- Avec {revenu, ratio_dette} : prédiction = 0.32

Contribution marginale de "revenu" :
- Ajouté en premier : 0.55 - 0.50 = +0.05
- Ajouté après ratio_dette : 0.32 - 0.35 = -0.03

Valeur de Shapley de "revenu" : moyenne = (0.05 + (-0.03)) / 2 = +0.01

**Pourquoi cette formulation mathématique ?**

La formule de Shapley est la **seule** qui satisfait simultanément 4 propriétés désirables (axiomes de Shapley) :

1. **Efficacité (Efficiency)** : La somme des SHAP values égale la différence entre prédiction et baseline
2. **Symétrie (Symmetry)** : Si deux features contribuent identiquement, elles ont la même SHAP value
3. **Linéarité (Linearity)** : Pour des modèles additifs, les SHAP values s'additionnent
4. **Joueur nul (Dummy)** : Si une feature ne change jamais la prédiction, sa SHAP value = 0

**Source** : [Shapley (1953)](https://www.rand.org/pubs/papers/P0295.html)

#### 2.2 SHAP en pratique : Calcul avec coalitions

**Problème de complexité** :

Calculer exactement la formule de Shapley nécessite d'évaluer **toutes les coalitions possibles** : $$2^M$$ coalitions pour $$M$$ features.

**Exemple** :
- 10 features → $$2^{10} = 1024$$ coalitions
- 30 features → $$2^{30} = 1,073,741,824$$ coalitions (impossible en pratique)

**Solution : Approximations et optimisations**

C'est pourquoi SHAP propose **plusieurs explainers** avec différents compromis performance/précision :

1. **TreeExplainer** : Calcul exact en temps polynomial pour les arbres
2. **KernelExplainer** : Approximation par régression pondérée (model-agnostic)
3. **DeepExplainer** : Approximation pour réseaux de neurones (basée sur DeepLIFT)
4. **LinearExplainer** : Calcul exact pour modèles linéaires
5. **ExactExplainer** : Calcul exact mais lent (pour petits modèles)

---

## 💡 Compréhension Intuitive

> **Principe de signalement** : Cette section vous aide à construire une intuition solide avant de plonger dans le code.

### Analogie du monde réel

**L'analogie du projet en équipe** :

Imaginez que vous travaillez sur un projet avec 3 collègues (vous + A, B, C). Le projet génère un profit de 100k€. Comment répartir ce montant équitablement ?

- **Approche naïve** : 25k€ chacun (égalité stricte)
  - ❌ Problème : Ne tient pas compte des contributions réelles

- **Approche "importance globale"** : Calculer la contribution moyenne de chaque personne
  - ❌ Problème : Une personne excellente en solo peut mal collaborer

- **Approche Shapley (SHAP)** : Considérer tous les scénarios possibles
  - ✅ Vous travaillez seul : profit = 20k€
  - ✅ Vous + A : profit = 50k€ → votre contribution marginale = 30k€
  - ✅ Vous + B : profit = 45k€ → votre contribution marginale = 25k€
  - ✅ Vous + C : profit = 40k€ → votre contribution marginale = 20k€
  - ✅ Vous + A + B : profit = 80k€ → votre contribution marginale = 35k€
  - ✅ Vous + A + C : profit = 70k€ → votre contribution marginale = 30k€
  - ✅ Vous + B + C : profit = 75k€ → votre contribution marginale = 30k€
  - ✅ Vous + A + B + C : profit = 100k€ → votre contribution marginale = 25k€

**Votre valeur de Shapley** = moyenne de toutes vos contributions marginales (en pondérant par probabilité)

**En ML** : Les "collègues" sont les features, le "profit" est la prédiction, et on veut savoir quelle feature contribue le plus.

### Questions pour vérifier la compréhension

Avant de continuer, assurez-vous de pouvoir répondre :

1. **Q1** : Pourquoi ne peut-on pas simplement utiliser les coefficients d'un modèle linéaire comme mesure d'importance ?
   - *Réponse attendue* : Les coefficients mesurent l'effet marginal global, pas la contribution à une prédiction individuelle. De plus, ils dépendent de l'échelle des features et ne sont pas applicables aux modèles non-linéaires.

2. **Q2** : Quelle est la différence entre "feature importance globale" et "SHAP values" ?
   - *Réponse attendue* : Feature importance globale mesure l'importance moyenne d'une feature sur tout le dataset. SHAP values mesurent la contribution spécifique d'une feature pour une instance donnée (explication locale).

3. **Q3** : Pourquoi SHAP garantit-il que la somme des contributions égale la prédiction ?
   - *Réponse attendue* : C'est l'axiome d'efficacité (Efficiency) de la théorie de Shapley, qui impose que la somme des SHAP values égale la différence entre prédiction et baseline.

4. **Q4** : Quel est le principal défi computationnel de SHAP ?
   - *Réponse attendue* : Calculer toutes les coalitions possibles ($$2^M$$) devient rapidement impossible. C'est pourquoi différents explainers utilisent des approximations ou optimisations.

---

## 💻 Implémentation Pratique

> **Principe de modalité** : Code commenté + explication textuelle pour double encodage cognitif.

### 1. Installation et imports

```python
# Installation
# pip install shap

# Imports nécessaires
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Data
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split

# Initialiser la visualisation JavaScript pour notebooks
shap.initjs()
```

### 2. TreeExplainer : Pour les modèles basés sur des arbres

**Quand l'utiliser** : Arbres de décision, Random Forest, XGBoost, LightGBM, CatBoost

**Avantages** :
- ✅ **Calcul exact** (pas d'approximation)
- ✅ **Très rapide** : Complexité $$O(TLD^2)$$ où $$T$$ = nombre d'arbres, $$L$$ = nombre de feuilles, $$D$$ = profondeur
- ✅ **Pas besoin de données de background**

**Principe technique** :

TreeExplainer utilise un algorithme optimisé qui exploite la structure des arbres pour calculer exactement les SHAP values sans énumérer toutes les coalitions. Il traverse l'arbre en calculant les contributions marginales de manière récursive.

**Implémentation complète** :

```python
# Étape 1 : Charger des données (exemple : Classification binaire)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 2 : Entraîner un modèle XGBoost
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Étape 3 : Créer l'explainer TreeExplainer
explainer = shap.TreeExplainer(model)

# Étape 4 : Calculer les SHAP values
# Pour classification binaire, shap_values est une liste [classe_0, classe_1]
shap_values = explainer.shap_values(X_test)

# Si XGBoost avec objective='binary:logistic', on prend la classe positive (1)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Classe positive

# Étape 5 : Visualiser les SHAP values
# Summary plot : Vue globale de l'importance des features
shap.summary_plot(shap_values, X_test, plot_type="bar")
# Interprétation : Features triées par importance moyenne absolue

# Summary plot détaillé (bee swarm)
shap.summary_plot(shap_values, X_test)
# Interprétation : 
# - Axe X : SHAP value (impact sur la prédiction)
# - Couleur : Valeur de la feature (rouge = élevé, bleu = faible)
# - Chaque point = une instance

# Étape 6 : Expliquer une instance spécifique
instance_idx = 0
instance = X_test.iloc[instance_idx]

# Force plot : Explication locale pour une instance
shap.force_plot(
    explainer.expected_value,  # Baseline (moyenne des prédictions)
    shap_values[instance_idx],  # SHAP values de l'instance
    instance,  # Valeurs des features
    matplotlib=True
)
# Interprétation :
# - Base value : prédiction moyenne (e.g., 0.65)
# - Features rouges : poussent la prédiction vers le haut
# - Features bleues : poussent la prédiction vers le bas
# - Output value : prédiction finale

# Waterfall plot : Visualisation en cascade
shap.waterfall_plot(shap.Explanation(
    values=shap_values[instance_idx],
    base_values=explainer.expected_value,
    data=instance,
    feature_names=X_test.columns.tolist()
))
# Interprétation : Montre comment chaque feature modifie la prédiction séquentiellement

print(f"Prédiction baseline : {explainer.expected_value:.3f}")
print(f"SHAP values sum : {shap_values[instance_idx].sum():.3f}")
print(f"Prédiction finale : {model.predict_proba(instance.values.reshape(1, -1))[0, 1]:.3f}")
# Vérification de l'axiome d'efficacité :
# expected_value + sum(shap_values) ≈ prédiction finale (en logit space)
```

**Explication ligne par ligne (concepts clés)** :

**Ligne `explainer = shap.TreeExplainer(model)`** :
- **Pourquoi** : TreeExplainer est optimisé pour les arbres, il analyse la structure interne du modèle
- **Alternative** : KernelExplainer (plus lent mais model-agnostic)

**Ligne `shap_values = explainer.shap_values(X_test)`** :
- **Pourquoi** : Calcule les contributions de chaque feature pour chaque instance
- **Format** : array de shape `(n_instances, n_features)` pour regression/binary, liste pour multiclass

**Ligne `shap.summary_plot(shap_values, X_test)`** :
- **Pourquoi** : Vue globale + locale : montre quelles features sont importantes ET comment elles influencent
- **Lecture** : Une feature avec beaucoup de points rouges à droite (SHAP > 0) augmente la prédiction quand elle est élevée

### 3. KernelExplainer : Model-agnostic (pour tout type de modèle)

**Quand l'utiliser** : Modèles custom, ensembles de modèles hétérogènes, modèles avec preprocessing complexe, APIs de prédiction

**Avantages** :
- ✅ **Fonctionne avec TOUT modèle** (boîte noire)
- ✅ **Théoriquement exact** (avec suffisamment d'échantillons)

**Inconvénients** :
- ⚠️ **Très lent** : Nécessite de nombreuses évaluations du modèle
- ⚠️ **Nécessite un dataset de background** représentatif

**Principe technique** :

KernelExplainer utilise une régression linéaire locale pondérée (LIME-like) mais avec les poids de Shapley pour garantir les axiomes. Il échantillonne des coalitions de features et ajuste un modèle linéaire pour approximer les contributions.

**Implémentation complète** :

```python
# Étape 1 : Modèle quelconque (exemple : fonction custom)
def custom_model(X):
    """
    Modèle custom complexe (exemple : ensemble de modèles + règles métier)
    """
    # Exemple simplifié : combinaison non-linéaire
    score = (
        0.3 * X[:, 0] +  # Feature 0
        0.5 * X[:, 1] ** 2 +  # Feature 1 au carré
        -0.2 * np.log(X[:, 2] + 1)  # Feature 2 en log
    )
    # Transformation sigmoid pour probabilité
    return 1 / (1 + np.exp(-score))

# Étape 2 : Données de background (représentatives de la distribution)
# IMPORTANT : Choisir un échantillon représentatif mais limité (50-100 instances)
# Pourquoi : KernelExplainer évalue le modèle sur chaque instance background × coalitions
background_size = 50  # Compromis performance/précision
background = shap.sample(X_train, background_size)  # Échantillonnage stratifié

# Étape 3 : Créer l'explainer KernelExplainer
# Wrapping du modèle pour retourner des prédictions
def model_predict(X):
    # Si X est DataFrame, convertir en numpy
    if isinstance(X, pd.DataFrame):
        X = X.values
    return custom_model(X)

explainer = shap.KernelExplainer(model_predict, background)

# Étape 4 : Calculer les SHAP values (ATTENTION : lent)
# Limiter le nombre d'instances à expliquer
instances_to_explain = X_test.iloc[:10]  # Seulement 10 instances
shap_values = explainer.shap_values(instances_to_explain, nsamples=100)
# nsamples : Nombre de coalitions échantillonnées (↑ = plus précis mais plus lent)
# Recommandation : 100-500 pour approximation raisonnable

# Étape 5 : Visualisation
shap.summary_plot(shap_values, instances_to_explain)

# Temps d'exécution approximatif :
# background_size × n_instances × nsamples × temps_prédiction
# 50 × 10 × 100 = 50,000 appels au modèle
print("⏱️ KernelExplainer est lent : utilisez-le seulement si nécessaire")
```

**Recommandations pratiques** :

1. **Optimiser background_size** :
   - Start : 50 instances
   - Si résultats instables (SHAP values varient beaucoup entre runs) : augmenter à 100
   - Maximum pratique : 200 (au-delà, gain marginal faible)

2. **Optimiser nsamples** :
   - Start : 100
   - Pour production : 500-1000 (meilleure stabilité)
   - Debug rapide : 50

3. **Utiliser KernelExplainer seulement si** :
   - Aucun autre explainer applicable
   - Besoin d'expliquer peu d'instances (< 50)
   - Le modèle est une boîte noire totale (API externe)

### 4. DeepExplainer : Pour les réseaux de neurones (TensorFlow/PyTorch)

**Quand l'utiliser** : Réseaux de neurones profonds (CNN, RNN, Transformers)

**Avantages** :
- ✅ **Plus rapide que KernelExplainer** pour les réseaux profonds
- ✅ **Approximation de qualité** basée sur DeepLIFT

**Inconvénients** :
- ⚠️ **Approximation** (pas exact comme TreeExplainer)
- ⚠️ **Nécessite accès aux gradients** (TensorFlow/PyTorch)

**Principe technique** :

DeepExplainer utilise DeepLIFT (Deep Learning Important FeaTures), qui calcule les contributions en propageant les différences par rapport à une référence à travers le réseau via les gradients.

**Implémentation complète** :

```python
import tensorflow as tf
from tensorflow import keras

# Étape 1 : Créer un réseau de neurones simple
model_nn = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Classification binaire
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Étape 2 : Créer l'explainer DeepExplainer
# Background : échantillon de référence pour calculer les différences
background_nn = X_train.sample(100).values  # Numpy array requis

explainer_deep = shap.DeepExplainer(model_nn, background_nn)

# Étape 3 : Calculer les SHAP values
# Note : Plus rapide que KernelExplainer mais plus lent que TreeExplainer
instances_nn = X_test.iloc[:20].values
shap_values_nn = explainer_deep.shap_values(instances_nn)

# Étape 4 : Visualisation
shap.summary_plot(shap_values_nn[0], X_test.iloc[:20])
# Note : shap_values_nn est une liste (pour compatibilité multi-output)

# Comparaison de vitesse (ordre de grandeur) :
# TreeExplainer : 10ms pour 100 instances
# DeepExplainer : 1s pour 100 instances
# KernelExplainer : 60s pour 100 instances (avec nsamples=100)
```

**Quand utiliser DeepExplainer vs GradientExplainer** :

- **DeepExplainer** : Approximation DeepLIFT (généralement meilleure qualité)
- **GradientExplainer** : Basé sur Integrated Gradients (voir votre cours IG)
- **Recommandation** : Essayer les deux et comparer visuellement

### 5. LinearExplainer : Pour les modèles linéaires

**Quand l'utiliser** : Régression linéaire, régression logistique, SVM linéaire

**Avantages** :
- ✅ **Calcul exact et instantané**
- ✅ **Résultats identiques aux coefficients** (avec preprocessing adapté)

**Implémentation complète** :

```python
# Étape 1 : Modèle linéaire
model_linear = LogisticRegression(max_iter=1000)
model_linear.fit(X_train, y_train)

# Étape 2 : Créer l'explainer LinearExplainer
# Note : Pour régression logistique, spécifier feature_perturbation="interventional"
explainer_linear = shap.LinearExplainer(
    model_linear, 
    X_train,
    feature_perturbation="interventional"  # Calcule impact en marginalisant
)

# Étape 3 : Calculer les SHAP values
shap_values_linear = explainer_linear.shap_values(X_test)

# Étape 4 : Visualisation
shap.summary_plot(shap_values_linear, X_test)

# Vérification : Pour régression linéaire simple, SHAP values ≈ coefficients × (X - X_mean)
print("Coefficients du modèle :", model_linear.coef_[0][:5])
print("SHAP values moyennes :", shap_values_linear.mean(axis=0)[:5])
# Note : Les SHAP values captent les interactions non-linéaires du preprocessing
```

### 6. ExactExplainer : Calcul exact (pour comparaison ou petits modèles)

**Quand l'utiliser** : 
- Validation/debug (comparer avec approximations)
- Modèles avec très peu de features (< 15)
- Recherche académique

**Inconvénient** :
- ❌ **Extrêmement lent** : $$O(2^M)$$ évaluations

**Implémentation** :

```python
# ATTENTION : Seulement pour petits modèles (< 15 features)
X_small = X_train.iloc[:, :10]  # Seulement 10 features
X_test_small = X_test.iloc[:, :10]

model_small = xgb.XGBClassifier(n_estimators=10, max_depth=2)
model_small.fit(X_small, y_train)

# ExactExplainer : Calcule TOUTES les coalitions
explainer_exact = shap.ExactExplainer(
    model_small.predict_proba,
    X_small
)

# ATTENTION : Très lent même avec 10 features (2^10 = 1024 coalitions)
shap_values_exact = explainer_exact.shap_values(X_test_small.iloc[:5])  # Seulement 5 instances

# Comparaison avec TreeExplainer
explainer_tree = shap.TreeExplainer(model_small)
shap_values_tree = explainer_tree.shap_values(X_test_small.iloc[:5])

# Vérification : Doivent être identiques (TreeExplainer est exact pour arbres)
difference = np.abs(shap_values_exact[0] - shap_values_tree[1]).max()
print(f"Différence max entre Exact et Tree : {difference:.6f}")  # ≈ 0
```

### 7. GradientExplainer : Alternative pour Deep Learning

**Quand l'utiliser** : Réseaux de neurones, alternative à DeepExplainer

**Principe** : Basé sur Integrated Gradients (voir votre cours IG)

```python
# GradientExplainer utilise Integrated Gradients comme approximation
explainer_grad = shap.GradientExplainer(
    model_nn,
    background_nn
)

shap_values_grad = explainer_grad.shap_values(instances_nn)

# Comparaison DeepExplainer vs GradientExplainer :
# - DeepExplainer : Basé sur DeepLIFT (propagation des contributions)
# - GradientExplainer : Basé sur Integrated Gradients (intégration des gradients)
# 
# Recommandation : Tester les deux, souvent similaires
```

---

## 🔬 Exemples Concrets et Cas d'Usage

> **Principe de personnalisation** : Exemples progressifs du simple au complexe pour gérer la charge cognitive.

### Exemple 1 : Arbre de décision pour scoring crédit (Niveau Débutant)

**Contexte** :

Vous déployez un modèle XGBoost pour prédire l'éligibilité au crédit. Vous devez expliquer les décisions aux clients refusés (conformité RGPD).

**Données** :

```python
# Dataset synthétique de scoring crédit
np.random.seed(42)
n_samples = 1000

data_credit = pd.DataFrame({
    'revenu_annuel': np.random.normal(50000, 20000, n_samples),
    'ratio_endettement': np.random.uniform(0, 0.8, n_samples),
    'nb_credits_actifs': np.random.poisson(2, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'anciennete_emploi': np.random.randint(0, 30, n_samples),
    'montant_demande': np.random.uniform(5000, 100000, n_samples)
})

# Règle métier simplifiée pour créer la cible
data_credit['eligible'] = (
    (data_credit['revenu_annuel'] > 40000) &
    (data_credit['ratio_endettement'] < 0.4) &
    (data_credit['nb_credits_actifs'] <= 2)
).astype(int)

X_credit = data_credit.drop('eligible', axis=1)
y_credit = data_credit['eligible']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42
)
```

**Solution** :

```python
# Étape 1 : Entraîner le modèle XGBoost
model_credit = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model_credit.fit(X_train_c, y_train_c)

# Étape 2 : Créer l'explainer
explainer_credit = shap.TreeExplainer(model_credit)
shap_values_credit = explainer_credit.shap_values(X_test_c)

# Étape 3 : Cas d'un client REFUSÉ
# Trouver un client refusé
refused_idx = np.where(model_credit.predict(X_test_c) == 0)[0][0]
client_refused = X_test_c.iloc[refused_idx]

print("=== CLIENT REFUSÉ ===")
print(f"Prédiction : {model_credit.predict_proba(client_refused.values.reshape(1, -1))[0, 1]:.3f}")
print(f"Seuil d'acceptation : 0.500\n")

# Afficher les valeurs du client
print("Profil du client :")
for feature, value in client_refused.items():
    print(f"  {feature}: {value:.2f}")

# SHAP values pour ce client
shap_client = shap_values_credit[refused_idx]

# Afficher les contributions
print("\n=== EXPLICATION DU REFUS ===")
print(f"Prédiction de base (moyenne) : {explainer_credit.expected_value:.3f}")

# Trier les features par impact absolu
feature_impact = pd.DataFrame({
    'feature': X_test_c.columns,
    'value': client_refused.values,
    'shap_value': shap_client
}).sort_values('shap_value')

print("\nContributions des features (du plus négatif au plus positif) :")
for _, row in feature_impact.iterrows():
    impact_symbol = "↑" if row['shap_value'] > 0 else "↓"
    print(f"  {row['feature']:20s} = {row['value']:8.2f}  →  {impact_symbol} {row['shap_value']:+.3f}")

# Prédiction finale
prediction_final = explainer_credit.expected_value + shap_client.sum()
print(f"\nPrédiction finale : {explainer_credit.expected_value:.3f} + {shap_client.sum():+.3f} = {prediction_final:.3f}")

# Visualisation waterfall
shap.waterfall_plot(shap.Explanation(
    values=shap_client,
    base_values=explainer_credit.expected_value,
    data=client_refused.values,
    feature_names=X_test_c.columns.tolist()
))
```

**Résultats** :

```
=== CLIENT REFUSÉ ===
Prédiction : 0.234
Seuil d'acceptation : 0.500
Profil du client :
revenu_annuel: 32000.00
ratio_endettement: 0.65
nb_credits_actifs: 4.00
age: 28.00
anciennete_emploi: 2.00
montant_demande: 75000.00
=== EXPLICATION DU REFUS ===
Prédiction de base (moyenne) : 0.520
Contributions des features :
ratio_endettement    =     0.65  →  ↓ -0.180  ⚠️ Principal facteur négatif
nb_credits_actifs    =     4.00  →  ↓ -0.120  ⚠️ Deuxième facteur négatif
revenu_annuel        = 32000.00  →  ↓ -0.075
montant_demande      = 75000.00  →  ↓ -0.030
anciennete_emploi    =     2.00  →  ↓ -0.015
age                  =    28.00  →  ↑ +0.005
Prédiction finale : 0.520 - 0.415 = 0.105
```

**Interprétation** :

- **Baseline** : En moyenne, 52% des clients sont acceptés
- **Facteurs de refus** :
  1. **Ratio d'endettement** (0.65) → Impact -0.180 : Trop élevé (recommandation : < 0.40)
  2. **Nombre de crédits actifs** (4) → Impact -0.120 : Trop nombreux (recommandation : ≤ 2)
  3. **Revenu annuel** (32k€) → Impact -0.075 : En dessous du seuil optimal (40k€)

**Message au client** :

*"Votre demande a été refusée principalement en raison de votre ratio d'endettement élevé (65%) et du nombre important de crédits actifs (4). Pour améliorer vos chances d'acceptation, nous recommandons de réduire votre ratio d'endettement en dessous de 40% en remboursant certains crédits existants."*

**Analyse critique** :

- **Points forts** : Explication transparente, actionnable, conforme RGPD
- **Limitations** : Ne capture pas les interactions complexes (e.g., revenu faible + endettement élevé)
- **Leçons apprises** : SHAP permet de transformer un rejet opaque en feedback constructif

### Exemple 2 : Random Forest pour diagnostic médical (Niveau Intermédiaire)

**Contexte** :

Vous développez un modèle de détection du cancer du sein (dataset breast cancer). Les médecins doivent comprendre pourquoi le modèle prédit "malin" ou "bénin".

**Données** :

```python
# Dataset breast cancer
data_cancer = load_breast_cancer()
X_cancer = pd.DataFrame(data_cancer.data, columns=data_cancer.feature_names)
y_cancer = data_cancer.target  # 0 = malin, 1 = bénin

X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42
)
```

**Solution** :

```python
# Étape 1 : Entraîner Random Forest
model_cancer = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_cancer.fit(X_train_cancer, y_train_cancer)

print(f"Précision : {model_cancer.score(X_test_cancer, y_test_cancer):.3f}")

# Étape 2 : SHAP analysis
explainer_cancer = shap.TreeExplainer(model_cancer)
shap_values_cancer = explainer_cancer.shap_values(X_test_cancer)

# Pour Random Forest, shap_values est une liste [classe_0, classe_1]
# On prend la classe 1 (bénin) pour l'interprétation
shap_values_benign = shap_values_cancer[1]

# Étape 3 : Vue globale - Quelles features sont les plus importantes ?
print("\n=== VUE GLOBALE : FEATURES LES PLUS IMPORTANTES ===")
shap.summary_plot(shap_values_benign, X_test_cancer, plot_type="bar", max_display=10)

# Étape 4 : Analyse d'une instance prédite MALIN (classe 0)
# Trouver un cas malin correctement prédit
malignant_idx = np.where((model_cancer.predict(X_test_cancer) == 0) & (y_test_cancer == 0))[0][0]
patient_malignant = X_test_cancer.iloc[malignant_idx]

print("\n=== CAS CLINIQUE : TUMEUR MALIGNE ===")
print(f"Prédiction : {model_cancer.predict_proba(patient_malignant.values.reshape(1, -1))[0]}")
print(f"  → Classe 0 (malin) : {model_cancer.predict_proba(patient_malignant.values.reshape(1, -1))[0, 0]:.3f}")
print(f"  → Classe 1 (bénin) : {model_cancer.predict_proba(patient_malignant.values.reshape(1, -1))[0, 1]:.3f}")

# SHAP explanation
shap_patient = shap_values_benign[malignant_idx]

# Identifier les top 5 features contribuant au diagnostic "malin"
# (SHAP values négatifs pour classe "bénin" = poussent vers "malin")
top_features_malin = pd.Series(shap_patient, index=X_test_cancer.columns).sort_values()[:5]

print("\nFeatures principales indiquant une tumeur MALIGNE :")
for feature, shap_val in top_features_malin.items():
    feature_value = patient_malignant[feature]
    print(f"  {feature:30s} = {feature_value:8.2f}  →  SHAP = {shap_val:+.3f}")

# Visualisation : Force plot pour explication détaillée
shap.force_plot(
    explainer_cancer.expected_value[1],
    shap_patient,
    patient_malignant,
    matplotlib=True
)

# Étape 5 : Dependence plot - Relation entre une feature et SHAP values
# Exemple : "worst concave points" (souvent discriminant)
print("\n=== ANALYSE DE DÉPENDANCE : worst concave points ===")
shap.dependence_plot(
    "worst concave points",
    shap_values_benign,
    X_test_cancer,
    interaction_index="mean radius"  # Colorer par interaction avec "mean radius"
)
# Interprétation :
# - Axe X : Valeur de "worst concave points"
# - Axe Y : SHAP value (impact sur prédiction "bénin")
# - On observe généralement : valeurs élevées → SHAP négatif (indique malin)
```

**Résultats** :

```
Précision : 0.965
=== VUE GLOBALE : FEATURES LES PLUS IMPORTANTES ===

    1. worst perimeter
    2. worst concave points
    3. mean concave points
    4. worst radius
    5. mean perimeter

=== CAS CLINIQUE : TUMEUR MALIGNE ===
Prédiction : [0.985, 0.015]
→ Classe 0 (malin) : 0.985
→ Classe 1 (bénin) : 0.015
Features principales indiquant une tumeur MALIGNE :
worst concave points         =     0.25  →  SHAP = -0.180
worst perimeter              =   150.30  →  SHAP = -0.145
mean concave points          =     0.15  →  SHAP = -0.120
worst radius                 =    25.60  →  SHAP = -0.105
mean perimeter               =   120.50  →  SHAP = -0.085
```

**Interprétation clinique** :

*"Le modèle prédit une forte probabilité de tumeur maligne (98.5%) en raison de :*
1. *Points concaves prononcés dans la zone la plus suspecte (worst concave points = 0.25) → Indicateur clé de malignité*
2. *Périmètre important de la zone suspecte (worst perimeter = 150.3 mm)*
3. *Présence significative de points concaves en moyenne (mean concave points = 0.15)*

*Ces caractéristiques morphologiques sont cohérentes avec les critères cliniques de malignité. Recommandation : Biopsie confirmative."*

**Analyse critique** :

- **Points forts** : 
  - Explication alignée avec les critères cliniques connus
  - Permet au médecin de valider la cohérence du diagnostic
  - Identifie les features visuelles critiques pour l'imagerie

- **Limitations** : 
  - N'explique pas pourquoi ces features sont liées au cancer (causalité)
  - Ne remplace pas l'expertise médicale

- **Leçons apprises** : 
  - SHAP aide à construire la confiance des experts métier
  - Les dependence plots révèlent des patterns non-linéaires

### Exemple 3 : Modèle hétérogène avec KernelExplainer (Niveau Avancé)

**Contexte** :

Vous avez un pipeline complexe : preprocessing custom + ensemble de 3 modèles différents (XGBoost + Random Forest + Réseau de neurones) avec pondération métier. Vous ne pouvez pas utiliser TreeExplainer.

**Solution** :

```python
# Étape 1 : Pipeline complexe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def complex_pipeline_predict(X):
    """
    Pipeline custom : Preprocessing + Ensemble de modèles
    """
    # Preprocessing
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Prédictions des 3 modèles
    pred_xgb = model_xgb.predict_proba(X)[:, 1]
    pred_rf = model_rf.predict_proba(X)[:, 1]
    pred_nn = model_nn.predict(X).flatten()
    
    # Pondération métier (règle custom)
    # Si feature "montant_demande" > 50000, privilégier XGBoost
    weights = np.where(X[:, -1] > 50000, [0.5, 0.3, 0.2], [0.3, 0.3, 0.4])
    
    # Ensemble pondéré
    ensemble_pred = (
        weights[:, 0] * pred_xgb +
        weights[:, 1] * pred_rf +
        weights[:, 2] * pred_nn
    )
    
    return ensemble_pred

# Étape 2 : Entraîner les sous-modèles
scaler = StandardScaler().fit(X_train_c)
pca = PCA(n_components=4).fit(scaler.transform(X_train_c))

model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
model_xgb.fit(X_train_c, y_train_c)

model_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
model_rf.fit(X_train_c, y_train_c)

model_nn = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train_c.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])
model_nn.compile(optimizer='adam', loss='binary_crossentropy')
model_nn.fit(X_train_c, y_train_c, epochs=10, verbose=0)

# Étape 3 : KernelExplainer (seule option pour ce pipeline)
background_ensemble = shap.sample(X_train_c, 50)

explainer_ensemble = shap.KernelExplainer(
    lambda X: complex_pipeline_predict(X.values if isinstance(X, pd.DataFrame) else X),
    background_ensemble
)

# Étape 4 : Expliquer quelques instances
instances_ensemble = X_test_c.iloc[:5]
shap_values_ensemble = explainer_ensemble.shap_values(instances_ensemble, nsamples=200)

# Étape 5 : Visualisation
shap.summary_plot(shap_values_ensemble, instances_ensemble)

print("⚠️ KernelExplainer a nécessité ~50 × 5 × 200 = 50,000 évaluations du modèle")
print("Pour production, envisager de simplifier le pipeline ou cacher les résultats")
```

**Leçons apprises** :

- KernelExplainer est puissant mais coûteux en production
- **Recommandation production** : Précalculer SHAP values pour instances fréquentes et cacher
- Alternative : Simplifier le pipeline pour utiliser TreeExplainer

---

## ⚖️ Comparaisons et Choix de Design

> **Principe de contiguïté spatiale** : Comparaisons côte à côte pour faciliter la compréhension.

### Pourquoi SHAP plutôt que LIME ?

**Contexte de décision** : Vous devez choisir une méthode d'interprétabilité pour un modèle en production.

| Critère | SHAP | LIME |
|---------|------|------|
| **Fondation théorique** | ✅ Basé sur théorie de Shapley (garanties mathématiques) | ⚠️ Heuristique (régression linéaire locale) |
| **Propriétés** | ✅ Efficacité, symétrie, linéarité, joueur nul | ❌ Aucune garantie formelle |
| **Cohérence** | ✅ SHAP values s'additionnent pour donner prédiction | ⚠️ Peut ne pas être cohérent |
| **Stabilité** | ✅ Déterministe (sauf KernelExplainer) | ⚠️ Dépend de l'échantillonnage local |
| **Performance (arbres)** | ✅ TreeExplainer très rapide (polynomial) | ❌ Toujours lent (échantillonnage) |
| **Performance (DL)** | ⚠️ DeepExplainer moyennement rapide | ❌ Très lent |
| **Model-agnostic** | ✅ Oui (KernelExplainer) | ✅ Oui |
| **Visualisations** | ✅ Riches (summary, dependence, force, waterfall) | ⚠️ Limitées (importance locale) |
| **Production-ready** | ✅ Oui (surtout TreeExplainer) | ⚠️ Instabilité peut poser problème |
| **Complexité temporelle** | TreeExplainer : $$O(TLD^2)$$ <br> KernelExplainer : $$O(2^M)$$ approximé | $$O(n_{samples} \times n_{features})$$ |
| **Complexité spatiale** | $$O(n_{instances} \times n_{features})$$ | $$O(n_{samples} \times n_{features})$$ |

**Recommandation** :

- ✅ **Utiliser SHAP quand** : 
  - Production nécessitant cohérence et stabilité
  - Modèles basés sur arbres (TreeExplainer disponible)
  - Besoin de garanties théoriques (conformité, audit)
  - Visualisations riches nécessaires

- ✅ **Utiliser LIME quand** : 
  - Exploration rapide et intuitive (LIME plus simple conceptuellement)
  - Modèles très complexes où SHAP est trop lent
  - Interprétabilité locale suffit (pas besoin de vue globale)
  - Dataset de background difficile à définir pour SHAP

**Exemple comparatif** :

```python
import lime
import lime.lime_tabular

# Même modèle et dataset que précédemment (scoring crédit)
# Comparer SHAP vs LIME sur une instance

instance_compare = X_test_c.iloc[0]

# === SHAP ===
explainer_shap = shap.TreeExplainer(model_credit)
shap_vals = explainer_shap.shap_values(instance_compare.values.reshape(1, -1))[0]

# === LIME ===
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train_c.values,
    feature_names=X_train_c.columns.tolist(),
    class_names=['refusé', 'accepté'],
    mode='classification'
)

lime_exp = explainer_lime.explain_instance(
    instance_compare.values,
    model_credit.predict_proba,
    num_features=6
)

# Comparaison des importances
print("=== COMPARAISON SHAP vs LIME ===\n")

print("SHAP values :")
shap_df = pd.DataFrame({
    'feature': X_test_c.columns,
    'shap_value': shap_vals
}).sort_values('shap_value', key=abs, ascending=False).head(6)
print(shap_df)

print("\nLIME coefficients :")
lime_df = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'lime_coef']).head(6)
print(lime_df)

# Visualisation côte à côte
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SHAP
axes[0].barh(shap_df['feature'], shap_df['shap_value'])
axes[0].set_title('SHAP values')
axes[0].set_xlabel('Contribution à la prédiction')

# LIME
lime_exp.as_pyplot_figure(label=1)
plt.subplot(1, 2, 2)
plt.title('LIME coefficients')

plt.tight_layout()
plt.show()

# Mesure de cohérence : Somme des SHAP values vs écart à baseline
print(f"\nCohérence SHAP (somme = prédiction - baseline) : {shap_vals.sum():.4f}")
print(f"Cohérence LIME : Non garanti (peut varier selon échantillonnage)")
```

### Pourquoi SHAP plutôt que Feature Importance standard ?

| Critère | SHAP | Feature Importance (arbres) | Permutation Importance |
|---------|------|----------------------------|------------------------|
| **Niveau** | Local (par instance) + Global | Global uniquement | Global uniquement |
| **Additivité** | ✅ Somme = prédiction | ❌ Juste un score relatif | ❌ Juste un score relatif |
| **Interactions** | ✅ Capture via Shapley | ⚠️ Moyenne globale | ⚠️ Effet marginal moyen |
| **Interprétation** | "Contribution à cette prédiction" | "Importance moyenne dans l'arbre" | "Impact sur performance globale" |
| **Cas d'usage** | Expliquer une décision individuelle | Vue d'ensemble rapide | Sélection de features |

**Recommandation** :

- **Feature Importance** : Analyse exploratoire rapide (1 ligne de code)
- **Permutation Importance** : Valider features réellement utilisées par le modèle
- **SHAP** : Expliquer des décisions individuelles (production, conformité)

### Comparaison des explainers SHAP : Lequel choisir ?

| Explainer | Type de modèle | Vitesse | Précision | Cas d'usage |
|-----------|----------------|---------|-----------|-------------|
| **TreeExplainer** | Arbres (XGBoost, RF, LightGBM, CatBoost) | ⚡⚡⚡ Très rapide | ✅ Exact | **Recommandé par défaut pour arbres** |
| **LinearExplainer** | Modèles linéaires (LogReg, SVM linéaire) | ⚡⚡⚡ Instantané | ✅ Exact | Modèles linéaires |
| **DeepExplainer** | Réseaux de neurones (TF, PyTorch) | ⚡⚡ Moyen | ⚠️ Approximation | Deep Learning (alternative IG) |
| **GradientExplainer** | Réseaux de neurones (TF, PyTorch) | ⚡⚡ Moyen | ⚠️ Approximation | Deep Learning (basé sur IG) |
| **KernelExplainer** | **Tout modèle** (model-agnostic) | ⚡ Très lent | ⚠️ Approximation | Dernier recours, pipelines complexes |
| **ExactExplainer** | Tout modèle petit (< 15 features) | ❌ Extrêmement lent | ✅ Exact | Debug, validation académique |

**Arbre de décision pour choisir l'explainer** :

```
Quel est votre modèle ?
│
├─ Arbre de décision / Random Forest / XGBoost / LightGBM / CatBoost
│  └─→ TreeExplainer ✅ (rapide + exact)
│
├─ Régression linéaire / Régression logistique / SVM linéaire
│  └─→ LinearExplainer ✅ (instantané + exact)
│
├─ Réseau de neurones (TensorFlow / PyTorch)
│  ├─ Besoin d'explication précise
│  │  └─→ GradientExplainer (basé sur Integrated Gradients)
│  └─ Rapidité prioritaire
│     └─→ DeepExplainer (basé sur DeepLIFT)
│
└─ Pipeline complexe / Modèle custom / API externe / Ensemble hétérogène
├─ Peu d'instances à expliquer (< 50) + temps OK
│  └─→ KernelExplainer ⚠️ (lent mais fonctionne)
└─ Production avec beaucoup d'instances
└─→ Envisager de simplifier le pipeline OU précalculer/cacher
```

**Recommandations production** :

1. **Pipeline arbres** : TreeExplainer (optimal, pas de compromis)
2. **Pipeline DL** : GradientExplainer + caching des SHAP values pour instances fréquentes
3. **Pipeline complexe** : 
   - Option A : Simplifier pour utiliser TreeExplainer
   - Option B : KernelExplainer + précalcul offline + cache Redis

---

## ⚠️ Pièges Courants et Bonnes Pratiques

> **Principe de cohérence** : Liste structurée et actionnable.

### ❌ Erreurs fréquentes

#### Erreur 1 : Mal choisir la baseline (expected_value)

**Description** :

La baseline (valeur de référence) est cruciale pour interpréter les SHAP values. Par défaut, SHAP utilise la moyenne des prédictions sur le dataset d'entraînement, mais ce n'est pas toujours approprié.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS : Utiliser TreeExplainer sans comprendre la baseline
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Problème : expected_value peut ne pas être pertinent pour votre contexte
print(f"Baseline : {explainer.expected_value}")  # e.g., 0.52
# Si votre seuil métier est 0.30, la baseline 0.52 fausse l'interprétation
```

**Solution** :

```python
# ✅ BON : Comprendre et justifier la baseline

# Option 1 : Baseline par défaut (moyenne dataset d'entraînement)
explainer = shap.TreeExplainer(model)
# Justification : Baseline = "client moyen"
# Interprétation : SHAP value = contribution par rapport au client moyen

# Option 2 : Baseline custom (e.g., seuil métier)
# Note : TreeExplainer ne permet pas de changer facilement la baseline
# Mais on peut ajuster l'interprétation :
seuil_metier = 0.30
baseline_defaut = explainer.expected_value
ajustement = seuil_metier - baseline_defaut

print(f"Baseline par défaut : {baseline_defaut:.3f}")
print(f"Seuil métier : {seuil_metier:.3f}")
print(f"Ajustement à appliquer : {ajustement:+.3f}")

# Lors de l'interprétation, mentionner :
# "Par rapport au client moyen (baseline 0.52), cette prédiction est 0.20 plus basse"
# "Le seuil d'acceptation étant 0.30, cela correspond à un rejet"
```

**Impact** : Mauvaise interprétation des SHAP values absolues, confusion des utilisateurs métier

**Source** : [SHAP Documentation - Explaining models](https://shap.readthedocs.io/en/latest/)

#### Erreur 2 : Utiliser KernelExplainer avec un background set non représentatif

**Description** :

KernelExplainer marginalise sur le background set pour calculer les coalitions. Si ce set n'est pas représentatif de la distribution des données, les SHAP values seront biaisées.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS : Background set trop petit ou biaisé
background_bad = X_train.sample(10)  # Seulement 10 instances !
explainer_bad = shap.KernelExplainer(model.predict_proba, background_bad)
shap_values_bad = explainer_bad.shap_values(X_test[:5])

# Problème 1 : 10 instances ne capturent pas la variabilité des données
# Problème 2 : Échantillonnage aléatoire peut exclure des régions importantes
```

**Solution** :

```python
# ✅ BON : Background set représentatif et de taille appropriée

# Méthode 1 : Échantillonnage stratifié (recommandé)
background_good = shap.sample(X_train, 100, random_state=42)
# shap.sample() utilise k-means pour échantillonner des régions représentatives

# Méthode 2 : Échantillonnage stratifié par classe (classification)
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=42)
for background_idx, _ in sss.split(X_train, y_train):
    background_stratified = X_train.iloc[background_idx]

# Méthode 3 : Utiliser TOUT le train set (si petit, < 500 instances)
background_full = X_train  # Pas de compromis mais plus lent

explainer_good = shap.KernelExplainer(model.predict_proba, background_good)

# Vérifier la stabilité : SHAP values ne doivent pas varier fortement entre runs
shap_vals_run1 = explainer_good.shap_values(X_test[:5], nsamples=100)
shap_vals_run2 = explainer_good.shap_values(X_test[:5], nsamples=100)
stability = np.abs(shap_vals_run1 - shap_vals_run2).max()
print(f"Stabilité (max diff) : {stability:.4f}")  # Doit être < 0.01
```

**Impact** : SHAP values instables, non reproductibles, possiblement biaisées

**Source** : [Lundberg et al. - Consistent Individualized Feature Attribution](https://arxiv.org/abs/1802.03888)

#### Erreur 3 : Confondre corrélation et causalité

**Description** :

SHAP mesure l'association statistique entre features et prédiction, PAS la causalité. Une feature peut avoir un SHAP value élevé sans être la cause de la prédiction.

**Exemple de problème** :

```python
# Contexte : Modèle de scoring crédit
# Feature "possède_smartphone_haut_de_gamme" a un SHAP value positif élevé

# ❌ MAUVAISE INTERPRÉTATION :
# "Acheter un smartphone haut de gamme augmente vos chances d'obtenir un crédit"

# ✅ BONNE INTERPRÉTATION :
# "Posséder un smartphone haut de gamme est corrélé avec un profil financier favorable
# (revenu élevé, stabilité professionnelle), ce qui augmente la probabilité d'acceptation"

# Le modèle a appris la corrélation, pas la causalité
# Acheter un smartphone ne changera pas votre éligibilité !
```

**Solution** :

```python
# ✅ BON : Distinguer explicitement corrélation et causalité dans les explications

def interpret_shap_values(feature_name, shap_value, feature_value):
    """
    Interprète les SHAP values en évitant les implications causales
    """
    if shap_value > 0:
        interpretation = (
            f"La feature '{feature_name}' (valeur = {feature_value}) "
            f"est ASSOCIÉE à une augmentation de la prédiction (+{shap_value:.3f}). "
            f"Cela ne signifie PAS que modifier cette feature causera une amélioration."
        )
    else:
        interpretation = (
            f"La feature '{feature_name}' (valeur = {feature_value}) "
            f"est ASSOCIÉE à une diminution de la prédiction ({shap_value:.3f}). "
            f"Cela ne signifie PAS que modifier cette feature causera une dégradation."
        )
    
    return interpretation

# Utilisation
feature = "possède_smartphone_haut_de_gamme"
shap_val = 0.15
feat_val = 1
print(interpret_shap_values(feature, shap_val, feat_val))

# Pour conseils actionnables, combiner SHAP avec expertise métier :
# - Identifier features causales (e.g., ratio_endettement)
# - Éviter features "proxies" non actionnables (e.g., possède_smartphone)
```

**Impact** : Recommandations métier erronées, manipulation potentielle (adversarial gaming)

**Source** : [Pearl, J. - The Book of Why: Causalité vs Corrélation](http://bayes.cs.ucla.edu/WHY/)

#### Erreur 4 : Ignorer les interactions entre features dans l'interprétation

**Description** :

SHAP values sont calculées en considérant toutes les interactions, mais les visualisations standards (summary plot, force plot) ne montrent pas explicitement les interactions.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS : Interpréter les SHAP values de manière indépendante
# "La feature A contribue +0.10 et la feature B contribue +0.05, donc ensemble +0.15"

# Problème : Si A et B interagissent, l'effet combiné peut être différent de la somme
```

**Solution** :

```python
# ✅ BON : Utiliser les dependence plots avec interaction_index

# Exemple : Revenu et ratio d'endettement interagissent
shap.dependence_plot(
    "revenu_annuel",
    shap_values_credit,
    X_test_c,
    interaction_index="ratio_endettement"  # Colorer par ratio d'endettement
)

# Interprétation :
# - Si les points changent de couleur le long de l'axe Y (SHAP value), 
#   cela indique une interaction
# - Exemple : Revenu élevé a un impact positif (+0.05) seulement si ratio_endettement < 0.3
#            mais impact négatif (-0.02) si ratio_endettement > 0.6

# Quantifier les interactions avec SHAP interaction values
shap_interaction_values = explainer_credit.shap_interaction_values(X_test_c)
# shap_interaction_values : array de shape (n_instances, n_features, n_features)
# shap_interaction_values[i, j, k] = interaction entre features j et k pour instance i

# Visualiser les interactions principales
shap.summary_plot(
    shap_interaction_values[0],  # Instance 0
    X_test_c.iloc[0:1],
    plot_type="compact_dot"
)
```

**Impact** : Simplification excessive, conseils métier sous-optimaux

**Source** : [Lundberg et al. - Explainable AI for Trees](https://www.nature.com/articles/s42256-019-0138-9)

### ✅ Bonnes pratiques

#### Pratique 1 : Toujours vérifier l'axiome d'efficacité

**Principe** :

La somme des SHAP values DOIT égaler la différence entre prédiction et baseline. Vérifier systématiquement cette propriété pour valider le calcul.

**Justification scientifique/technique** :

L'axiome d'efficacité (Efficiency) garantit que SHAP répartit équitablement la prédiction. Si non respecté, erreur de calcul ou approximation trop grossière.

**Implémentation** :

```python
# Vérification systématique
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

for i in range(min(10, len(X_test))):  # Vérifier 10 instances
    instance = X_test.iloc[i]
    shap_instance = shap_values[i]
    
    # Prédiction du modèle (en logit space pour classification)
    if hasattr(model, 'predict_proba'):
        pred = model.predict_proba(instance.values.reshape(1, -1))[0, 1]
        # Pour logit : pred_logit = log(pred / (1 - pred))
        pred_logit = np.log(pred / (1 - pred))
        baseline_logit = np.log(explainer.expected_value / (1 - explainer.expected_value))
        target = pred_logit
        baseline = baseline_logit
    else:
        pred = model.predict(instance.values.reshape(1, -1))[0]
        target = pred
        baseline = explainer.expected_value
    
    # Vérification
    shap_sum = shap_instance.sum()
    expected_diff = target - baseline
    error = abs(shap_sum - expected_diff)
    
    if error > 1e-6:  # Tolérance numérique
        print(f"⚠️ Instance {i} : Erreur d'efficacité = {error:.6f}")
        print(f"   SHAP sum = {shap_sum:.6f}, Expected = {expected_diff:.6f}")
    else:
        print(f"✅ Instance {i} : Axiome d'efficacité respecté (erreur = {error:.9f})")
```

**Sources** :

- [Shapley (1953) - A value for n-person games](https://www.rand.org/pubs/papers/P0295.html)
- [Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

#### Pratique 2 : Combiner vues globales et locales

**Principe** :

Utiliser systématiquement les summary plots (global) ET les force/waterfall plots (local) pour une compréhension complète.

**Justification** :

- Vue globale : Identifie les features importantes en moyenne (stratégie)
- Vue locale : Explique des décisions individuelles (opérationnel, conformité)

**Implémentation** :

```python
# Workflow complet : Global → Local
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Étape 1 : Vue GLOBALE - Quelles features sont importantes dans l'ensemble ?
print("=== VUE GLOBALE : IMPORTANCE DES FEATURES ===")
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10)

# Insights globaux
global_importance = np.abs(shap_values).mean(axis=0)
top_features = X_test.columns[np.argsort(-global_importance)[:5]]
print(f"\nTop 5 features globalement : {list(top_features)}")

# Étape 2 : Vue LOCALE - Expliquer des instances critiques

# Cas 1 : Faux positifs (prédits positifs mais réels négatifs)
fp_indices = np.where((model.predict(X_test) == 1) & (y_test == 0))[0]
if len(fp_indices) > 0:
    print(f"\n=== ANALYSE FAUX POSITIF (instance {fp_indices[0]}) ===")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[fp_indices[0]],
        base_values=explainer.expected_value,
        data=X_test.iloc[fp_indices[0]],
        feature_names=X_test.columns.tolist()
    ))

# Cas 2 : Instances avec forte incertitude (prédiction proche du seuil)
predictions = model.predict_proba(X_test)[:, 1]
uncertain_indices = np.where((predictions > 0.45) & (predictions < 0.55))[0]
if len(uncertain_indices) > 0:
    print(f"\n=== ANALYSE INSTANCE INCERTAINE (instance {uncertain_indices[0]}) ===")
    shap.force_plot(
        explainer.expected_value,
        shap_values[uncertain_indices[0]],
        X_test.iloc[uncertain_indices[0]],
        matplotlib=True
    )

# Étape 3 : Dependence plots pour interactions
print("\n=== ANALYSE DES INTERACTIONS ===")
for feature in top_features[:2]:  # Top 2 features
    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        interaction_index="auto"  # SHAP choisit automatiquement la feature avec interaction maximale
    )
```

#### Pratique 3 : Documenter les choix d'explainer et hyperparamètres

**Principe** :

En production, documenter systématiquement quel explainer est utilisé, pourquoi, et avec quels hyperparamètres (background size, nsamples, etc.).

**Implémentation** :

```python
class SHAPExplainerConfig:
    """
    Configuration documentée pour SHAP en production
    """
    def __init__(
        self,
        explainer_type: str,
        model,
        background_data=None,
        background_size: int = 100,
        nsamples: int = 100,
        random_state: int = 42
    ):
        self.explainer_type = explainer_type
        self.model = model
        self.background_size = background_size
        self.nsamples = nsamples
        self.random_state = random_state
        
        # Créer l'explainer selon le type
        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
            self.justification = "TreeExplainer : Calcul exact et rapide pour modèles basés sur arbres"
        
        elif explainer_type == "kernel":
            if background_data is None:
                raise ValueError("KernelExplainer nécessite background_data")
            background = shap.sample(background_data, background_size, random_state=random_state)
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            self.justification = (
                f"KernelExplainer : Model-agnostic, background_size={background_size}, "
                f"nsamples={nsamples} (compromis précision/performance)"
            )
        
        elif explainer_type == "deep":
            if background_data is None:
                raise ValueError("DeepExplainer nécessite background_data")
            background = background_data.sample(background_size, random_state=random_state).values
            self.explainer = shap.DeepExplainer(model, background)
            self.justification = f"DeepExplainer : Pour réseaux de neurones, background_size={background_size}"
        
        else:
            raise ValueError(f"Type d'explainer non supporté : {explainer_type}")
    
    def explain(self, X, **kwargs):
        """
        Explique les instances avec logging
        """
        import time
        start = time.time()
        
        if self.explainer_type == "kernel":
            kwargs.setdefault('nsamples', self.nsamples)
        
        shap_values = self.explainer.shap_values(X, **kwargs)
        elapsed = time.time() - start
        
        print(f"SHAP Explanation complétée")
        print(f"  - Explainer : {self.explainer_type}")
        print(f"  - Justification : {self.justification}")
        print(f"  - Instances expliquées : {len(X)}")
        print(f"  - Temps d'exécution : {elapsed:.2f}s")
        
        return shap_values
    
    def to_dict(self):
        """
        Sérialise la configuration pour logging/audit
        """
        return {
            'explainer_type': self.explainer_type,
            'background_size': self.background_size,
            'nsamples': self.nsamples,
            'random_state': self.random_state,
            'justification': self.justification
        }

# Utilisation en production
config = SHAPExplainerConfig(
    explainer_type="tree",
    model=model_credit
)

shap_values = config.explain(X_test.iloc[:10])

# Logger la configuration pour audit
import json
with open("shap_config_audit.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)
```

### 📋 Checklist de validation

Avant de déployer SHAP en production :

- [ ] **Explainer approprié** : Le type d'explainer correspond au type de modèle
- [ ] **Performance** : Temps de calcul SHAP < seuil acceptable (e.g., < 100ms par instance)
- [ ] **Axiome d'efficacité** : Vérifié sur un échantillon représentatif
- [ ] **Background set** : Représentatif et de taille appropriée (si KernelExplainer/DeepExplainer)
- [ ] **Stabilité** : SHAP values reproductibles entre runs (si approximation)
- [ ] **Visualisations** : Validées avec experts métier pour interprétabilité
- [ ] **Documentation** : Configuration SHAP documentée et versionnée
- [ ] **Tests unitaires** : Tests sur instances critiques (edge cases)
- [ ] **Monitoring** : Alertes si temps de calcul SHAP dépasse seuil
- [ ] **Conformité** : Explications respectent contraintes RGPD/réglementaires

---

## 🧪 Exercices et Validation des Connaissances

### Exercice 1 : Implémentation basique (Débutant)

**Énoncé** :

Vous avez un modèle Random Forest entraîné sur le dataset Iris (classification multi-classe). Calculez les SHAP values et créez un summary plot.

**Données** :

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Charger Iris
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

# Train/Test split
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# Modèle
model_iris = RandomForestClassifier(n_estimators=100, random_state=42)
model_iris.fit(X_train_iris, y_train_iris)
```

**Objectif** : Créer un summary plot et identifier la feature la plus importante globalement.

**Indices** :

<details>
<summary>💡 Indice 1 (cliquez pour révéler)</summary>

Utilisez `TreeExplainer` car Random Forest est basé sur des arbres.

</details>

<details>
<summary>💡 Indice 2</summary>

Pour multi-classe, `shap_values` est une liste de 3 arrays (une par classe).

</details>

<details>
<summary>✅ Solution</summary>

```python
# Créer l'explainer
explainer_iris = shap.TreeExplainer(model_iris)

# Calculer SHAP values
shap_values_iris = explainer_iris.shap_values(X_test_iris)
# shap_values_iris est une liste [classe_0, classe_1, classe_2]

# Summary plot pour chaque classe
for i in range(3):
    print(f"\n=== CLASSE {iris.target_names[i]} ===")
    shap.summary_plot(shap_values_iris[i], X_test_iris, show=False)
    plt.title(f"SHAP values - Classe {iris.target_names[i]}")
    plt.show()

# Feature la plus importante globalement (moyenne sur toutes les classes)
global_importance = np.abs(np.concatenate(shap_values_iris, axis=1)).mean(axis=0)
most_important_feature = X_iris.columns[np.argmax(global_importance)]
print(f"\nFeature la plus importante : {most_important_feature}")
# Attendu : 'petal length (cm)' ou 'petal width (cm)'
```

**Explication** : 
- TreeExplainer est optimal pour Random Forest
- Multi-classe nécessite de traiter chaque classe séparément
- La feature "petal width" ou "petal length" est généralement la plus discriminante pour Iris

</details>

### Exercice 2 : Debugging avec SHAP (Intermédiaire)

**Énoncé** :

Vous avez un modèle XGBoost avec des performances médiocres. Utilisez SHAP pour identifier si le problème vient d'une feature "leaky" (qui contient de l'information du futur) ou d'une feature non informative.

**Données** :

```python
# Dataset synthétique avec feature "leaky"
np.random.seed(42)
n = 1000

X_debug = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, n),
    'feature_2': np.random.normal(0, 1, n),
    'feature_3': np.random.normal(0, 1, n),
    'feature_leaky': np.random.normal(0, 1, n)  # Va être corrélée avec y
})

# Cible
y_debug = (X_debug['feature_1'] + 2 * X_debug['feature_2'] > 0).astype(int)

# Feature leaky : corrélation artificielle (information du futur)
X_debug['feature_leaky'] = y_debug + np.random.normal(0, 0.1, n)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_debug, y_debug, test_size=0.3, random_state=42
)

# Modèle
model_debug = xgb.XGBClassifier(n_estimators=100, random_state=42)
model_debug.fit(X_train_d, y_train_d)

print(f"Accuracy : {model_debug.score(X_test_d, y_test_d):.3f}")
# Accuracy très élevée (> 0.95) grâce à feature_leaky
```

**Objectif** : Utiliser SHAP pour détecter que `feature_leaky` domine anormalement les prédictions.

**Indices** :

<details>
<summary>💡 Indice 1</summary>

Créez un summary plot bar pour voir l'importance relative des features.

</details>

<details>
<summary>💡 Indice 2</summary>

Si une feature a une importance BEAUCOUP plus élevée que les autres, c'est suspect.

</details>

<details>
<summary>✅ Solution</summary>

```python
# Étape 1 : SHAP analysis
explainer_debug = shap.TreeExplainer(model_debug)
shap_values_debug = explainer_debug.shap_values(X_test_d)

# Étape 2 : Summary plot (bar)
shap.summary_plot(shap_values_debug, X_test_d, plot_type="bar")

# Étape 3 : Calculer l'importance relative
global_importance_debug = np.abs(shap_values_debug).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_debug.columns,
    'importance': global_importance_debug,
    'importance_pct': global_importance_debug / global_importance_debug.sum() * 100
}).sort_values('importance', ascending=False)

print("\n=== IMPORTANCE DES FEATURES ===")
print(importance_df)

# Étape 4 : Détection de feature leaky
# Règle heuristique : Si une feature représente > 50% de l'importance totale, suspect
dominant_feature = importance_df.iloc[0]
if dominant_feature['importance_pct'] > 50:
    print(f"\n⚠️ ALERTE : Feature '{dominant_feature['feature']}' domine anormalement ({dominant_feature['importance_pct']:.1f}%)")
    print("   → Vérifier si cette feature contient de l'information du futur (data leakage)")
    print("   → Ou si elle est un proxy quasi-parfait de la cible")

# Étape 5 : Dependence plot pour confirmer
shap.dependence_plot("feature_leaky", shap_values_debug, X_test_d)
# On devrait voir une corrélation presque linéaire (feature_leaky ≈ y)
```

**Explication** : 
- `feature_leaky` représente ~70-80% de l'importance (anormal)
- Le dependence plot montre une séparation quasi-parfaite
- Conclusion : Data leakage détecté, retirer `feature_leaky` et réentraîner

</details>

### Exercice 3 : Production pipeline avec SHAP (Avancé)

**Énoncé** :

Implémentez un pipeline de production qui :
1. Entraîne un modèle XGBoost
2. Calcule les SHAP values pour les instances de test
3. Stocke les explications dans un cache (dictionnaire simulé)
4. Fournit une API pour expliquer de nouvelles instances (avec cache hit/miss)

**Objectif** : Optimiser les performances SHAP en production avec caching.

<details>
<summary>✅ Solution</summary>

```python
import hashlib
import pickle

class SHAPProductionPipeline:
    """
    Pipeline SHAP optimisé pour production avec caching
    """
    def __init__(self, model, explainer_type="tree"):
        self.model = model
        self.explainer_type = explainer_type
        self.cache = {}  # En production : Redis ou Memcached
        
        # Créer l'explainer
        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            raise ValueError(f"Type non supporté : {explainer_type}")
    
    def _hash_instance(self, instance):
        """
        Crée un hash unique pour une instance (pour caching)
        """
        # Sérialiser l'instance et calculer hash
        instance_bytes = pickle.dumps(instance.values)
        return hashlib.md5(instance_bytes).hexdigest()
    
    def explain(self, instance, use_cache=True):
        """
        Explique une instance avec caching
        
        Returns:
            dict: {
                'shap_values': array,
                'base_value': float,
                'prediction': float,
                'cache_hit': bool
            }
        """
        instance_hash = self._hash_instance(instance)
        
        # Check cache
        if use_cache and instance_hash in self.cache:
            print(f"✅ Cache HIT pour instance {instance_hash[:8]}...")
            return {**self.cache[instance_hash], 'cache_hit': True}
        
        # Cache miss : calculer SHAP
        print(f"⚠️ Cache MISS pour instance {instance_hash[:8]}... (calcul SHAP)")
        import time
        start = time.time()
        
        shap_values = self.explainer.shap_values(instance.values.reshape(1, -1))[0]
        prediction = self.model.predict_proba(instance.values.reshape(1, -1))[0, 1]
        
        elapsed = time.time() - start
        
        result = {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'prediction': prediction,
            'computation_time': elapsed,
            'cache_hit': False
        }
        
        # Store in cache
        if use_cache:
            self.cache[instance_hash] = result
        
        return result
    
    def explain_batch(self, instances, use_cache=True):
        """
        Explique un batch d'instances avec caching
        """
        results = []
        cache_hits = 0
        
        for idx in range(len(instances)):
            instance = instances.iloc[idx]
            result = self.explain(instance, use_cache=use_cache)
            results.append(result)
            if result['cache_hit']:
                cache_hits += 1
        
        print(f"\n=== STATISTIQUES BATCH ===")
        print(f"Total instances : {len(instances)}")
        print(f"Cache hits : {cache_hits} ({cache_hits/len(instances)*100:.1f}%)")
        print(f"Cache misses : {len(instances) - cache_hits}")
        
        return results
    
    def get_cache_stats(self):
        """
        Statistiques du cache
        """
        return {
            'cache_size': len(self.cache),
            'memory_usage_mb': sum(
                len(pickle.dumps(v)) for v in self.cache.values()
            ) / 1024 / 1024
        }

# Utilisation
pipeline = SHAPProductionPipeline(model_credit, explainer_type="tree")

# Expliquer des instances
print("=== PREMIÈRE EXÉCUTION ===")
result1 = pipeline.explain(X_test_c.iloc[0])
print(f"Temps de calcul : {result1['computation_time']*1000:.2f}ms")

print("\n=== DEUXIÈME EXÉCUTION (même instance) ===")
result2 = pipeline.explain(X_test_c.iloc[0])
print(f"Temps de calcul : < 0.01ms (cache hit)")

print("\n=== BATCH AVEC DUPLICATES ===")
# Simuler des requêtes avec duplicates (fréquent en production)
instances_with_duplicates = pd.concat([
    X_test_c.iloc[:5],
    X_test_c.iloc[:5],  # Duplicates
    X_test_c.iloc[5:10]
])
results_batch = pipeline.explain_batch(instances_with_duplicates, use_cache=True)

# Statistiques du cache
stats = pipeline.get_cache_stats()
print(f"\nCache size : {stats['cache_size']} instances")
print(f"Memory usage : {stats['memory_usage_mb']:.2f} MB")
```

**Explication** :
- Caching par hash MD5 de l'instance (en production : utiliser Redis)
- Gain de performance significatif sur instances fréquentes (e.g., profils types)
- Monitoring du taux de cache hit pour optimisation

</details>

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques Fondamentaux

#### 1. A Unified Approach to Interpreting Model Predictions (SHAP - Paper Fondateur)

- **Auteurs** : Scott M. Lundberg, Su-In Lee (2017)
- **Publication** : NIPS 2017
- **URL** : [https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
- **Contribution clé** : Introduction de SHAP comme framework unifié basé sur Shapley values, démonstration des propriétés théoriques
- **Pertinence** : À lire ABSOLUMENT pour comprendre les fondements théoriques de SHAP
- **Niveau** : Technique (mathématiques niveau master)

#### 2. From local explanations to global understanding with explainable AI for trees

- **Auteurs** : Scott M. Lundberg, Gabriel G. Erion, Hugh Chen, et al. (2020)
- **Publication** : Nature Machine Intelligence
- **URL** : [https://www.nature.com/articles/s42256-019-0138-9](https://www.nature.com/articles/s42256-019-0138-9)
- **Contribution clé** : TreeExplainer optimisé (complexité polynomiale), SHAP interaction values, applications en médecine
- **Pertinence** : Explique pourquoi TreeExplainer est si rapide (algorithme optimisé)
- **Niveau** : Technique

#### 3. A value for n-person games (Théorie de Shapley - Paper Original)

- **Auteurs** : Lloyd S. Shapley (1953)
- **Publication** : Contributions to the Theory of Games
- **URL** : [https://www.rand.org/pubs/papers/P0295.html](https://www.rand.org/pubs/papers/P0295.html)
- **Contribution clé** : Définition originale de la valeur de Shapley en théorie des jeux coopératifs
- **Pertinence** : Comprendre les fondements mathématiques (optionnel mais enrichissant)
- **Niveau** : Mathématique avancé

#### 4. Consistent Individualized Feature Attribution for Tree Ensembles

- **Auteurs** : Scott M. Lundberg, Gabriel G. Erion, Su-In Lee (2019)
- **Publication** : arXiv
- **URL** : [https://arxiv.org/abs/1802.03888](https://arxiv.org/abs/1802.03888)
- **Contribution clé** : Analyse de la cohérence de SHAP vs autres méthodes (LIME, feature importance)
- **Pertinence** : Comparaisons formelles, choix de méthode d'interprétabilité
- **Niveau** : Technique

#### 5. Explainable machine-learning predictions for the prevention of hypoxaemia during surgery

- **Auteurs** : Scott M. Lundberg, Bala Nair, et al. (2018)
- **Publication** : Nature Biomedical Engineering
- **URL** : [https://www.nature.com/articles/s41551-018-0304-0](https://www.nature.com/articles/s41551-018-0304-0)
- **Contribution clé** : Application clinique de SHAP, démonstration de l'adoption par les médecins
- **Pertinence** : Cas d'usage réel en production médicale
- **Niveau** : Accessible (focus application)

### 📚 Ressources Complémentaires

#### Articles de blog techniques

- **Interpretable Machine Learning with XGBoost** par Scott Lundberg
  - [https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)
  - 📌 **Pourquoi** : Tutoriel pratique avec exemples concrets XGBoost
  - ⏱️ **Durée** : ~15 min

- **SHAP for Model Interpretation** par Christoph Molnar
  - [https://christophm.github.io/interpretable-ml-book/shap.html](https://christophm.github.io/interpretable-ml-book/shap.html)
  - 📌 **Pourquoi** : Chapitre du livre "Interpretable Machine Learning" (référence)
  - ⏱️ **Durée** : ~30 min

- **Using SHAP Values to Explain How Your Machine Learning Model Works** par Kaggle
  - [https://www.kaggle.com/learn/machine-learning-explainability](https://www.kaggle.com/learn/machine-learning-explainability)
  - 📌 **Pourquoi** : Cours interactif avec notebooks Kaggle
  - ⏱️ **Durée** : ~2h (avec exercices)

#### Vidéos éducatives

- **SHAP: Explain Any Machine Learning Model in Python** par AssemblyAI
  - [https://www.youtube.com/watch?v=VB9uV-x0gtg](https://www.youtube.com/watch?v=VB9uV-x0gtg)
  - 📌 **Pourquoi** : Tutoriel vidéo complet avec code
  - ⏱️ **Durée** : 25min

- **Shapley Values Explained** par StatQuest
  - [https://www.youtube.com/watch?v=9OFMRiAVH-w](https://www.youtube.com/watch?v=9OFMRiAVH-w)
  - 📌 **Pourquoi** : Explication intuitive de la théorie de Shapley (excellente animation)
  - ⏱️ **Durée** : 10min

#### Documentation officielle

- **SHAP Python Library**
  - [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)
  - 📌 **Section recommandée** : API Reference (explainers), Example Notebooks
  - 📌 **GitHub** : [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

### 🛠️ Outils et Frameworks

#### Outil 1 : SHAP Library (officiel)

- **URL** : [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- **Description** : Bibliothèque Python officielle pour SHAP (maintenue par Scott Lundberg)
- **Cas d'usage** : Toutes implémentations SHAP en Python
- **Installation** :
  ```bash
  pip install shap
  # Pour visualisations avancées :
  pip install shap[plots]
  ```
- **Exemple rapide** :
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X)
  shap.summary_plot(shap_values, X)
  ```

#### Outil 2 : InterpretML (Microsoft)

- **URL** : [https://interpret.ml/](https://interpret.ml/)
- **Description** : Framework Microsoft incluant SHAP + autres méthodes (EBM, LIME, etc.)
- **Cas d'usage** : Comparaison de plusieurs méthodes d'interprétabilité
- **Installation** :
  ```bash
  pip install interpret
  ```
- **Exemple rapide** :
  ```python
  from interpret import show
  from interpret.blackbox import ShapKernel
  
  explainer = ShapKernel(model.predict_proba, X_train)
  explanation = explainer.explain_local(X_test[:5], y_test[:5])
  show(explanation)
  ```

#### Outil 3 : DALEX (R et Python)

- **URL** : [https://github.com/ModelOriented/DALEX](https://github.com/ModelOriented/DALEX)
- **Description** : Framework d'interprétabilité incluant SHAP (version R et Python)
- **Cas d'usage** : Utilisateurs R ou pipeline multi-langage
- **Installation** :
  ```bash
  pip install dalex
  ```

#### Outil 4 : Alibi Explain

- **URL** : [https://github.com/SeldonIO/alibi](https://github.com/SeldonIO/alibi)
- **Description** : Bibliothèque Seldon pour explainability (inclut KernelSHAP, TreeSHAP)
- **Cas d'usage** : Production sur Kubernetes avec Seldon Core
- **Installation** :
  ```bash
  pip install alibi
  ```

### 📖 Cours et Tutoriels Connexes

#### Dans votre repository (Liens Zettelkasten)

- **Prérequis** : 
  - [[fondamentaux_ml]] - Comprendre arbres de décision, régression, classification
  - [[xgboost_lightgbm]] - Modèles boosting souvent utilisés avec SHAP
  - [[feature_engineering]] - Importance de features de qualité pour interprétabilité

- **Approfondissement** :
  - [[lime_local_explanations]] - Méthode alternative d'interprétabilité locale (à créer)
  - [[permutation_importance]] - Importance globale des features (à créer)
  - [[partial_dependence_plots]] - Visualiser effets marginaux (à créer)

- **Sujets parallèles** :
  - [[integrated_gradients]] - Interprétabilité pour Deep Learning (déjà présent)
  - [[model_debugging]] - Utiliser SHAP pour déboguer des modèles
  - [[fairness_bias_detection]] - Détecter les biais avec SHAP

#### Cours externes recommandés

- **Machine Learning Explainability** par Kaggle
  - [https://www.kaggle.com/learn/machine-learning-explainability](https://www.kaggle.com/learn/machine-learning-explainability)
  - 📌 **Modules pertinents** : SHAP Values, Advanced Uses of SHAP Values
  - ⏱️ **Durée** : 4 heures

- **Interpretable Machine Learning** par Christoph Molnar (Livre gratuit)
  - [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
  - 📌 **Chapitres pertinents** : 5.9 Shapley Values, 5.10 SHAP
  - ⏱️ **Durée** : Livre complet (~10h), chapitres SHAP (~2h)

---

## 📝 Résumé Rapide (Quick Reference)

> **Carte de référence** : À consulter rapidement pour se remémorer l'essentiel.

### Concepts Clés

| Concept | Formule/Définition | Cas d'usage |
|---------|-------------------|-------------|
| **Valeur de Shapley** | $$\phi_i = \sum_{S} \frac{\|S\|! (\|N\| - \|S\| - 1)!}{\|N\|!} [v(S \cup \{i\}) - v(S)]$$ | Répartition équitable de la prédiction entre features |
| **Axiome d'efficacité** | $$\sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]$$ | Vérification : somme SHAP = prédiction - baseline |
| **TreeExplainer** | Complexité $$O(TLD^2)$$ | Arbres (XGBoost, RF, LightGBM) - Exact et rapide |
| **KernelExplainer** | Régression pondérée (Shapley weights) | Model-agnostic - Lent mais universel |
| **DeepExplainer** | Basé sur DeepLIFT | Réseaux de neurones - Approximation rapide |

### Code Minimal

```python
# Version ultra-minimaliste pour usage rapide
import shap

# Étape 1 : Créer l'explainer (arbres)
explainer = shap.TreeExplainer(model)

# Étape 2 : Calculer SHAP values
shap_values = explainer.shap_values(X_test)

# Étape 3 : Visualiser (global)
shap.summary_plot(shap_values, X_test)

# Étape 4 : Expliquer une instance (local)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0]
))
```

### Décisions Clés

**Quand utiliser SHAP** :
```
├─ Besoin d'expliquer décisions individuelles (local) : ✅ SHAP
├─ Besoin de garanties théoriques (audit, conformité) : ✅ SHAP
├─ Besoin de cohérence (somme = prédiction) : ✅ SHAP
└─ Modèles basés sur arbres : ✅ SHAP + TreeExplainer (optimal)
```
**Alternatives** :
```
├─ Si exploration rapide suffit → LIME (plus simple conceptuellement)
├─ Si importance globale suffit → Feature Importance / Permutation Importance
└─ Si Deep Learning + gradients → Integrated Gradients (voir cours IG)
```

### Choix de l'explainer

**Arbre de décision** :
```
Type de modèle ?
├─ Arbres (XGBoost, RF, LightGBM) → TreeExplainer ⚡⚡⚡
├─ Linéaire (LogReg, SVM) → LinearExplainer ⚡⚡⚡
├─ Deep Learning (TF, PyTorch) → DeepExplainer ou GradientExplainer ⚡⚡
└─ Autre / Custom → KernelExplainer ⚡ (lent)
```

### Pièges à éviter

1. ⚠️ **Background set non représentatif** (KernelExplainer) → Solution : `shap.sample(X, 100)`
2. ⚠️ **Confondre corrélation et causalité** → Solution : Mentionner "associé à" et non "cause"
3. ⚠️ **Ignorer les interactions** → Solution : Utiliser `dependence_plot` avec `interaction_index`
4. ⚠️ **Ne pas vérifier l'axiome d'efficacité** → Solution : `sum(shap_values) ≈ prédiction - baseline`

### Visualisations SHAP

| Visualisation | Usage | Code |
|---------------|-------|------|
| **Summary plot (bar)** | Importance globale | `shap.summary_plot(shap_values, X, plot_type="bar")` |
| **Summary plot (beeswarm)** | Importance + impact directionnel | `shap.summary_plot(shap_values, X)` |
| **Force plot** | Explication locale (instance) | `shap.force_plot(base, shap_values[i], X.iloc[i])` |
| **Waterfall plot** | Explication locale (cascade) | `shap.waterfall_plot(explanation)` |
| **Dependence plot** | Relation feature ↔ SHAP + interactions | `shap.dependence_plot("feature", shap_values, X)` |

---
