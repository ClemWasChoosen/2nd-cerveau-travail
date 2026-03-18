# 📏 Mesures de Dispersion : Variance, Écart-type, Quantiles

> **Résumé en une phrase** : Les mesures de dispersion quantifient la variabilité des données autour de leur centre, permettant de distinguer un dataset homogène d'un dataset très dispersé ayant pourtant la même moyenne.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-18 |
| **Dernière mise à jour** | 2026-03-18 |
| **Domaine** | Statistiques Descriptives |
| **Niveau** | Débutant-Intermédiaire |
| **Durée de lecture** | ~35 minutes |
| **Fichier** | `measures_dispersion.md` |
| **Emplacement** | `/00_statistics_foundations/01_descriptive_statistics/` |
| **Tags** | `#statistics` `#descriptive` `#variance` `#std` `#quantiles` `#iqr` `#dispersion` |

### Prérequis

- [x] [[measures_central_tendency]] - Moyenne, médiane, mode
- [ ] Mathématiques : carrés, racines carrées, sommes

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[measures_central_tendency]] - Mesures de tendance centrale
- **Complémentaires** : 
  - [[data_visualization_principles]] - Box plots, violin plots
  - [[distribution_analysis]] - Skewness et kurtosis
- **Suite recommandée** : 
  - [[gaussian_distribution]] - Loi normale (utilise variance)
  - [[hypothesis_testing]] - Tests statistiques basés sur variance
  - [[regression_diagnostics]] - Analyse de variance des résidus

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Savoir que vos données ont une moyenne de 50 ne suffit pas ! Sont-elles toutes proches de 50 (ex: 48, 49, 50, 51, 52) ou très dispersées (ex: 10, 30, 50, 70, 90) ? Les **mesures de dispersion** répondent à cette question fondamentale. Vous apprendrez à **quantifier la variabilité**, **détecter les outliers**, **comparer l'homogénéité** de différents groupes, et surtout **comprendre l'incertitude** autour de vos prédictions.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Définir variance, écart-type, quantiles, IQR et leur signification intuitive
2. **Appliquer** : Calculer ces mesures en Python et interpréter leurs valeurs
3. **Analyser** : Identifier quelle mesure utiliser selon la distribution et le contexte (outliers, asymétrie)
4. **Évaluer** : Comparer l'homogénéité de groupes, détecter anomalies, quantifier risque/incertitude
5. **Créer** : Construire des systèmes de détection d'outliers et features ML basées sur dispersion

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

Imaginez deux Data Scientists analysant des temps de réponse d'API :

**Data Scientist A** : *"Notre API répond en moyenne en 100ms !"*  
**Data Scientist B** : *"Notre API répond en moyenne en 100ms, avec un écart-type de 500ms !"*

**Qui a raison ?** Les deux ! Mais B donne une **information complète**. Un écart-type de 500ms signifie que certaines requêtes prennent 1 seconde (ou plus), créant une **expérience utilisateur désastreuse** malgré une "bonne" moyenne.

**Principe fondamental** : 
> **"Moyenne sans mesure de dispersion est une information incomplète et potentiellement trompeuse."**

En Data Science, vous rencontrerez constamment ces questions :
- *"Quelle est la **fiabilité** de ce modèle ?"* → Variance des prédictions
- *"Ces clients sont-ils **homogènes** ?"* → Écart-type comportemental
- *"Quel est le **risque** de cet investissement ?"* → Volatilité (écart-type financier)
- *"Y a-t-il des **valeurs aberrantes** ?"* → Détection via IQR ou écart-type

### Quel problème résout-il ?

**Problème** : Deux classes d'étudiants ont la même moyenne de 12/20. Le directeur veut savoir quelle classe est la "meilleure".

**Classe A** : Notes = [11, 11, 12, 12, 12, 13, 13]  
**Classe B** : Notes = [2, 5, 8, 12, 16, 19, 20]

```python
import numpy as np
import matplotlib.pyplot as plt

classe_A = np.array([11, 11, 12, 12, 12, 13, 13])
classe_B = np.array([2, 5, 8, 12, 16, 19, 20])

# Moyennes
print(f"Moyenne A : {np.mean(classe_A):.1f}/20")  # 12.0
print(f"Moyenne B : {np.mean(classe_B):.1f}/20")  # 11.7 (≈12)

# Écarts-types
print(f"Écart-type A : {np.std(classe_A, ddof=1):.2f}")  # 0.82
print(f"Écart-type B : {np.std(classe_B, ddof=1):.2f}")  # 6.73

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(range(len(classe_A)), classe_A, s=100, alpha=0.6, color='blue')
axes[0].axhline(np.mean(classe_A), color='red', linestyle='--', label=f'Moyenne={np.mean(classe_A):.1f}')
axes[0].set_ylim(0, 20)
axes[0].set_title(f'Classe A (σ={np.std(classe_A, ddof=1):.2f})\nHomogène')
axes[0].set_ylabel('Note /20')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(range(len(classe_B)), classe_B, s=100, alpha=0.6, color='green')
axes[1].axhline(np.mean(classe_B), color='red', linestyle='--', label=f'Moyenne={np.mean(classe_B):.1f}')
axes[1].set_ylim(0, 20)
axes[1].set_title(f'Classe B (σ={np.std(classe_B, ddof=1):.2f})\nTrès Hétérogène')
axes[1].set_ylabel('Note /20')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Interprétation** :
- **Classe A** : Écart-type faible (0.82) → Niveau **homogène**, enseignement efficace
- **Classe B** : Écart-type élevé (6.73) → **Hétérogénéité** forte, certains excellents, d'autres en échec

**Sans mesure de dispersion**, impossible de distinguer ces deux situations radicalement différentes !

### Applications dans le monde réel

1. **Finance / Trading** :
   - **Volatilité** = écart-type des rendements (mesure du risque)
   - **VaR (Value at Risk)** = quantile pour gestion du risque
   - **Sharpe Ratio** = rendement moyen / écart-type (ratio risque/récompense)

2. **Contrôle Qualité Industriel** :
   - Six Sigma = processus avec écart-type très faible (3.4 défauts / million)
   - Cartes de contrôle basées sur limites ±3σ
   - Coefficient de variation pour comparer variabilité entre produits

3. **Machine Learning** :
   - **Feature scaling** : normalisation basée sur écart-type (standardisation)
   - **Détection d'anomalies** : points > 3 écarts-types de la moyenne
   - **Variance expliquée** en PCA
   - **Confidence intervals** : prédiction ± 2×écart-type

4. **A/B Testing / Expérimentation** :
   - Taille d'échantillon dépend de la variance attendue
   - Puissance statistique = fonction de l'écart-type
   - Intervalles de confiance : moyenne ± $$t \times \frac{s}{\sqrt{n}}$$

5. **Santé / Médecine** :
   - **Plages de normalité** : moyenne ± 2σ (ex: tension artérielle)
   - **Croissance enfants** : percentiles (P3, P50, P97)
   - **Variabilité biologique** entre individus

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous construisons progressivement, en commençant par le concept le plus simple (range) vers le plus sophistiqué (variance décomposée).

### 1. Le Range (Étendue)

#### 1.1 Définition

**Range** : Différence entre valeur maximale et minimale.

$$\text{Range} = \max(x) - \min(x)$$

**Intuition** :
La "portée" des données, du point le plus bas au plus haut.

**Pourquoi cette définition ?**
C'est la mesure de dispersion la plus simple et intuitive : "Combien d'espace occupent mes données ?"

```python
import numpy as np

data = np.array([10, 15, 18, 22, 25, 30])

range_value = np.ptp(data)  # Peak-To-Peak
# ou manuellement
range_manual = np.max(data) - np.min(data)

print(f"Range : {range_value}")  # 20
```

#### 1.2 Avantages et Limitations

**✅ Avantages** :
- Extrêmement simple à calculer et comprendre
- Donne bornes min/max (utile pour scaling)

**❌ Limitations MAJEURES** :
- **Très sensible aux outliers** (1 seule valeur extrême change tout)
- **Ne donne aucune info** sur la distribution entre min et max
- **Instable** : augmente avec la taille de l'échantillon

```python
# Démonstration sensibilité
data_normal = np.array([10, 12, 13, 14, 15, 16, 18])
data_avec_outlier = np.array([10, 12, 13, 14, 15, 16, 100])

print(f"Range sans outlier : {np.ptp(data_normal)}")      # 8
print(f"Range avec outlier : {np.ptp(data_avec_outlier)}")  # 90 (explosé !)
```

**Quand utiliser** :
- Exploration rapide initiale
- Définir bornes pour scaling/normalisation
- **JAMAIS comme mesure principale de dispersion** (trop instable)

---

### 2. Variance et Écart-type

#### 2.1 Définition de la Variance

**Variance** : Moyenne des carrés des écarts à la moyenne.

**Variance de la population** ($$\sigma^2$$) :

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

**Variance de l'échantillon** ($$s^2$$) :

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Où** :
- $$\sigma^2$$ : Variance de la population (paramètre théorique)
- $$s^2$$ : Variance de l'échantillon (estimateur)
- $$\mu$$ : Moyenne de la population
- $$\bar{x}$$ : Moyenne de l'échantillon
- $$N$$ : Taille population
- $$n$$ : Taille échantillon
- $$n-1$$ : **Correction de Bessel** (degré de liberté)

**Intuition** :
La variance mesure "à quelle distance en moyenne les points se trouvent de la moyenne". Plus la variance est élevée, plus les données sont dispersées.

**Pourquoi élever au carré ?**

1. **Annuler les signes** : $$\sum (x_i - \bar{x}) = 0$$ toujours (propriété moyenne), donc on ne peut pas juste sommer les écarts
2. **Pénaliser davantage les grands écarts** : Un point à 10 de la moyenne contribue 100 à la variance (vs 10 si simple valeur absolue)
3. **Propriétés mathématiques** : Le carré rend la variance **dérivable**, essentiel pour optimisation

#### 2.2 Pourquoi $$n-1$$ au lieu de $$n$$ ? (Correction de Bessel)

**Question mathématique fondamentale** : Pourquoi diviser par $$n-1$$ pour l'échantillon et $$N$$ pour la population ?

**Réponse** : Biais statistique !

Si on utilisait $$\frac{1}{n} \sum (x_i - \bar{x})^2$$, on sous-estimerait **systématiquement** la vraie variance de la population. Pourquoi ?

- On utilise $$\bar{x}$$ (moyenne échantillon) au lieu de $$\mu$$ (moyenne population inconnue)
- $$\bar{x}$$ est calculé **à partir des mêmes données**, donc "trop proche" d'elles
- Résultat : écarts à $$\bar{x}$$ sont plus petits qu'écarts à $$\mu$$

Diviser par $$n-1$$ corrige ce biais et rend l'estimateur **sans biais** :

$$\mathbb{E}[s^2] = \sigma^2$$

```python
# Démonstration empirique du biais
np.random.seed(42)
vraie_variance = 100  # Variance population

# Simulation : 10,000 échantillons de taille 10
n_simulations = 10000
n_sample = 10

variances_n = []      # Divisé par n (BIAISÉ)
variances_n_minus_1 = []  # Divisé par n-1 (NON BIAISÉ)

for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(vraie_variance), n_sample)
    
    # Variance biaisée (n)
    var_n = np.sum((sample - np.mean(sample))**2) / n_sample
    variances_n.append(var_n)
    
    # Variance non biaisée (n-1)
    var_n_minus_1 = np.sum((sample - np.mean(sample))**2) / (n_sample - 1)
    variances_n_minus_1.append(var_n_minus_1)

print(f"Vraie variance population : {vraie_variance}")
print(f"Moyenne variance (divisé par n)   : {np.mean(variances_n):.2f}  ❌ SOUS-ESTIMÉ")
print(f"Moyenne variance (divisé par n-1) : {np.mean(variances_n_minus_1):.2f}  ✅ NON BIAISÉ")
```

**Résultat typique** :
```
Vraie variance population : 100
Moyenne variance (divisé par n)   : 90.12  ❌ SOUS-ESTIMÉ
Moyenne variance (divisé par n-1) : 100.13  ✅ NON BIAISÉ
```

**Règle pratique** :
- Variance **population** (toutes les données) : diviser par $$N$$
- Variance **échantillon** (estimation) : diviser par $$n-1$$ (paramètre `ddof=1` en Python)

#### 2.3 Écart-type (Standard Deviation)

**Écart-type** : Racine carrée de la variance.

$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

**Pourquoi prendre la racine carrée ?**

La variance est en **unités au carré** (ex: si données en mètres, variance en m²). L'écart-type revient aux **unités originales** → beaucoup plus **interprétable** !

**Exemple** :
- Salaires en € : variance en €² (abstrait), écart-type en € (concret)
- Temps en secondes : variance en s², écart-type en s

**Calcul Python** :

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50])

# Variance et écart-type POPULATION (ddof=0)
variance_pop = np.var(data, ddof=0)
std_pop = np.std(data, ddof=0)

# Variance et écart-type ÉCHANTILLON (ddof=1) ← DÉFAUT RECOMMANDÉ
variance_sample = np.var(data, ddof=1)
std_sample = np.std(data, ddof=1)

print(f"Population - Variance : {variance_pop:.2f}, Écart-type : {std_pop:.2f}")
print(f"Échantillon - Variance : {variance_sample:.2f}, Écart-type : {std_sample:.2f}")

# Vérification : σ = √(variance)
print(f"\nVérification : √{variance_sample:.2f} = {np.sqrt(variance_sample):.2f}")
```

#### 2.4 Propriétés de la Variance

**Propriété 1 : Non-négativité**

$$\sigma^2 \geq 0$$

Variance nulle ⟺ toutes les valeurs identiques (aucune dispersion).

**Propriété 2 : Linéarité partielle**

Si $$Y = aX + b$$, alors :

$$\text{Var}(Y) = a^2 \cdot \text{Var}(X)$$

$$\sigma_Y = |a| \cdot \sigma_X$$

**Attention** : La constante $$b$$ disparaît (translation ne change pas dispersion), mais multiplicateur $$a$$ est au **carré** pour variance !

```python
# Démonstration
X = np.array([10, 20, 30, 40, 50])
a, b = 3, 100

Y = a * X + b  # Y = 3X + 100

var_X = np.var(X, ddof=1)
var_Y = np.var(Y, ddof=1)

print(f"Var(X) = {var_X:.2f}")
print(f"Var(Y) = Var(3X + 100) = {var_Y:.2f}")
print(f"3² × Var(X) = {(a**2) * var_X:.2f}")  # Égal à Var(Y)

std_X = np.std(X, ddof=1)
std_Y = np.std(Y, ddof=1)

print(f"\nStd(X) = {std_X:.2f}")
print(f"Std(Y) = {std_Y:.2f}")
print(f"|3| × Std(X) = {abs(a) * std_X:.2f}")  # Égal à Std(Y)
```

**Application pratique** : Standardisation (Z-score)

$$Z = \frac{X - \mu}{\sigma}$$

Propriété : $$\text{Var}(Z) = 1$$, $$\mathbb{E}[Z] = 0$$

**Propriété 3 : Variance d'une somme (variables indépendantes)**

Si $$X$$ et $$Y$$ sont **indépendantes** :

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**Attention** : Ne fonctionne que si indépendantes ! Sinon :

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

(Voir [[correlation_covariance]] pour détails)

**Propriété 4 : Décomposition variance** (formule computationnelle)

$$\sigma^2 = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

Utile pour calcul efficace :

```python
# Méthode classique (2 passes sur données)
def variance_classic(data):
    mean = np.mean(data)
    return np.mean((data - mean)**2)

# Méthode computationnelle (1 passe) - ATTENTION instabilité numérique
def variance_computational(data):
    return np.mean(data**2) - (np.mean(data))**2

data = np.array([10, 20, 30, 40, 50])
print(f"Variance classique : {variance_classic(data):.2f}")
print(f"Variance computationnelle : {variance_computational(data):.2f}")
# ⚠️ Pour grandes valeurs, méthode computationnelle peut avoir erreurs numériques
```

**Sources académiques** :
- Bessel, F. W. (1838). Correction de biais pour variance échantillon
- [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm) - Variance

---

### 3. Quantiles et Percentiles

#### 3.1 Définitions

**Quantile** : Valeur qui divise les données ordonnées en proportions spécifiques.

**Quantile d'ordre $$p$$** ($$0 \leq p \leq 1$$) : Valeur $$q_p$$ telle que :
- $$100p\%$$ des données ≤ $$q_p$$
- $$100(1-p)\%$$ des données ≥ $$q_p$$

**Percentile** : Quantile exprimé en pourcentage ($$p \times 100$$).

**Nomenclature courante** :

| Nom | Notation | Percentile | Signification |
|-----|----------|-----------|---------------|
| **Minimum** | $$Q_0$$ | P0 | 0% des données ≤ min |
| **Premier quartile** | $$Q_1$$ | P25 | 25% des données ≤ Q1 |
| **Médiane** | $$Q_2$$ | P50 | 50% des données ≤ médiane |
| **Troisième quartile** | $$Q_3$$ | P75 | 75% des données ≤ Q3 |
| **Maximum** | $$Q_4$$ | P100 | 100% des données ≤ max |

**Visualisation** :

```
Données triées : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                  ↑           ↑           ↑            ↑
                 Min         Q1          Q2           Q3        Max
                 (0%)       (25%)       (50%)        (75%)     (100%)
```

#### 3.2 Calcul des Quantiles

**Méthode** : Il existe plusieurs définitions (9 méthodes différentes !) pour gérer cas où quantile tombe "entre" deux valeurs.

NumPy utilise **interpolation linéaire** par défaut :

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Quantiles individuels
q1 = np.quantile(data, 0.25)   # 25ème percentile
q2 = np.quantile(data, 0.50)   # Médiane
q3 = np.quantile(data, 0.75)   # 75ème percentile

print(f"Q1 (25%) : {q1}")
print(f"Q2 (50%) : {q2}")
print(f"Q3 (75%) : {q3}")

# Multiples quantiles en une fois
quantiles = np.quantile(data, [0.25, 0.50, 0.75])
print(f"\nQuartiles : {quantiles}")

# Percentiles (syntaxe alternative)
p25 = np.percentile(data, 25)
p50 = np.percentile(data, 50)
p75 = np.percentile(data, 75)
p95 = np.percentile(data, 95)

print(f"\nP25={p25}, P50={p50}, P75={p75}, P95={p95}")
```

#### 3.3 Intervalle Interquartile (IQR)

**IQR (Interquartile Range)** : Différence entre Q3 et Q1.

$$\text{IQR} = Q_3 - Q_1$$

**Intuition** :
IQR mesure la dispersion des **50% centraux** des données (entre 25ème et 75ème percentile).

**Pourquoi c'est important ?**
- **Robuste aux outliers** (contrairement au range ou écart-type)
- Base de la **règle de détection d'outliers** (box plot)

```python
import numpy as np
from scipy import stats

data = np.array([10, 15, 18, 22, 25, 28, 30, 35, 40, 100])  # 100 = outlier

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = stats.iqr(data)  # ou simplement q3 - q1

print(f"Q1 = {q1}")
print(f"Q3 = {q3}")
print(f"IQR = {iqr}")

# Comparaison avec range
range_value = np.ptp(data)
print(f"\nRange = {range_value} (affecté par outlier 100)")
print(f"IQR = {iqr:.1f} (robuste, ignore outlier)")
```

#### 3.4 Règle de Détection d'Outliers (Méthode Box Plot)

**Règle de Tukey** (standard dans les box plots) :

- **Outlier mild** (modéré) : Valeur < $$Q_1 - 1.5 \times \text{IQR}$$ ou > $$Q_3 + 1.5 \times \text{IQR}$$
- **Outlier extreme** : Valeur < $$Q_1 - 3 \times \text{IQR}$$ ou > $$Q_3 + 3 \times \text{IQR}$$

**Implémentation** :

```python
def detect_outliers_iqr(data, method='tukey'):
    """
    Détecte outliers par méthode IQR.
    
    Args:
        data (array): Données numériques
        method (str): 'tukey' (1.5×IQR) ou 'extreme' (3×IQR)
    
    Returns:
        dict: Indices outliers, bornes, statistiques
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    if method == 'tukey':
        multiplier = 1.5
    elif method == 'extreme':
        multiplier = 3.0
    else:
        raise ValueError("method doit être 'tukey' ou 'extreme'")
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    outliers_indices = np.where(outliers_mask)[0]
    outliers_values = data[outliers_mask]
    
    return {
        'n_outliers': len(outliers_indices),
        'outliers_indices': outliers_indices,
        'outliers_values': outliers_values,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }

# Test
data = np.array([10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 100, 105])
result = detect_outliers_iqr(data, method='tukey')

print(f"Données : {data}")
print(f"\nQ1 = {result['q1']}, Q3 = {result['q3']}, IQR = {result['iqr']}")
print(f"Bornes acceptables : [{result['lower_bound']:.1f}, {result['upper_bound']:.1f}]")
print(f"Outliers détectés : {result['n_outliers']}")
print(f"Valeurs outliers : {result['outliers_values']}")
```

**Comparaison IQR vs Écart-type pour détection outliers** :

| Critère | IQR (Tukey) | Écart-type (±3σ) |
|---------|-------------|------------------|
| **Robustesse** | ✅ Robuste (basé sur quartiles) | ❌ Sensible (outliers influencent σ) |
| **Distribution** | ✅ Fonctionne sur toute distribution | ⚠️ Optimal pour distribution normale |
| **Seuil** | Q1-1.5×IQR, Q3+1.5×IQR | μ-3σ, μ+3σ |
| **Taux faux positifs** | ~0.7% (si normale) | ~0.3% (si normale) |
| **Usage** | Box plots, données asymétriques | Processus industriels, contrôle qualité |

**Sources académiques** :
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley - Règle IQR originale

---

### 4. Coefficient de Variation (CV)

#### 4.1 Définition

**Coefficient de Variation** : Ratio écart-type / moyenne, exprimé en pourcentage.

$$CV = \frac{\sigma}{\mu} \times 100\%$$

ou pour échantillon :

$$CV = \frac{s}{\bar{x}} \times 100\%$$

**Intuition** :
Le CV mesure la dispersion **relative** à la moyenne. C'est une **mesure sans unité** qui permet de comparer la variabilité de datasets ayant des échelles différentes.

**Pourquoi cette définition ?**

Comparer l'écart-type brut de deux datasets d'échelles différentes n'a pas de sens :
- Dataset A : Poids en grammes, σ = 50g
- Dataset B : Poids en kg, σ = 0.05kg

Ils ont la **même variabilité**, mais σ numériquement différent ! Le CV résout ce problème.

#### 4.2 Interprétation

**Règle empirique** :

- **CV < 15%** : Faible variabilité (données homogènes)
- **15% ≤ CV < 30%** : Variabilité modérée
- **CV ≥ 30%** : Haute variabilité (données hétérogènes)

**Attention** : Ces seuils dépendent du contexte (processus industriel vs phénomène naturel).

#### 4.3 Exemple Pratique

```python
import numpy as np

# Deux processus de production
processus_A = np.array([99.8, 100.1, 99.9, 100.0, 100.2, 99.7, 100.3])  # Précision haute
processus_B = np.array([95, 105, 98, 102, 97, 103, 100])  # Moins précis

mean_A, std_A = np.mean(processus_A), np.std(processus_A, ddof=1)
mean_B, std_B = np.mean(processus_B), np.std(processus_B, ddof=1)

cv_A = (std_A / mean_A) * 100
cv_B = (std_B / mean_B) * 100

print(f"Processus A :")
print(f"  Moyenne = {mean_A:.2f}, Écart-type = {std_A:.3f}")
print(f"  CV = {cv_A:.2f}%  → Très stable")

print(f"\nProcessus B :")
print(f"  Moyenne = {mean_B:.2f}, Écart-type = {std_B:.2f}")
print(f"  CV = {cv_B:.2f}%  → Moins stable")

print(f"\n💡 Bien que moyennes similaires (~100), processus A est {cv_B/cv_A:.1f}× plus stable")
```

#### 4.4 Quand NE PAS Utiliser le CV

**❌ Moyenne proche de zéro** : CV explose ou n'a pas de sens

```python
# Exemple problématique
temperatures_celsius = np.array([-2, -1, 0, 1, 2])
mean_temp = np.mean(temperatures_celsius)  # 0°C
std_temp = np.std(temperatures_celsius, ddof=1)  # 1.58°C

# CV = 1.58 / 0 = inf ou indéfini !
# Solution : utiliser Kelvin ou ne pas utiliser CV pour données avec zéro naturel
```

**❌ Données avec échelle arbitraire** : Température Celsius/Fahrenheit, dates

**✅ Quand utiliser** :
- Données à échelle de ratio (zéro absolu) : poids, longueur, prix, concentrations
- Comparaison variabilité entre datasets différentes échelles

**Sources académiques** :
- [Coefficient of Variation - NIST](https://www.itl.nist.gov/div898/handbook/) - Applications en contrôle qualité

---

### 5. MAD (Median Absolute Deviation)

#### 5.1 Définition

**MAD** : Médiane des écarts absolus à la médiane.

$$\text{MAD} = \text{median}(|X_i - \text{median}(X)|)$$

**Intuition** :
MAD est l'équivalent **robuste** de l'écart-type. Au lieu de mesurer écarts à la moyenne (sensible aux outliers), on mesure écarts à la médiane (robuste).

**Pourquoi cette définition ?**

L'écart-type peut être fortement influencé par un seul outlier. MAD reste stable car :
1. Basé sur la **médiane** (robuste)
2. Utilise **valeur absolue** (pas carré qui amplifie outliers)
3. Prend **médiane** des écarts (pas moyenne)

#### 5.2 Calcul et Normalisation

```python
from scipy import stats
import numpy as np

def mad(data, scale='normal'):
    """
    Calcule Median Absolute Deviation.
    
    Args:
        data (array): Données
        scale (str): 'normal' pour estimateur cohérent avec std si distribution normale
    
    Returns:
        float: MAD (éventuellement normalisé)
    """
    median = np.median(data)
    mad_value = np.median(np.abs(data - median))
    
    if scale == 'normal':
        # Facteur pour être comparable à écart-type si distribution normale
        # Théoriquement : σ ≈ 1.4826 × MAD
        mad_value *= 1.4826
    
    return mad_value

# Exemple
data_normal = np.array([10, 12, 13, 14, 15, 16, 17, 18, 20])
data_avec_outliers = np.array([10, 12, 13, 14, 15, 16, 17, 18, 200])

std_normal = np.std(data_normal, ddof=1)
std_outliers = np.std(data_avec_outliers, ddof=1)

mad_normal = mad(data_normal, scale='normal')
mad_outliers = mad(data_avec_outliers, scale='normal')

print(f"SANS OUTLIERS :")
print(f"  Écart-type : {std_normal:.2f}")
print(f"  MAD normalisé : {mad_normal:.2f}  (similaire)")

print(f"\nAVEC OUTLIERS (200) :")
print(f"  Écart-type : {std_outliers:.2f}  ❌ EXPLOSE")
print(f"  MAD normalisé : {mad_outliers:.2f}  ✅ ROBUSTE")
```

**SciPy a fonction intégrée** :

```python
from scipy.stats import median_abs_deviation

mad_scipy = median_abs_deviation(data_avec_outliers, scale='normal')
print(f"MAD (SciPy) : {mad_scipy:.2f}")
```

#### 5.3 Quand Utiliser MAD vs Écart-type

| Critère | Écart-type ($$\sigma$$) | MAD |
|---------|------------------------|-----|
| **Robustesse** | ❌ Sensible outliers | ✅ Très robuste |
| **Distribution** | Optimal si normale | Fonctionne toute distribution |
| **Efficacité (normale)** | ✅ Optimal | ⚠️ Moins efficace (~64% efficient) |
| **Interprétation** | ✅ Standard, bien connu | ⚠️ Moins familier |
| **Calcul** | $$O(n)$$ | $$O(n \log n)$$ (tri pour médiane) |
| **Usage ML** | Standardisation classique | Détection outliers, données sales |

**Recommandation** :
- **Données propres, normale** → Écart-type
- **Données avec outliers, distribution inconnue** → MAD
- **Détection d'anomalies** → MAD (règle : > médiane + 3×MAD)

---

## 💡 Compréhension Intuitive

### Analogie du monde réel

Imaginez deux archers visant une cible :

**Archer A (faible dispersion)** :
Flèches : toutes dans le cercle central (10 points)
- Moyenne : 10 points
- Écart-type : 0.5 points
- **Interprétation** : Très **précis** et **consistant**

**Archer B (forte dispersion)** :
Flèches : partout sur la cible (de 3 à 10 points)
- Moyenne : 10 points (par chance)
- Écart-type : 3.5 points
- **Interprétation** : Même moyenne, mais **inconsistant** et **peu fiable**

**Leçon** : En compétition, Archer A gagnera car **faible variabilité = prévisibilité = fiabilité**.

En Data Science : Un modèle ML avec **faible variance des prédictions** sur validation set est plus fiable qu'un modèle avec même moyenne mais haute variance !

### Questions pour vérifier la compréhension

Avant de continuer, assurez-vous de pouvoir répondre :

1. **Q1** : Pourquoi divise-t-on par $$n-1$$ pour variance échantillon ?
   - *Réponse* : Corriger biais (estimateur sans biais de variance population)

2. **Q2** : Si toutes les données sont multipliées par 2, comment change l'écart-type ?
   - *Réponse* : Multiplié par 2 également ($$\sigma_{2X} = 2\sigma_X$$)

3. **Q3** : IQR ou écart-type pour détecter outliers dans données très asymétriques ?
   - *Réponse* : IQR (robuste aux outliers et asymétrie)

4. **Q4** : Deux datasets ont σ=10. Peut-on dire qu'ils ont même variabilité ?
   - *Réponse* : Non ! Dépend de la moyenne. Comparer via coefficient de variation.

---

## 💻 Implémentation Pratique

> **Principe de modalité** : Code commenté + explication textuelle pour double encodage cognitif.

### 1. Fonction Complète de Statistiques Descriptives

```python
"""
Titre : Fonction complète de statistiques descriptives
Objectif : Calculer toutes mesures de tendance centrale ET dispersion
Complexité : O(n log n) (à cause du tri pour médiane/quantiles)
"""

import numpy as np
from scipy import stats

def descriptive_statistics(data, name="Data", show_plot=False):
    """
    Calcule statistiques descriptives complètes.
    
    Args:
        data (array-like): Données numériques
        name (str): Nom du dataset
        show_plot (bool): Afficher visualisations
    
    Returns:
        dict: Toutes les statistiques
    """
    import numpy as np
    from scipy import stats
    
    # Conversion et nettoyage
    data = np.asarray(data, dtype=float)
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        raise ValueError("Aucune donnée valide !")
    
    n = len(data_clean)
    n_missing = len(data) - n
    
    # ========== TENDANCE CENTRALE ==========
    mean = np.mean(data_clean)
    median = np.median(data_clean)
    try:
        mode_result = stats.mode(data_clean, keepdims=True)
        mode = mode_result.mode[0]
    except:
        mode = np.nan
    
    # ========== DISPERSION ==========
    # Variance et écart-type (échantillon)
    variance = np.var(data_clean, ddof=1)
    std = np.std(data_clean, ddof=1)
    
    # Range
    data_range = np.ptp(data_clean)
    data_min = np.min(data_clean)
    data_max = np.max(data_clean)
    
    # Quantiles
    q1 = np.percentile(data_clean, 25)
    q3 = np.percentile(data_clean, 75)
    iqr = q3 - q1
    
    # Percentiles additionnels
    p5 = np.percentile(data_clean, 5)
    p95 = np.percentile(data_clean, 95)
    p99 = np.percentile(data_clean, 99)
    
    # Coefficient de variation
    cv = (std / mean * 100) if mean != 0 else np.nan
    
    # MAD
    mad_value = stats.median_abs_deviation(data_clean, scale='normal')
    
    # ========== FORME DISTRIBUTION ==========
    skewness = stats.skew(data_clean)
    kurtosis_value = stats.kurtosis(data_clean)
    
    # ========== DÉTECTION OUTLIERS ==========
    # Méthode IQR
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_iqr = data_clean[(data_clean < lower_fence) | (data_clean > upper_fence)]
    
    # Méthode Z-score
    z_scores = np.abs(stats.zscore(data_clean))
    outliers_zscore = data_clean[z_scores > 3]
    
    # ========== RÉSULTATS ==========
    results = {
        # Général
        'n': n,
        'n_missing': n_missing,
        
        # Tendance centrale
        'mean': mean,
        'median': median,
        'mode': mode,
        
        # Dispersion
        'variance': variance,
        'std': std,
        'range': data_range,
        'min': data_min,
        'max': data_max,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'p5': p5,
        'p95': p95,
        'p99': p99,
        'cv': cv,
        'mad': mad_value,
        
        # Forme
        'skewness': skewness,
        'kurtosis': kurtosis_value,
        
        # Outliers
        'n_outliers_iqr': len(outliers_iqr),
        'n_outliers_zscore': len(outliers_zscore),
        'outliers_iqr': outliers_iqr,
        'outliers_zscore': outliers_zscore,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence
    }
    
    # ========== AFFICHAGE ==========
    print(f"\n{'='*70}")
    print(f"STATISTIQUES DESCRIPTIVES - {name}")
    print(f"{'='*70}")
    
    print(f"\n📊 GÉNÉRAL")
    print(f"  Observations       : {n:,} (dont {n_missing} manquantes)")
    
    print(f"\n📍 TENDANCE CENTRALE")
    print(f"  Moyenne            : {mean:.4f}")
    print(f"  Médiane            : {median:.4f}")
    print(f"  Mode               : {mode:.4f}" if not np.isnan(mode) else "  Mode               : Non unique")
    
    print(f"\n📏 DISPERSION")
    print(f"  Variance           : {variance:.4f}")
    print(f"  Écart-type         : {std:.4f}")
    print(f"  CV                 : {cv:.2f}%" if not np.isnan(cv) else "  CV                 : N/A")
    print(f"  MAD (normalisé)    : {mad_value:.4f}")
    print(f"  Range              : {data_range:.4f} [{data_min:.2f}, {data_max:.2f}]")
    print(f"  IQR                : {iqr:.4f} [Q1={q1:.2f}, Q3={q3:.2f}]")
    
    print(f"\n📈 QUANTILES")
    print(f"  P5                 : {p5:.4f}")
    print(f"  Q1 (P25)           : {q1:.4f}")
    print(f"  Médiane (P50)      : {median:.4f}")
    print(f"  Q3 (P75)           : {q3:.4f}")
    print(f"  P95                : {p95:.4f}")
    print(f"  P99                : {p99:.4f}")
    
    print(f"\n🔍 FORME DISTRIBUTION")
    skew_interp = "symétrique" if abs(skewness) < 0.5 else ("asymétrique droite" if skewness > 0 else "asymétrique gauche")
    print(f"  Skewness           : {skewness:.4f} ({skew_interp})")
    kurt_interp = "normale" if abs(kurtosis_value) < 0.5 else ("queues lourdes" if kurtosis_value > 0 else "queues légères")
    print(f"  Kurtosis           : {kurtosis_value:.4f} ({kurt_interp})")
    
    print(f"\n⚠️  OUTLIERS")
    print(f"  Méthode IQR (1.5×)  : {results['n_outliers_iqr']} outliers détectés")
    print(f"  Méthode Z-score (>3σ): {results['n_outliers_zscore']} outliers détectés")
    
    # ========== VISUALISATION ==========
    if show_plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogramme + KDE
        axes[0, 0].hist(data_clean, bins=30, density=True, alpha=0.7, edgecolor='black', label='Données')
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data_clean)
            x_range = np.linspace(data_min, data_max, 100)
            axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        axes[0, 0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Moyenne={mean:.2f}')
        axes[0, 0].axvline(median, color='blue', linestyle='--', linewidth=2, label=f'Médiane={median:.2f}')
        axes[0, 0].set_xlabel('Valeur')
        axes[0, 0].set_ylabel('Densité')
        axes[0, 0].set_title(f'{name} - Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Box plot
        bp = axes[0, 1].boxplot(data_clean, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_ylabel('Valeur')
        axes[0, 1].set_title(f'{name} - Box Plot')
        axes[0, 1].grid(alpha=0.3)
        
        # QQ plot (normalité)
        stats.probplot(data_clean, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{name} - QQ Plot (Test Normalité)')
        axes[1, 0].grid(alpha=0.3)
        
        # Résumé textuel
        axes[1, 1].axis('off')
        summary_text = f"""
        📊 RÉSUMÉ STATISTIQUE
        
        n = {n:,}
        
        Tendance Centrale:
        • Moyenne = {mean:.2f}
        • Médiane = {median:.2f}
        • Écart = {abs(mean-median):.2f}
        
        Dispersion:
        • Écart-type = {std:.2f}
        • CV = {cv:.1f}%
        • IQR = {iqr:.2f}
        
        Distribution:
        • Skewness = {skewness:.2f}
        • Kurtosis = {kurtosis_value:.2f}
        
        Outliers:
        • IQR method: {results['n_outliers_iqr']}
        • Z-score: {results['n_outliers_zscore']}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    return results

# ========== TEST ==========
np.random.seed(42)
# Données log-normales (asymétriques avec outliers)
data_test = np.concatenate([
    np.random.lognormal(mean=3, sigma=0.5, size=200),
    np.array([200, 250, 300])  # Outliers
])

stats_result = descriptive_statistics(data_test, name="Temps Réponse API (ms)", show_plot=True)
```

### 2. Comparaison Robustesse des Mesures

```python
"""
Titre : Comparaison robustesse : Écart-type vs MAD vs IQR
Objectif : Démontrer impact des outliers sur différentes mesures
"""

import numpy as np
import matplotlib.pyplot as plt

def compare_dispersion_measures(data, outlier_values, name="Dataset"):
    """
    Compare comment différentes mesures réagissent aux outliers.
    """
    measures_original = {}
    measures_with_outliers = {}
    
    # Calculs sur données originales
    measures_original['Écart-type'] = np.std(data, ddof=1)
    measures_original['IQR'] = np.percentile(data, 75) - np.percentile(data, 25)
    from scipy.stats import median_abs_deviation
    measures_original['MAD'] = median_abs_deviation(data, scale='normal')
    measures_original['Range'] = np.ptp(data)
    
    # Ajout outliers
    data_with_outliers = np.concatenate([data, outlier_values])
    
    # Calculs avec outliers
    measures_with_outliers['Écart-type'] = np.std(data_with_outliers, ddof=1)
    measures_with_outliers['IQR'] = np.percentile(data_with_outliers, 75) - np.percentile(data_with_outliers, 25)
    measures_with_outliers['MAD'] = median_abs_deviation(data_with_outliers, scale='normal')
    measures_with_outliers['Range'] = np.ptp(data_with_outliers)
    
    # Calcul pourcentage changement
    changes = {}
    for key in measures_original:
        original = measures_original[key]
        with_out = measures_with_outliers[key]
        pct_change = ((with_out - original) / original) * 100
        changes[key] = pct_change
    
    # Affichage
    print(f"\n{'='*70}")
    print(f"IMPACT OUTLIERS SUR MESURES DE DISPERSION - {name}")
    print(f"{'='*70}")
    print(f"Données originales : n={len(data)}, Range=[{np.min(data):.1f}, {np.max(data):.1f}]")
    print(f"Outliers ajoutés : {outlier_values}")
    print(f"\n{'Mesure':<15} | {'Original':>12} | {'Avec Outliers':>15} | {'Changement %':>15}")
    print("-"*70)
    for key in measures_original:
        orig = measures_original[key]
        with_out = measures_with_outliers[key]
        chg = changes[key]
        symbol = "🚨" if abs(chg) > 50 else ("⚠️" if abs(chg) > 20 else "✅")
        print(f"{key:<15} | {orig:>12.2f} | {with_out:>15.2f} | {chg:>14.1f}% {symbol}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Barplot comparaison
    measures_names = list(measures_original.keys())
    x = np.arange(len(measures_names))
    width = 0.35
    
    original_vals = [measures_original[m] for m in measures_names]
    outlier_vals = [measures_with_outliers[m] for m in measures_names]
    
    axes[0].bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color='blue')
    axes[0].bar(x + width/2, outlier_vals, width, label='Avec Outliers', alpha=0.8, color='red')
    axes[0].set_ylabel('Valeur')
    axes[0].set_title('Comparaison Mesures de Dispersion')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(measures_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Barplot % changement
    change_vals = [changes[m] for m in measures_names]
    colors = ['red' if abs(c) > 50 else 'orange' if abs(c) > 20 else 'green' for c in change_vals]
    
    axes[1].bar(measures_names, change_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_ylabel('Changement (%)')
    axes[1].set_title('Impact Outliers (% Changement)')
    axes[1].set_xticklabels(measures_names, rotation=45, ha='right')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return measures_original, measures_with_outliers, changes

# ========== TEST ==========
np.random.seed(123)

# Dataset normal (distribution raisonnablement symétrique)
data_normal = np.random.normal(loc=50, scale=5, size=100)

# Ajout outliers extrêmes
outliers = np.array([150, 200, 250])

compare_dispersion_measures(data_normal, outliers, name="Exemple Salaires")
```

### 3. Détection d'Anomalies Multi-Méthodes

```python
"""
Titre : Système de détection d'anomalies
Objectif : Comparer IQR, Z-score, MAD pour détecter outliers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def detect_outliers_comprehensive(data, methods=['iqr', 'zscore', 'mad'], visualize=True):
    """
    Détecte outliers par plusieurs méthodes et compare résultats.
    
    Args:
        data (array): Données
        methods (list): Liste méthodes ('iqr', 'zscore', 'mad')
        visualize (bool): Afficher graphiques
    
    Returns:
        dict: Résultats pour chaque méthode
    """
    results = {}
    
    # ========== MÉTHODE IQR ==========
    if 'iqr' in methods:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers_mask = (data < lower) | (data > upper)
        results['iqr'] = {
            'outliers_indices': np.where(outliers_mask)[0],
            'outliers_values': data[outliers_mask],
            'n_outliers': outliers_mask.sum(),
            'lower_bound': lower,
            'upper_bound': upper,
            'method_name': 'IQR (Tukey)'
        }
    
    # ========== MÉTHODE Z-SCORE ==========
    if 'zscore' in methods:
        z_scores = np.abs(stats.zscore(data))
        outliers_mask = z_scores > 3
        
        results['zscore'] = {
            'outliers_indices': np.where(outliers_mask)[0],
            'outliers_values': data[outliers_mask],
            'n_outliers': outliers_mask.sum(),
            'z_scores': z_scores,
            'threshold': 3,
            'method_name': 'Z-score (>3σ)'
        }
    
    # ========== MÉTHODE MAD ==========
    if 'mad' in methods:
        median = np.median(data)
        mad_value = stats.median_abs_deviation(data, scale='normal')
        
        # Seuil : médiane ± 3×MAD
        modified_z_scores = np.abs((data - median) / mad_value)
        outliers_mask = modified_z_scores > 3
        
        results['mad'] = {
            'outliers_indices': np.where(outliers_mask)[0],
            'outliers_values': data[outliers_mask],
            'n_outliers': outliers_mask.sum(),
            'modified_z_scores': modified_z_scores,
            'threshold': 3,
            'method_name': 'MAD (>3×MAD)'
        }
    
    # ========== AFFICHAGE ==========
    print(f"\n{'='*70}")
    print(f"DÉTECTION D'OUTLIERS - COMPARAISON MÉTHODES")
    print(f"{'='*70}")
    print(f"Données : n={len(data)}, Range=[{np.min(data):.1f}, {np.max(data):.1f}]")
    print()
    
    for method_key, result in results.items():
        print(f"{result['method_name']:<20} : {result['n_outliers']} outliers détectés")
        if result['n_outliers'] > 0 and result['n_outliers'] <= 10:
            print(f"  Valeurs : {result['outliers_values']}")
    
    # Consensus (outliers détectés par toutes méthodes)
    if len(results) > 1:
        all_indices = [set(r['outliers_indices']) for r in results.values()]
        consensus = set.intersection(*all_indices)
        print(f"\n🎯 CONSENSUS (détecté par toutes méthodes) : {len(consensus)} outliers")
        if len(consensus) > 0 and len(consensus) <= 10:
            print(f"  Indices : {sorted(list(consensus))}")
    
    # ========== VISUALISATION ==========
    if visualize and len(data) < 1000:  # Éviter surcharge si trop de points
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5*(n_methods+1), 5))
        
        if n_methods == 1:
            axes = [axes]
        
        # Plot original
        axes[0].scatter(range(len(data)), data, alpha=0.6, s=30, color='gray')
        axes[0].set_title('Données Originales')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Valeur')
        axes[0].grid(alpha=0.3)
        
        # Plot chaque méthode
        for idx, (method_key, result) in enumerate(results.items(), start=1):
            # Points normaux en bleu
            normal_mask = np.ones(len(data), dtype=bool)
            normal_mask[result['outliers_indices']] = False
            
            axes[idx].scatter(np.where(normal_mask)[0], data[normal_mask], 
                            alpha=0.6, s=30, color='blue', label='Normal')
            
            # Outliers en rouge
            if result['n_outliers'] > 0:
                axes[idx].scatter(result['outliers_indices'], result['outliers_values'],
                                alpha=0.8, s=50, color='red', marker='X', label='Outliers')
            
            # Lignes de seuil si applicable
            if 'lower_bound' in result:
                axes[idx].axhline(result['lower_bound'], color='orange', linestyle='--', alpha=0.7, label='Bornes')
                axes[idx].axhline(result['upper_bound'], color='orange', linestyle='--', alpha=0.7)
            
            axes[idx].set_title(f"{result['method_name']}\n({result['n_outliers']} outliers)")
            axes[idx].set_xlabel('Index')
            axes[idx].set_ylabel('Valeur')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results

# ========== TEST ==========
np.random.seed(456)

# Génération données avec outliers de différents types
n = 100
data_normal = np.random.normal(50, 10, n-10)
outliers_extreme = np.array([150, 160, 170])  # Très extrêmes
outliers_moderate = np.array([90, 95, 100, 102, 105, 108, 110])  # Modérés

data_test = np.concatenate([data_normal, outliers_extreme, outliers_moderate])
np.random.shuffle(data_test)

results = detect_outliers_comprehensive(data_test, methods=['iqr', 'zscore', 'mad'], visualize=True)
```

---

## 🔬 Exemples Concrets et Cas d'Usage

> **Principe de personnalisation** : Exemples progressifs du simple au complexe.

### Exemple 1 : Contrôle Qualité Industriel - Niveau Débutant

**Contexte** :
Vous travaillez dans une usine produisant des boulons. La longueur cible est 50mm. Vous mesurez 30 boulons.

**Données** :

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(789)

# Production Process A (machine bien calibrée)
process_A = np.random.normal(loc=50, scale=0.5, size=30)

# Production Process B (machine déréglée, haute variabilité)
process_B = np.random.normal(loc=50, scale=2.5, size=30)

print("PROCESSUS A (Machine 1) :")
stats_A = descriptive_statistics(process_A, name="Process A", show_plot=False)

print("\n\nPROCESSUS B (Machine 2) :")
stats_B = descriptive_statistics(process_B, name="Process B", show_plot=False)
```

**Analyse Comparative** :

```python
# Comparaison
print(f"\n{'='*70}")
print(f"COMPARAISON QUALITÉ - PROCESSUS A vs B")
print(f"{'='*70}")
print(f"{'Métrique':<20} | {'Process A':>15} | {'Process B':>15} | Meilleur")
print("-"*70)

metrics = {
    'Moyenne (cible=50)': (stats_A['mean'], stats_B['mean']),
    'Écart-type': (stats_A['std'], stats_B['std']),
    'CV (%)': (stats_A['cv'], stats_B['cv']),
    'Range': (stats_A['range'], stats_B['range']),
    'IQR': (stats_A['iqr'], stats_B['iqr'])
}

for metric, (val_a, val_b) in metrics.items():
    better = "A ✅" if val_a <= val_b else "B ✅"
    if metric == 'Moyenne (cible=50)':
        better = "A ✅" if abs(val_a - 50) <= abs(val_b - 50) else "B ✅"
    print(f"{metric:<20} | {val_a:>15.3f} | {val_b:>15.3f} | {better}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist([process_A, process_B], bins=20, label=['Process A', 'Process B'], 
             alpha=0.7, edgecolor='black')
axes[0].axvline(50, color='green', linestyle='--', linewidth=2, label='Cible (50mm)')
axes[0].set_xlabel('Longueur (mm)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution Longueurs Boulons')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot([process_A, process_B], labels=['Process A', 'Process B'],
                patch_artist=True)
axes[1].axhline(50, color='green', linestyle='--', linewidth=2, label='Cible')
axes[1].set_ylabel('Longueur (mm)')
axes[1].set_title('Comparaison Variabilité (Box Plot)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Interprétation** :
- **Process A** : σ ≈ 0.5mm, CV ≈ 1% → **Excellente consistance**, machine bien calibrée
- **Process B** : σ ≈ 2.5mm, CV ≈ 5% → **Haute variabilité**, machine à réviser
- **Décision** : Process A conforme aux standards Six Sigma (Cpk élevé)

**Analyse critique** :
- **Points forts** : Coefficient de variation permet comparaison objective de qualité
- **Limitations** : Ne montre pas dérives temporelles (voir cartes de contrôle)
- **Leçons apprises** : Faible variance = haute qualité = moins de rebuts

---

### Exemple 2 : Analyse de Risque Financier (Portfolio) - Niveau Intermédiaire

**Contexte** :
Vous gérez 2 portfolios d'investissement. Rendements mensuels sur 24 mois.

**Données** :

```python
np.random.seed(321)

# Portfolio A : Actions tech (haute volatilité, haut rendement)
portfolio_A_returns = np.random.normal(loc=1.5, scale=8, size=24)  # Moyenne 1.5%, σ=8%

# Portfolio B : Obligations (faible volatilité, faible rendement)
portfolio_B_returns = np.random.normal(loc=0.5, scale=2, size=24)  # Moyenne 0.5%, σ=2%

# Affichage stats
print("PORTFOLIO A (Actions Tech) :")
stats_portf_A = descriptive_statistics(portfolio_A_returns, name="Portfolio A", show_plot=False)

print("\n\nPORTFOLIO B (Obligations) :")
stats_portf_B = descriptive_statistics(portfolio_B_returns, name="Portfolio B", show_plot=False)
```

**Calcul Métriques Financières** :

```python
# ========== MÉTRIQUES RISQUE-RENDEMENT ==========

# Sharpe Ratio = (Rendement moyen - Taux sans risque) / Écart-type
risk_free_rate = 0.2  # 0.2% / mois (taux sans risque)

sharpe_A = (stats_portf_A['mean'] - risk_free_rate) / stats_portf_A['std']
sharpe_B = (stats_portf_B['mean'] - risk_free_rate) / stats_portf_B['std']

# VaR (Value at Risk) 95% = Perte maximale avec 95% confiance
var_95_A = np.percentile(portfolio_A_returns, 5)  # 5ème percentile (queue gauche)
var_95_B = np.percentile(portfolio_B_returns, 5)

# CVaR (Conditional VaR) = Perte moyenne au-delà du VaR
cvar_95_A = portfolio_A_returns[portfolio_A_returns <= var_95_A].mean()
cvar_95_B = portfolio_B_returns[portfolio_B_returns <= var_95_B].mean()

# Affichage
print(f"\n{'='*70}")
print(f"ANALYSE RISQUE-RENDEMENT")
print(f"{'='*70}")
print(f"{'Métrique':<25} | {'Portfolio A':>15} | {'Portfolio B':>15}")
print("-"*70)
print(f"{'Rendement moyen (%)':<25} | {stats_portf_A['mean']:>15.2f} | {stats_portf_B['mean']:>15.2f}")
print(f"{'Volatilité (σ) (%)':<25} | {stats_portf_A['std']:>15.2f} | {stats_portf_B['std']:>15.2f}")
print(f"{'Sharpe Ratio':<25} | {sharpe_A:>15.3f} | {sharpe_B:>15.3f}")
print(f"{'VaR 95% (%)':<25} | {var_95_A:>15.2f} | {var_95_B:>15.2f}")
print(f"{'CVaR 95% (%)':<25} | {cvar_95_A:>15.2f} | {cvar_95_B:>15.2f}")

print(f"\n💡 INTERPRÉTATION :")
print(f"  - Sharpe Ratio : Mesure rendement par unité de risque")
print(f"    Plus élevé = meilleur ({sharpe_A:.3f} vs {sharpe_B:.3f})")
print(f"  - VaR 95% : Perte maximale espérée 95% du temps")
print(f"    A: {var_95_A:.1f}%, B: {var_95_B:.1f}%")
print(f"  - Portfolio A : Risque {stats_portf_A['std']/stats_portf_B['std']:.1f}× plus élevé")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Série temporelle
mois = np.arange(1, 25)
axes[0, 0].plot(mois, portfolio_A_returns, marker='o', label='Portfolio A', alpha=0.7)
axes[0, 0].plot(mois, portfolio_B_returns, marker='s', label='Portfolio B', alpha=0.7)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0, 0].set_xlabel('Mois')
axes[0, 0].set_ylabel('Rendement (%)')
axes[0, 0].set_title('Rendements Mensuels')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Histogrammes
axes[0, 1].hist([portfolio_A_returns, portfolio_B_returns], bins=15, 
                label=['Portfolio A', 'Portfolio B'], alpha=0.7, edgecolor='black')
axes[0, 1].axvline(var_95_A, color='red', linestyle='--', label=f'VaR 95% A={var_95_A:.1f}%')
axes[0, 1].axvline(var_95_B, color='blue', linestyle='--', label=f'VaR 95% B={var_95_B:.1f}%')
axes[0, 1].set_xlabel('Rendement (%)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution Rendements')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Scatter Risque-Rendement
axes[1, 0].scatter(stats_portf_A['std'], stats_portf_A['mean'], s=200, alpha=0.7, 
                   label='Portfolio A', color='red', marker='o')
axes[1, 0].scatter(stats_portf_B['std'], stats_portf_B['mean'], s=200, alpha=0.7, 
                   label='Portfolio B', color='blue', marker='s')
axes[1, 0].set_xlabel('Risque (Écart-type %)')
axes[1, 0].set_ylabel('Rendement Moyen (%)')
axes[1, 0].set_title('Risque vs Rendement')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Box plot
axes[1, 1].boxplot([portfolio_A_returns, portfolio_B_returns], 
                    labels=['Portfolio A', 'Portfolio B'],
                    patch_artist=True)
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1, 1].set_ylabel('Rendement (%)')
axes[1, 1].set_title('Comparaison Variabilité')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Décision Investissement** :

```python
print(f"\n🎯 RECOMMANDATION SELON PROFIL INVESTISSEUR :")
print(f"\n1. PROFIL CONSERVATEUR (aversion au risque) :")
print(f"   → Portfolio B (Obligations)")
print(f"   Raison : Faible volatilité (σ={stats_portf_B['std']:.1f}%), VaR faible")

print(f"\n2. PROFIL AGRESSIF (tolérance au risque) :")
print(f"   → Portfolio A (Actions)")
print(f"   Raison : Sharpe Ratio supérieur ({sharpe_A:.3f} vs {sharpe_B:.3f})")
print(f"   Rendement moyen {stats_portf_A['mean']/stats_portf_B['mean']:.1f}× plus élevé")

print(f"\n3. PROFIL ÉQUILIBRÉ :")
print(f"   → Portefeuille mixte : 60% B + 40% A")
print(f"   Raison : Diversification réduit variance totale (corrélation faible)")
```

**Analyse critique** :
- **Points forts** : Écart-type = mesure universelle du risque financier
- **Limitations** : Suppose distribution symétrique (réalité = queues lourdes)
- **Leçons apprises** : Variance seule ne suffit pas, regarder aussi VaR/CVaR

---

### Exemple 3 : Feature Engineering pour ML - Niveau Avancé

**Contexte** :
Vous préparez features pour modèle de churn prediction. Créer features basées sur variance comportementale.

**Données** :

```python
import pandas as pd
import numpy as np

np.random.seed(654)

# Simulation données clients e-commerce (6 mois d'historique)
n_clients = 1000
n_mois = 6

# Génération patterns
data_clients = []

for client_id in range(n_clients):
    # 20% clients churners (comportement erratique)
    is_churner = np.random.random() < 0.2
    
    if is_churner:
        # Churners : montants très variables, décroissance progressive
        montants = np.random.gamma(shape=2, scale=50, size=n_mois) * np.linspace(1, 0.3, n_mois)
        nb_visites = np.random.poisson(lam=3, size=n_mois)
    else:
        # Clients fidèles : montants stables
        montants = np.random.gamma(shape=5, scale=20, size=n_mois)
        nb_visites = np.random.poisson(lam=8, size=n_mois)
    
    data_clients.append({
        'client_id': client_id,
        'is_churner': is_churner,
        'montants': montants,
        'nb_visites': nb_visites
    })

df_clients = pd.DataFrame(data_clients)
```

**Feature Engineering basé Dispersion** :

```python
# ========== CRÉATION FEATURES DE DISPERSION ==========

def compute_behavioral_features(client_data):
    """
    Calcule features comportementales basées sur variabilité.
    """
    montants = client_data['montants']
    visites = client_data['nb_visites']
    
    features = {
        # Features tendance centrale
        'montant_moyen': np.mean(montants),
        'montant_median': np.median(montants),
        'visites_moyennes': np.mean(visites),
        
        # Features dispersion (CLÉS POUR CHURN)
        'montant_std': np.std(montants, ddof=1),
        'montant_cv': (np.std(montants, ddof=1) / np.mean(montants) * 100) if np.mean(montants) > 0 else 0,
        'montant_iqr': np.percentile(montants, 75) - np.percentile(montants, 25),
        'montant_range': np.ptp(montants),
        
        # Variabilité temporelle (trend)
        'montant_trend': np.polyfit(range(len(montants)), montants, deg=1)[0],  # Pente régression
        'montant_mad': stats.median_abs_deviation(montants, scale='normal'),
        
        # Ratio dispersion
        'ratio_std_mean': np.std(montants, ddof=1) / np.mean(montants) if np.mean(montants) > 0 else 0,
        
        # Stabilité (inverse CV)
        'stabilite_score': 1 / (1 + np.std(montants, ddof=1) / np.mean(montants)) if np.mean(montants) > 0 else 0,
        
        # Target
        'is_churner': client_data['is_churner']
    }
    
    return features

# Application sur tous clients
features_list = [compute_behavioral_features(row) for _, row in df_clients.iterrows()]
df_features = pd.DataFrame(features_list)

print("FEATURES CRÉÉES :")
print(df_features.head(10))
```

**Analyse Pouvoir Prédictif** :

```python
# Comparaison Churners vs Non-Churners

print(f"\n{'='*80}")
print(f"COMPARAISON FEATURES : CHURNERS vs NON-CHURNERS")
print(f"{'='*80}")

features_to_compare = ['montant_moyen', 'montant_std', 'montant_cv', 'montant_trend', 'stabilite_score']

for feature in features_to_compare:
    churners = df_features[df_features['is_churner'] == True][feature]
    non_churners = df_features[df_features['is_churner'] == False][feature]
    
    mean_churn = churners.mean()
    mean_non_churn = non_churners.mean()
    
    # Test t pour significativité
    t_stat, p_value = stats.ttest_ind(churners, non_churners)
    
    print(f"\n{feature.upper()}")
    print(f"  Churners     : {mean_churn:.3f}")
    print(f"  Non-Churners : {mean_non_churn:.3f}")
    print(f"  Différence   : {abs(mean_churn - mean_non_churn):.3f}")
    print(f"  p-value      : {p_value:.4f} {'✅ Significatif' if p_value < 0.05 else '❌ Non significatif'}")

# Visualisation
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(features_to_compare):
    churners = df_features[df_features['is_churner'] == True][feature]
    non_churners = df_features[df_features['is_churner'] == False][feature]
    
    axes[idx].hist([non_churners, churners], bins=30, label=['Non-Churners', 'Churners'], 
                   alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Fréquence')
    axes[idx].set_title(f'Distribution {feature}')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

axes[5].axis('off')  # Dernier subplot vide

plt.tight_layout()
plt.show()
```

**Modèle ML Simple (Logistic Regression)** :

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Préparation données
X = df_features[features_to_compare].values
y = df_features['is_churner'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entraînement
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Évaluation
print(f"\n{'='*70}")
print("PERFORMANCE MODÈLE CHURN PREDICTION")
print(f"{'='*70}")
print(classification_report(y_test, y_pred, target_names=['Non-Churner', 'Churner']))
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_pred_proba):.3f}")

# Importance features
feature_importance = pd.DataFrame({
    'Feature': features_to_compare,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nIMPORTANCE FEATURES (Coefficients Logistic Regression) :")
print(feature_importance)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC - Churn Prediction')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Interprétation** :

```python
print(f"\n💡 INSIGHTS :")
print(f"1. Features de DISPERSION (std, CV, MAD) sont prédictives du churn")
print(f"2. Churners ont comportement PLUS VARIABLE (CV élevé)")
print(f"3. 'stabilite_score' (inverse CV) = feature puissante")
print(f"4. Trend négatif (montant_trend < 0) corrélé au churn")
print(f"\n🎯 Leçon : Variance comportementale = signal fort pour churn/anomalie !")
```

**Analyse critique** :
- **Points forts** : Features de dispersion capturent patterns temporels
- **Limitations** : Nécessite historique suffisant (min 3-6 mois)
- **Leçons apprises** : Variabilité = information aussi importante que niveau moyen

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Oublier `ddof=1` pour Variance Échantillon

**Description** :
Utiliser `ddof=0` (défaut NumPy) au lieu de `ddof=1` conduit à sous-estimer variance.

**Exemple problématique** :

```python
# ❌ MAUVAIS
data = np.array([10, 20, 30, 40, 50])
var_wrong = np.var(data)  # ddof=0 par défaut
std_wrong = np.std(data)

print(f"Variance (ddof=0) : {var_wrong}")  # 200.0 (BIAISÉ)
```

**Solution** :

```python
# ✅ BON
var_correct = np.var(data, ddof=1)  # Correction Bessel
std_correct = np.std(data, ddof=1)

print(f"Variance (ddof=1) : {var_correct}")  # 250.0 (NON BIAISÉ)

# Ou utiliser Pandas (ddof=1 par défaut)
import pandas as pd
var_pandas = pd.Series(data).var()  # ddof=1 automatique
print(f"Variance (Pandas) : {var_pandas}")  # 250.0
```

**Impact** : Sous-estimation de la vraie variabilité → Intervalles de confiance trop étroits, tests statistiques biaisés.

---

### ❌ Erreur 2 : Utiliser CV avec Moyenne Proche de Zéro

**Description** :
Le coefficient de variation explose si moyenne proche de zéro.

**Exemple problématique** :

```python
# ❌ MAUVAIS
temperatures_celsius = np.array([-2, -1, 0, 1, 2])
mean_temp = np.mean(temperatures_celsius)  # 0°C
std_temp = np.std(temperatures_celsius, ddof=1)

cv = (std_temp / mean_temp) * 100  # Division par zéro !
print(f"CV : {cv}")  # inf ou nan
```

**Solution** :

```python
# ✅ BON - Option 1 : Utiliser échelle absolue (Kelvin)
temperatures_kelvin = temperatures_celsius + 273.15
cv_kelvin = (np.std(temperatures_kelvin, ddof=1) / np.mean(temperatures_kelvin)) * 100
print(f"CV (Kelvin) : {cv_kelvin:.2f}%")  # ValParfait ! Je crée maintenant le cours **[[measures_dispersion]]** - Mesures de Dispersion. 🚀

---

````md
# 📏 Mesures de Dispersion : Variance, Écart-type, Quantiles

> **Résumé en une phrase** : Les mesures de dispersion quantifient la variabilité des données autour de leur centre, permettant de comprendre si vos données sont concentrées ou très dispersées.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-18 |
| **Dernière mise à jour** | 2026-03-18 |
| **Domaine** | Statistiques Descriptives |
| **Niveau** | Débutant |
| **Durée de lecture** | ~35 minutes |
| **Fichier** | `measures_dispersion.md` |
| **Emplacement** | `/00_statistics_foundations/01_descriptive_statistics/` |
| **Tags** | `#statistics` `#descriptive` `#variance` `#standard-deviation` `#quantiles` `#IQR` `#dispersion` |

### Prérequis

- [x] [[measures_central_tendency]] - Moyenne, médiane, mode (ESSENTIEL)
- [ ] Mathématiques de base (puissances, racines carrées)
- [ ] Notions de base en Python (optionnel pour partie code)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[measures_central_tendency]] - Mesures de tendance centrale
- **Complémentaires** : 
  - [[data_visualization_principles]] - Box plots, violin plots
  - [[distribution_analysis]] - Skewness, kurtosis
- **Suite recommandée** : 
  - [[probability_foundations]] - Variables aléatoires et variance théorique
  - [[confidence_intervals]] - Utilisation de l'écart-type pour IC
  - [[hypothesis_testing]] - Tests basés sur variance

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Les mesures de tendance centrale (moyenne, médiane) ne racontent que **la moitié de l'histoire**. Deux datasets peuvent avoir exactement la même moyenne mais des **distributions complètement différentes**. Ce cours vous apprend à **quantifier la variabilité** de vos données : sont-elles concentrées autour de la moyenne ou très dispersées ? Vous maîtriserez les outils essentiels (variance, écart-type, quantiles) et saurez **quand** utiliser chacun selon votre contexte métier.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Interpréter variance, écart-type, IQR et leur signification concrète
2. **Appliquer** : Calculer ces mesures manuellement et avec NumPy/Pandas
3. **Analyser** : Identifier outliers avec méthodes basées sur écart-type ou IQR
4. **Évaluer** : Choisir la mesure appropriée selon distribution (normale vs asymétrique)
5. **Créer** : Construire des box plots et interpréter leur structure (Q1, Q2, Q3, whiskers)
6. **Synthétiser** : Combiner mesures centrales + dispersion pour décrire complètement une distribution

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

Imaginez deux machines produisant des pièces métalliques de 10.0 cm :

**Machine A** : 9.99, 10.00, 10.01, 10.00, 9.99, 10.01 → Moyenne = 10.00 cm  
**Machine B** : 8.50, 11.20, 9.30, 12.00, 8.00, 11.00 → Moyenne = 10.00 cm

**Même moyenne, mais Machine B est inutilisable !** La **dispersion** mesure cette différence cruciale :
- Machine A : Faible dispersion → Qualité excellente
- Machine B : Forte dispersion → Défauts de fabrication

En Data Science, la dispersion révèle :
1. **La fiabilité** : Prédictions avec faible variance = modèle stable
2. **Les outliers** : Points très éloignés de la moyenne/médiane
3. **L'hétérogénéité** : Forte dispersion = population non homogène
4. **L'incertitude** : Large variance = prédictions moins fiables

### Quel problème résout-il ?

**Problème** : Vous comparez deux modèles ML prédisant des prix immobiliers. Les deux ont la même erreur moyenne (MAE = 20k€), mais lequel est meilleur ?

**Exemple concret** :

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Modèle A : Erreurs concentrées autour de 0
erreurs_modele_A = np.random.normal(loc=0, scale=15, size=1000)

# Modèle B : Erreurs très variables
erreurs_modele_B = np.random.normal(loc=0, scale=40, size=1000)

# Même MAE (par construction avec biais nul)
mae_A = np.mean(np.abs(erreurs_modele_A))
mae_B = np.mean(np.abs(erreurs_modele_B))

# Mais écart-types différents !
std_A = np.std(erreurs_modele_A, ddof=1)
std_B = np.std(erreurs_modele_B, ddof=1)

print(f"Modèle A - MAE: {mae_A:.1f}k€, Std: {std_A:.1f}k€")
print(f"Modèle B - MAE: {mae_B:.1f}k€, Std: {std_B:.1f}k€")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(erreurs_modele_A, bins=50, alpha=0.7, edgecolor='black', color='green')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'Modèle A: Std = {std_A:.1f}k€ (Fiable)')
axes[0].set_xlabel('Erreur de Prédiction (k€)')
axes[0].set_ylabel('Fréquence')
axes[0].grid(alpha=0.3)

axes[1].hist(erreurs_modele_B, bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title(f'Modèle B: Std = {std_B:.1f}k€ (Non fiable)')
axes[1].set_xlabel('Erreur de Prédiction (k€)')
axes[1].set_ylabel('Fréquence')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Résultat** :
- **Modèle A** : Std ≈ 15k€ → Erreurs prévisibles, concentrées
- **Modèle B** : Std ≈ 40k€ → Erreurs erratiques, quelques prédictions très fausses

**Le Modèle A est supérieur** : Même MAE moyenne, mais dispersion plus faible = prédictions plus fiables.

### Applications dans le monde réel

1. **Finance** :
   - Volatilité d'un actif (écart-type des rendements)
   - Risque d'un portefeuille (variance = mesure standard)
   - Value at Risk (VaR) : Percentile 5% des pertes

2. **Qualité / Manufacturing** :
   - Contrôle qualité : écart-type des mesures < tolérance
   - Six Sigma : Processus avec ±6σ de la cible
   - Capabilité processus (Cp, Cpk)

3. **Machine Learning** :
   - Variance des prédictions (modèles ensemble)
   - Biais-variance tradeoff (cours [[bias_variance_tradeoff]])
   - Normalisation/Standardisation des features (division par écart-type)

4. **A/B Testing** :
   - Puissance statistique dépend de la variance
   - Calcul de sample size nécessite estimation de variance

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous commençons par les mesures basées sur la **moyenne** (variance, écart-type), puis les mesures basées sur les **quantiles** (IQR), et enfin les comparons.

### 1. La Variance

#### 1.1 Définition Mathématique

**Variance de population** :

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

**Variance d'échantillon** (la plus utilisée en pratique) :

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Où** :
- $$\sigma^2$$ (sigma carré) : Variance de la population
- $$s^2$$ : Variance de l'échantillon
- $$N$$ : Taille de la population
- $$n$$ : Taille de l'échantillon
- $$\mu$$ : Moyenne de la population
- $$\bar{x}$$ : Moyenne de l'échantillon
- $$(x_i - \bar{x})^2$$ : Carré de l'écart à la moyenne (toujours positif !)

**Intuition** :
La variance est la **moyenne des carrés des écarts à la moyenne**. Elle quantifie "à quel point les données s'éloignent en moyenne de la moyenne".

**Pourquoi cette définition ?**

1. **Pourquoi élever au carré ?**
   - Sans carré : $$\sum (x_i - \bar{x}) = 0$$ (les écarts positifs et négatifs s'annulent)
   - Avec carré : Tous les écarts deviennent positifs et se somment
   - Pénalise davantage les écarts importants (écart de 10 pèse 100, écart de 2 pèse 4)

2. **Pourquoi diviser par $$n-1$$ et pas $$n$$ ?**
   - **Correction de Bessel** : Sans elle, la variance échantillonnale sous-estime systématiquement la vraie variance
   - $$n-1$$ = "degrés de liberté" : Après avoir calculé $$\bar{x}$$, seulement $$n-1$$ valeurs sont "libres"
   - En pratique, différence minime pour grand $$n$$, mais importante théoriquement

**Visualisation conceptuelle** :

```
Données : [2, 4, 4, 4, 5, 5, 7, 9]
Moyenne : 5

Écarts :     [-3, -1, -1, -1,  0,  0,  2,  4]
Carrés :     [ 9,  1,  1,  1,  0,  0,  4, 16] = 32
Variance :   32 / (8-1) = 32 / 7 ≈ 4.57
```

#### 1.2 Propriétés Mathématiques de la Variance

**Propriété 1 : Unité au carré**

Si $$x$$ est en mètres, $$s^2$$ est en mètres². **Problème** : Difficile à interpréter directement → D'où l'écart-type !

**Propriété 2 : Variance d'une constante = 0**

$$\text{Var}(c) = 0$$

Si toutes les valeurs sont identiques, pas de dispersion.

**Propriété 3 : Ajouter constante ne change pas variance**

$$\text{Var}(X + c) = \text{Var}(X)$$

Décaler toutes les données de +10 ne change pas leur dispersion.

**Propriété 4 : Multiplier par constante multiplie variance par $$c^2$$**

$$\text{Var}(cX) = c^2 \cdot \text{Var}(X)$$

```python
# Exemple
data = np.array([1, 2, 3, 4, 5])
var_originale = np.var(data, ddof=1)

# Multiplier par 3
data_x3 = 3 * data
var_x3 = np.var(data_x3, ddof=1)

print(f"Var(X)   : {var_originale:.2f}")
print(f"Var(3X)  : {var_x3:.2f}")
print(f"9*Var(X) : {9 * var_originale:.2f}")  # 9 = 3²
```

**Propriété 5 : Variance de la somme (variables indépendantes)**

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

(Si $$X$$ et $$Y$$ sont indépendantes. Sinon : $$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$)

**Propriété 6 : Formule alternative (computationnelle)**

$$s^2 = \frac{1}{n-1} \left( \sum_{i=1}^{n} x_i^2 - n\bar{x}^2 \right)$$

Plus stable numériquement pour calculs informatiques.

#### 1.3 Calcul Efficace de la Variance

```python
import numpy as np

data = np.array([10, 12, 23, 23, 16, 23, 21, 16])

# Méthode 1 : NumPy (recommandée)
var_numpy = np.var(data, ddof=1)  # ddof=1 pour variance échantillon
print(f"Variance (NumPy) : {var_numpy:.2f}")

# Méthode 2 : Formule définition (pédagogique)
moyenne = np.mean(data)
ecarts_carres = (data - moyenne) ** 2
var_manuelle = np.sum(ecarts_carres) / (len(data) - 1)
print(f"Variance (manuelle) : {var_manuelle:.2f}")

# Méthode 3 : Formule computationnelle
var_comp = (np.sum(data**2) - len(data) * moyenne**2) / (len(data) - 1)
print(f"Variance (computationnelle) : {var_comp:.2f}")

# Vérification égalité
print(f"Égalité : {np.allclose([var_numpy, var_manuelle, var_comp], var_numpy)}")
```

**Sources académiques** :
- Wackerly, D., Mendenhall, W., & Scheaffer, R. (2008). *Mathematical Statistics with Applications*. Thomson
- [NumPy variance documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html)

---

### 2. L'Écart-type (Standard Deviation)

#### 2.1 Définition Mathématique

**Écart-type** :

$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

**Intuition** :
L'écart-type est la **racine carrée de la variance**. Il représente "l'écart typique moyen" d'un point par rapport à la moyenne, **dans les mêmes unités que les données**.

**Pourquoi prendre la racine carrée de la variance ?**

1. **Unité cohérente** : Si $$x$$ en mètres → $$s$$ en mètres (pas m²)
2. **Interprétabilité** : "Les données varient typiquement de ±15k€" est plus clair que "variance = 225 millions €²"
3. **Usage pratique** : Directement comparable aux données originales

#### 2.2 Interprétation avec la Règle Empirique (68-95-99.7)

**Pour distribution normale** (Gaussienne) :

- **68%** des données dans $$[\bar{x} - s, \bar{x} + s]$$
- **95%** des données dans $$[\bar{x} - 2s, \bar{x} + 2s]$$
- **99.7%** des données dans $$[\bar{x} - 3s, \bar{x} + 3s]$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Génération distribution normale
np.random.seed(123)
data_normale = np.random.normal(loc=100, scale=15, size=10000)

moyenne = np.mean(data_normale)
std = np.std(data_normale, ddof=1)

# Calcul pourcentages empiriques
pct_1std = np.sum((data_normale >= moyenne - std) & (data_normale <= moyenne + std)) / len(data_normale) * 100
pct_2std = np.sum((data_normale >= moyenne - 2*std) & (data_normale <= moyenne + 2*std)) / len(data_normale) * 100
pct_3std = np.sum((data_normale >= moyenne - 3*std) & (data_normale <= moyenne + 3*std)) / len(data_normale) * 100

print(f"Moyenne : {moyenne:.2f}")
print(f"Écart-type : {std:.2f}")
print(f"\nRègle 68-95-99.7 (vérification empirique) :")
print(f"±1σ : {pct_1std:.1f}% (théorique: 68%)")
print(f"±2σ : {pct_2std:.1f}% (théorique: 95%)")
print(f"±3σ : {pct_3std:.1f}% (théorique: 99.7%)")

# Visualisation
fig, ax = plt.subplots(figsize=(12, 6))

# Histogramme
ax.hist(data_normale, bins=100, density=True, alpha=0.7, edgecolor='black', color='skyblue')

# Courbe théorique normale
x_range = np.linspace(data_normale.min(), data_normale.max(), 1000)
ax.plot(x_range, norm.pdf(x_range, moyenne, std), 'r-', linewidth=2, label='Distribution Normale Théorique')

# Zones ±1σ, ±2σ, ±3σ
ax.axvline(moyenne, color='black', linestyle='-', linewidth=2, label=f'Moyenne = {moyenne:.1f}')
ax.axvline(moyenne - std, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(moyenne + std, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'±1σ (68%)')
ax.axvline(moyenne - 2*std, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(moyenne + 2*std, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'±2σ (95%)')
ax.axvline(moyenne - 3*std, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(moyenne + 3*std, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'±3σ (99.7%)')

ax.set_xlabel('Valeur')
ax.set_ylabel('Densité')
ax.set_title('Règle Empirique 68-95-99.7 pour Distribution Normale')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**⚠️ Attention** : Cette règle ne s'applique rigoureusement qu'aux **distributions normales** ! Pour distributions asymétriques, utilisez **inégalité de Tchebychev** (voir ci-dessous).

#### 2.3 Inégalité de Tchebychev (Valable TOUTE Distribution)

Pour **n'importe quelle distribution** (même non normale), au moins cette proportion de données est dans $$[\bar{x} - ks, \bar{x} + ks]$$ :

$$1 - \frac{1}{k^2}$$

**Exemples** :
- $$k=2$$ : Au moins $$1 - 1/4 = 75\%$$ dans $$\bar{x} \pm 2s$$
- $$k=3$$ : Au moins $$1 - 1/9 = 89\%$$ dans $$\bar{x} \pm 3s$$
- $$k=4$$ : Au moins $$1 - 1/16 = 94\%$$ dans $$\bar{x} \pm 4s$$

C'est une **borne inférieure garantie**, quelle que soit la forme de la distribution !

```python
# Test sur distribution très asymétrique (log-normale)
data_asymetrique = np.random.lognormal(mean=3, sigma=1, size=10000)

moyenne_asym = np.mean(data_asymetrique)
std_asym = np.std(data_asymetrique, ddof=1)

# Vérification Tchebychev
for k in [2, 3, 4]:
    dans_intervalle = np.sum((data_asymetrique >= moyenne_asym - k*std_asym) & 
                              (data_asymetrique <= moyenne_asym + k*std_asym)) / len(data_asymetrique) * 100
    borne_tchebychev = (1 - 1/k**2) * 100
    
    print(f"±{k}σ : {dans_intervalle:.1f}% (Tchebychev garantit ≥{borne_tchebychev:.1f}%)")
```

**Sources académiques** :
- [Chebyshev's Inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) - Démonstration et applications

---

### 3. Coefficient de Variation (CV)

#### 3.1 Définition

**Coefficient de variation** :

$$CV = \frac{s}{\bar{x}} \times 100\%$$

**Intuition** :
Le CV est l'écart-type **relatif** à la moyenne, exprimé en pourcentage. Il permet de comparer la variabilité de datasets ayant des **échelles différentes**.

**Pourquoi cette mesure ?**

Problème : Comparer directement écarts-types de datasets d'échelles différentes n'a pas de sens.

**Exemple** :
- Dataset A : Poids de souris (moyenne 25g, std 3g)
- Dataset B : Poids d'éléphants (moyenne 5000kg, std 500kg)

Qui est plus variable ? Impossible de comparer 3g vs 500kg !

**Solution avec CV** :
- CV(A) = 3/25 = 12%
- CV(B) = 500/5000 = 10%

Les souris sont **plus variables** relativement à leur poids moyen.

#### 3.2 Interprétation et Usage

**Règles générales** :
- **CV < 10%** : Faible variabilité (homogène)
- **10% ≤ CV ≤ 20%** : Variabilité modérée
- **CV > 20%** : Forte variabilité (hétérogène)
- **CV > 100%** : Variabilité extrême (souvent problème qualité données)

**Quand utiliser le CV ?**

✅ **Situations appropriées** :
- Comparer variabilités d'échelles différentes (poids souris vs éléphants)
- Comparer processus de mesures différentes (température vs pression)
- Contrôle qualité inter-produits
- Finance : Ratio de Sharpe (rendement / volatilité)

❌ **Situations inappropriées** :
- Moyenne proche de zéro (CV explose)
- Données avec valeurs négatives (CV perd son sens)
- Échelles arbitraires (ex: températures en Celsius vs Fahrenheit)

```python
import numpy as np

# Exemple : Comparer variabilité de différentes mesures
mesures = {
    'Température (°C)': {'data': np.array([19.5, 20.1, 20.3, 19.8, 20.0, 20.2])},
    'Pression (hPa)': {'data': np.array([1012, 1015, 1018, 1011, 1016, 1013])},
    'Humidité (%)': {'data': np.array([65, 72, 68, 70, 66, 69])}
}

print("COMPARAISON DE VARIABILITÉ AVEC COEFFICIENT DE VARIATION")
print("="*70)
print(f"{'Mesure':<20} | {'Moyenne':>10} | {'Std':>10} | {'CV':>10}")
print("-"*70)

for nom, info in mesures.items():
    data = info['data']
    moyenne = np.mean(data)
    std = np.std(data, ddof=1)
    cv = (std / moyenne) * 100
    
    print(f"{nom:<20} | {moyenne:>10.2f} | {std:>10.2f} | {cv:>9.2f}%")

print("\n💡 L'humidité a le CV le plus élevé → mesure la plus variable relativement")
```

**Sources académiques** :
- Reed, G. F., Lynn, F., & Meade, B. D. (2002). "Use of Coefficient of Variation in Assessing Variability". *Clinical and Diagnostic Laboratory Immunology*.

---

### 4. Quantiles, Percentiles et Quartiles

#### 4.1 Définitions

**Quantile d'ordre $$p$$** (où $$0 < p < 1$$) : Valeur $$q_p$$ telle que **proportion $$p$$** des données sont ≤ $$q_p$$.

**Percentile** : Quantile exprimé en pourcentage (0-100%).
- $$P_{25}$$ = 25ème percentile = 1er quartile (Q1)
- $$P_{50}$$ = 50ème percentile = Médiane (Q2)
- $$P_{75}$$ = 75ème percentile = 3ème quartile (Q3)

**Quartiles** : Valeurs qui divisent les données ordonnées en 4 parts égales.
- **Q1** (1er quartile) : 25% des données en-dessous
- **Q2** (2ème quartile) : 50% en-dessous = **Médiane**
- **Q3** (3ème quartile) : 75% en-dessous

**Visualisation** :

```
Données triées : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                     ↑        ↑        ↑
                     Q1       Q2       Q3
                    (3.25)   (6.5)   (9.75)
                     
25% des données < Q1
50% des données < Q2 (médiane)
75% des données < Q3
```

#### 4.2 Calcul des Quantiles

**Il existe plusieurs méthodes** de calcul des quantiles (NumPy en propose 9 !). La plus courante :

**Méthode linéaire** (NumPy par défaut) :

1. Trier les données
2. Calculer position : $$\text{pos} = p \times (n - 1) + 1$$
3. Si pos entier → prendre cette valeur
4. Si pos décimal → interpoler linéairement

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Calcul avec NumPy
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # = médiane
q3 = np.percentile(data, 75)

print(f"Q1 (25ème percentile) : {q1}")
print(f"Q2 (50ème percentile) : {q2}")
print(f"Q3 (75ème percentile) : {q3}")

# Calcul manuel Q1 (pour comprendre)
n = len(data)
pos_q1 = 0.25 * (n - 1) + 1  # Position = 3.25
# → Interpolation entre valeurs en positions 3 et 4
data_sorted = np.sort(data)
# Position 3 (index 2) = 3, Position 4 (index 3) = 4
q1_manuel = data_sorted[2] + 0.25 * (data_sorted[3] - data_sorted[2])
print(f"\nQ1 manuel : {q1_manuel}")

# Percentiles multiples en une fois
percentiles = np.percentile(data, [10, 25, 50, 75, 90, 95, 99])
print(f"\nPercentiles [10, 25, 50, 75, 90, 95, 99] :")
print(percentiles)
```

**Note** : La médiane calculée par `np.median()` peut légèrement différer de `np.percentile(data, 50)` selon la méthode d'interpolation. En pratique, différence négligeable.

#### 4.3 Interprétation des Percentiles

**Percentiles en contexte** :

| Percentile | Signification | Usage typique |
|------------|---------------|---------------|
| P5, P95 | 90% des données entre ces valeurs | SLAs web (P95 < 200ms) |
| P25, P75 | Quartiles, utilisés pour IQR | Box plots, détection outliers |
| P50 | Médiane | Mesure de tendance centrale robuste |
| P90, P99 | Queues de distribution | Latences extrêmes, risques |
| P1, P99 | Valeurs extrêmes | Value at Risk (VaR) en finance |

**Exemple Finance** : VaR 95% = Perte maximale attendue dans 95% des cas (= percentile 5% des rendements)

**Exemple Performance Web** : "P95 latency < 300ms" signifie "95% des requêtes traitées en < 300ms"

```python
# Exemple : Latences API
np.random.seed(456)
latences_ms = np.random.lognormal(mean=3, sigma=0.8, size=10000)

# Percentiles clés
p50 = np.percentile(latences_ms, 50)
p95 = np.percentile(latences_ms, 95)
p99 = np.percentile(latences_ms, 99)

print("LATENCES API (SLA Performance)")
print("="*50)
print(f"P50 (médiane)  : {p50:.1f} ms")
print(f"P95            : {p95:.1f} ms")
print(f"P99            : {p99:.1f} ms")
print(f"\n✅ SLA respecté si P95 < 200ms : {'OUI' if p95 < 200 else 'NON'}")
```

---

### 5. Intervalle Interquartile (IQR)

#### 5.1 Définition

**IQR (Interquartile Range)** :

$$IQR = Q3 - Q1$$

**Intuition** :
L'IQR mesure la **dispersion des 50% centraux** des données (entre Q1 et Q3). C'est une mesure de dispersion **robuste aux outliers**, contrairement à l'écart-type.

**Pourquoi l'IQR est-il robuste ?**

Les outliers affectent Q1 et Q3 minimalement (seulement s'ils changent leur position dans le tri), donc l'IQR reste stable.

```python
import numpy as np

# Données sans outliers
data_clean = np.array([10, 12, 14, 15, 16, 18, 20, 22, 24, 26])
q1_clean = np.percentile(data_clean, 25)
q3_clean = np.percentile(data_clean, 75)
iqr_clean = q3_clean - q1_clean

print("SANS OUTLIERS :")
print(f"Q1 = {q1_clean}, Q3 = {q3_clean}, IQR = {iqr_clean}")

# Ajout d'outliers extrêmes
data_outliers = np.append(data_clean, [100, 200, 300])
q1_outliers = np.percentile(data_outliers, 25)
q3_outliers = np.percentile(data_outliers, 75)
iqr_outliers = q3_outliers - q1_outliers

print("\nAVEC OUTLIERS (100, 200, 300) :")
print(f"Q1 = {q1_outliers}, Q3 = {q3_outliers}, IQR = {iqr_outliers}")
print(f"\n💡 IQR quasi inchangé : {iqr_clean} → {iqr_outliers}")

# Comparaison avec écart-type (très sensible)
std_clean = np.std(data_clean, ddof=1)
std_outliers = np.std(data_outliers, ddof=1)
print(f"\nÉcart-type (sensible) : {std_clean:.1f} → {std_outliers:.1f} (+{((std_outliers/std_clean - 1)*100):.0f}%)")
```

#### 5.2 Détection d'Outliers avec la Méthode IQR (Règle de Tukey)

**Règle standard** :

Un point est considéré **outlier** si :

$$x < Q1 - 1.5 \times IQR \quad \text{ou} \quad x > Q3 + 1.5 \times IQR$$

**Outlier extrême** (parfois distingué) :

$$x < Q1 - 3 \times IQR \quad \text{ou} \quad x > Q3 + 3 \times IQR$$

**Visualisation avec Box Plot** :

```
            Outliers ●
                |
    ┌───────────┴───────────┐
    │     Q3 + 1.5×IQR      │ ← Whisker supérieur
    │                        │
    ├───────────────────────┤ ← Q3
    │                        │
    │         BOX            │
    │     (IQR region)       │
    │                        │
    ├───────────────────────┤ ← Q2 (Médiane)
    │                        │
    ├───────────────────────┤ ← Q1
    │                        │
    │     Q1 - 1.5×IQR      │ ← Whisker inférieur
    └───────────┬───────────┘
                |
            Outliers ●
```

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_outliers_iqr(data):
    """
    Détecte outliers avec méthode IQR.
    
    Returns:
        dict: Indices outliers, bornes, statistiques
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Bornes
    borne_inf = q1 - 1.5 * iqr
    borne_sup = q3 + 1.5 * iqr
    
    # Détection
    outliers_mask = (data < borne_inf) | (data > borne_sup)
    outliers_indices = np.where(outliers_mask)[0]
    outliers_values = data[outliers_mask]
    
    return {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'borne_inf': borne_inf,
        'borne_sup': borne_sup,
        'n_outliers': len(outliers_indices),
        'outliers_indices': outliers_indices,
        'outliers_values': outliers_values,
        'pct_outliers': len(outliers_indices) / len(data) * 100
    }

# Exemple
np.random.seed(789)
data_test = np.concatenate([
    np.random.normal(50, 10, 100),  # Données normales
    np.array([5, 8, 95, 100])  # Outliers
])

result = detect_outliers_iqr(data_test)

print("DÉTECTION OUTLIERS (Méthode IQR)")
print("="*60)
print(f"Q1             : {result['q1']:.2f}")
print(f"Q3             : {result['q3']:.2f}")
print(f"IQR            : {result['iqr']:.2f}")
print(f"Borne inférieure : {result['borne_inf']:.2f}")
print(f"Borne supérieure : {result['borne_sup']:.2f}")
print(f"\nOutliers détectés : {result['n_outliers']} ({result['pct_outliers']:.1f}%)")
print(f"Valeurs outliers : {result['outliers_values']}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
axes[0].boxplot(data_test, vert=True)
axes[0].scatter(np.ones(len(result['outliers_values'])), result['outliers_values'], 
                color='red', s=100, zorder=3, label='Outliers IQR')
axes[0].set_ylabel('Valeur')
axes[0].set_title('Box Plot avec Outliers Détectés')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Histogramme avec bornes
axes[1].hist(data_test, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(result['borne_inf'], color='red', linestyle='--', linewidth=2, label='Bornes IQR')
axes[1].axvline(result['borne_sup'], color='red', linestyle='--', linewidth=2)
axes[1].axvline(result['q1'], color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Q1, Q3')
axes[1].axvline(result['q3'], color='blue', linestyle='--', linewidth=2, alpha=0.5)
axes[1].set_xlabel('Valeur')
axes[1].set_ylabel('Fréquence')
axes[1].set_title('Distribution avec Bornes Outliers')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Avantages méthode IQR** :
- ✅ Robuste aux outliers (ne dépend pas d'eux pour les détecter)
- ✅ Fonctionne avec distributions asymétriques
- ✅ Standard en visualisation (box plots)

**Limites** :
- ❌ Assume que outliers sont dans les queues (pas au centre)
- ❌ Seuil 1.5× IQR est arbitraire (ajustable selon contexte)

**Sources académiques** :
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. - Créateur du box plot et règle IQR

---

### 6. Autres Mesures de Dispersion

#### 6.1 Range (Étendue)

**Définition** :

$$\text{Range} = \max(x) - \min(x)$$

**Avantages** :
- Calcul trivial
- Intuitivement compréhensible

**Inconvénients** :
- **Extrêmement sensible aux outliers** (pire que la moyenne !)
- Ne capture que 2 points sur tout le dataset
- Peu utilisé en pratique statistique sérieuse

```python
data = np.array([10, 12, 14, 15, 16, 18, 20, 22, 100])
range_val = np.max(data) - np.min(data)
print(f"Range : {range_val}")  # 90 (dominé par l'outlier 100)
```

#### 6.2 MAD (Median Absolute Deviation)

**Définition** :

$$MAD = \text{médiane}(|x_i - \text{médiane}(x)|)$$

**Intuition** :
Analogue **robuste** de l'écart-type, basé sur la médiane plutôt que la moyenne.

**Pourquoi utiliser MAD ?**

Pour distributions **très asymétriques ou avec outliers**, où même l'IQR pourrait ne pas suffire.

**Relation avec écart-type** (pour distribution normale) :

$$\sigma \approx 1.4826 \times MAD$$

Le facteur 1.4826 rend MAD comparable à l'écart-type pour données normales.

```python
from scipy.stats import median_abs_deviation

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

# Écart-type (biaisé par outlier)
std = np.std(data, ddof=1)

# MAD (robuste)
mad = median_abs_deviation(data)
mad_normalized = mad * 1.4826  # Pour comparaison avec std

print(f"Écart-type : {std:.2f} (sensible à 100)")
print(f"MAD        : {mad:.2f}")
print(f"MAD (normalisé) : {mad_normalized:.2f} (robuste)")
```

**Usage** :
- Détection d'outliers robuste
- Preprocessing en ML sur données sales
- Alternative à std pour données très contaminées

**Sources** :
- Rousseeuw, P. J., & Croux, C. (1993). "Alternatives to the Median Absolute Deviation". *Journal of the American Statistical Association*.

---

## 💡 Compréhension Intuitive

### Analogie du monde réel

Imaginez deux archers tirant sur une cible :

**Archer A** : Flèches très groupées, mais toutes à 10cm du centre  
→ **Faible variance (précis), mais biaisé**

**Archer B** : Flèches dispersées partout, mais centrées sur le bullseye en moyenne  
→ **Forte variance (imprécis), mais non biaisé**

**Archer C** : Flèches groupées ET centrées  
→ **Faible variance ET non biaisé** ⭐ (objectif idéal)

En Data Science :
- **Variance** = Précision (consistency)
- **Biais** = Exactitude (accuracy)
- **Objectif** : Minimiser les deux !

### Questions pour vérifier la compréhension

1. **Q1** : Si toutes les valeurs d'un dataset sont doublées, comment change la variance ?
   - *Réponse* : Multipliée par $$2^2 = 4$$

2. **Q2** : Dataset A : std=10. Dataset B : std=20. Lequel est plus dispersé ?
   - *Réponse* : Impossible à dire sans connaître les moyennes ! Utiliser CV si échelles différentes.

3. **Q3** : Une distribution a IQR=10 et std=50. Que conclure ?
   - *Réponse* : Présence probable d'outliers (std très élevé vs IQR modéré)

4. **Q4** : Vous devez détecter outliers dans des salaires. Méthode écart-type ou IQR ?
   - *Réponse* : IQR (salaires sont typiquement log-normaux avec outliers)

---

## 💻 Implémentation Pratique

### 1. Fonction Complète de Statistiques Descriptives

```python
"""
Titre : Fonction complète d'analyse de dispersion
Objectif : Calculer toutes les mesures de dispersion + détection outliers
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

def analyse_dispersion_complete(data, nom="Données"):
    """
    Analyse complète de dispersion avec détection outliers.
    
    Args:
        data (array-like): Données numériques
        nom (str): Nom du dataset pour affichage
    
    Returns:
        dict: Toutes les statistiques de dispersion
    """
    data = np.asarray(data)
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        raise ValueError("Toutes les valeurs sont NaN!")
    
    # Tendance centrale
    moyenne = np.mean(data_clean)
    mediane = np.median(data_clean)
    
    # Dispersion absolue
    variance = np.var(data_clean, ddof=1)
    std = np.std(data_clean, ddof=1)
    range_val = np.max(data_clean) - np.min(data_clean)
    mad = median_abs_deviation(data_clean)
    
    # Dispersion relative
    cv = (std / moyenne * 100) if moyenne != 0 else np.nan
    
    # Quantiles
    q1 = np.percentile(data_clean, 25)
    q2 = np.percentile(data_clean, 50)
    q3 = np.percentile(data_clean, 75)
    iqr = q3 - q1
    
    # Détection outliers (IQR)
    borne_inf_iqr = q1 - 1.5 * iqr
    borne_sup_iqr = q3 + 1.5 * iqr
    outliers_iqr = data_clean[(data_clean < borne_inf_iqr) | (data_clean > borne_sup_iqr)]
    
    # Détection outliers (Z-score)
    z_scores = np.abs((data_clean - moyenne) / std)
    outliers_zscore = data_clean[z_scores > 3]
    
    resultats = {
        # Basique
        'n': len(data_clean),
        'n_missing': len(data) - len(data_clean),
        
        # Tendance centrale
        'moyenne': moyenne,
        'mediane': mediane,
        
        # Dispersion absolue
        'variance': variance,
        'std': std,
        'range': range_val,
        'mad': mad,
        'iqr': iqr,
        
        # Dispersion relative
        'cv_pct': cv,
        
        # Quantiles
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'min': np.min(data_clean),
        'max': np.max(data_clean),
        
        # Outliers
        'n_outliers_iqr': len(outliers_iqr),
        'outliers_iqr': outliers_iqr,
        'n_outliers_zscore': len(outliers_zscore),
        'outliers_zscore': outliers_zscore,
        
        # Bornes
        'borne_inf_iqr': borne_inf_iqr,
        'borne_sup_iqr': borne_sup_iqr,
    }
    
    # Affichage formaté
    print(f"\n{'='*70}")
    print(f"ANALYSE DE DISPERSION : {nom}")
    print(f"{'='*70}")
    print(f"\n📊 STATISTIQUES BASIQUES")
    print(f"  Nombre observations : {resultats['n']}")
    print(f"  Valeurs manquantes  : {resultats['n_missing']}")
    
    print(f"\n📍 TENDANCE CENTRALE")
    print(f"  Moyenne   : {resultats['moyenne']:.2f}")
    print(f"  Médiane   : {resultats['mediane']:.2f}")
    
    print(f"\n📏 DISPERSION ABSOLUE")
    print(f"  Variance  : {resultats['variance']:.2f}")
    print(f"  Écart-type: {resultats['std']:.2f}")
    print(f"  Range     : {resultats['range']:.2f}")
    print(f"  IQR       : {resultats['iqr']:.2f}")
    print(f"  MAD       : {resultats['mad']:.2f}")
    
    print(f"\n📈 DISPERSION RELATIVE")
    print(f"  CV        : {resultats['cv_pct']:.1f}%")
    
    print(f"\n🎯 QUANTILES")
    print(f"  Min       : {resultats['min']:.2f}")
    print(f"  Q1 (25%)  : {resultats['q1']:.2f}")
    print(f"  Q2 (50%)  : {resultats['q2']:.2f}")
    print(f"  Q3 (75%)  : {resultats['q3']:.2f}")
    print(f"  Max       : {resultats['max']:.2f}")
    
    print(f"\n⚠️  OUTLIERS")
    print(f"  Méthode IQR    : {resultats['n_outliers_iqr']} outliers ({resultats['n_outliers_iqr']/resultats['n']*100:.1f}%)")
    print(f"  Méthode Z-score: {resultats['n_outliers_zscore']} outliers ({resultats['n_outliers_zscore']/resultats['n']*100:.1f}%)")
    
    if resultats['n_outliers_iqr'] > 0:
        print(f"  Valeurs (IQR)  : {resultats['outliers_iqr'][:5]}{'...' if len(resultats['outliers_iqr']) > 5 else ''}")
    
    print(f"\n{'='*70}\n")
    
    return resultats

# Test
np.random.seed(100)
data_test = np.concatenate([
    np.random.normal(100, 15, 200),
    np.array([20, 25, 180, 190, 200])  # Outliers
])

stats = analyse_dispersion_complete(data_test, "Exemple Dataset")
```

### 2. Visualisation Complète

```python
"""
Titre : Visualisation complète des mesures de dispersion
Objectif : Créer dashboard visuel complet
"""

import matplotlib.pyplot as plt
import seaborn as sns

def visualiser_dispersion(data, nom="Données"):
    """
    Crée dashboard de 6 visualisations pour analyse dispersion.
    """
    stats = analyse_dispersion_complete(data, nom)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histogramme avec statistiques
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(stats['moyenne'], color='red', linestyle='--', linewidth=2, label=f"Moyenne={stats['moyenne']:.1f}")
    ax1.axvline(stats['mediane'], color='blue', linestyle='--', linewidth=2, label=f"Médiane={stats['mediane']:.1f}")
    ax1.axvline(stats['moyenne'] - stats['std'], color='green', linestyle=':', alpha=0.7)
    ax1.axvline(stats['moyenne'] + stats['std'], color='green', linestyle=':', alpha=0.7, label=f"±1σ")
    ax1.set_xlabel('Valeur')
    ax1.set_ylabel('Fréquence')
    ax1.set_title(f'Distribution : {nom}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Box plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.boxplot(data, vert=True)
    ax2.set_ylabel('Valeur')
    ax2.set_title('Box Plot')
    ax2.grid(alpha=0.3)
    
    # 3. Violin plot
    ax3 = fig.add_subplot(gs[1, 0])
    parts = ax3.violinplot([data], vert=True, showmeans=True, showmedians=True)
    ax3.set_ylabel('Valeur')
    ax3.set_title('Violin Plot')
    ax3.grid(alpha=0.3)
    
    # 4. QQ plot (normalité)
    ax4 = fig.add_subplot(gs[1, 1])
    from scipy import stats as scipy_stats
    scipy_stats.probplot(data, dist="norm", plot=ax4)
    ax4.set_title('QQ Plot (Test Normalité)')
    ax4.grid(alpha=0.3)
    
    # 5. Tableau statistiques
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    table_data = [
        ['Statistique', 'Valeur'],
        ['─────────────', '──────'],
        ['n', f"{stats['n']}"],
        ['Moyenne', f"{stats['moyenne']:.2f}"],
        ['Médiane', f"{stats['mediane']:.2f}"],
        ['Std', f"{stats['std']:.2f}"],
        ['Variance', f"{stats['variance']:.2f}"],
        ['CV', f"{stats['cv_pct']:.1f}%"],
        ['IQR', f"{stats['iqr']:.2f}"],
        ['Range', f"{stats['range']:.2f}"],
        ['Outliers (IQR)', f"{stats['n_outliers_iqr']}"],
    ]
    table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax5.set_title('Résumé Statistique')
    
    # 6. Scatter plot avec outliers
    ax6 = fig.add_subplot(gs[2, :])
    indices = np.arange(len(data))
    outliers_mask_iqr = (data < stats['borne_inf_iqr']) | (data > stats['borne_sup_iqr'])
    
    ax6.scatter(indices[~outliers_mask_iqr], data[~outliers_mask_iqr], 
                alpha=0.6, s=20, color='blue', label='Données normales')
    ax6.scatter(indices[outliers_mask_iqr], data[outliers_mask_iqr], 
                alpha=0.9, s=50, color='red', marker='x', label='Outliers (IQR)')
    ax6.axhline(stats['moyenne'], color='green', linestyle='--', alpha=0.5, label='Moyenne')
    ax6.axhline(stats['borne_sup_iqr'], color='red', linestyle='--', alpha=0.5, label='Bornes IQR')
    ax6.axhline(stats['borne_inf_iqr'], color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Index')
    ax6.set_ylabel('Valeur')
    ax6.set_title('Séquence des Données avec Outliers')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.suptitle(f'Dashboard Complet : {nom}', fontsize=16, fontweight='bold')
    plt.show()

# Test
visualiser_dispersion(data_test, "Exemple Dataset avec Outliers")
```

---

## 🔬 Exemples Concrets et Cas d'Usage

### Exemple 1 : Contrôle Qualité Industriel - Niveau Débutant

**Contexte** :
Une usine produit des boulons de 10.0mm de diamètre. La spécification tolère ±0.2mm (9.8mm à 10.2mm). Vous analysez un échantillon de 100 boulons.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(321)

# Machine A : Bien calibrée
machine_A = np.random.normal(loc=10.0, scale=0.05, size=100)

# Machine B : Moins précise
machine_B = np.random.normal(loc=10.0, scale=0.12, size=100)

# Spécifications
spec_inf = 9.8
spec_sup = 10.2

# Analyse Machine A
stats_A = analyse_dispersion_complete(machine_A, "Machine A")
defauts_A = np.sum((machine_A < spec_inf) | (machine_A > spec_sup))
taux_defaut_A = defauts_A / len(machine_A) * 100

# Analyse Machine B
stats_B = analyse_dispersion_complete(machine_B, "Machine B")
defauts_B = np.sum((machine_B < spec_inf) | (machine_B > spec_sup))
taux_defaut_B = defauts_B / len(machine_B) * 100

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (data, nom, stats) in enumerate([(machine_A, "Machine A", stats_A),
                                            (machine_B, "Machine B", stats_B)]):
    axes[idx].hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
    axes[idx].axvline(10.0, color='green', linestyle='-', linewidth=2, label='Cible (10.0mm)')
    axes[idx].axvline(spec_inf, color='red', linestyle='--', linewidth=2, label='Limites Spec')
    axes[idx].axvline(spec_sup, color='red', linestyle='--', linewidth=2)
    axes[idx].axvline(stats['moyenne'] - stats['std'], color='orange', linestyle=':', alpha=0.7)
    axes[idx].axvline(stats['moyenne'] + stats['std'], color='orange', linestyle=':', alpha=0.7, label=f'±1σ = {stats["std"]:.3f}')
    axes[idx].set_xlabel('Diamètre (mm)')
    axes[idx].set_ylabel('Fréquence')
    axes[idx].set_title(f'{nom} - σ={stats["std"]:.3f}mm')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Rapport qualité
print("\n" + "="*70)
print("RAPPORT CONTRÔLE QUALITÉ")
print("="*70)
print(f"\nSpécifications : {spec_inf}mm à {spec_sup}mm (tolérance ±0.2mm)")
print(f"\n{'Machine':<15} | {'σ (mm)':>10} | {'Défauts':>10} | {'Taux':>10}")
print("-"*70)
print(f"{'Machine A':<15} | {stats_A['std']:>10.4f} | {defauts_A:>10} | {taux_defaut_A:>9.1f}%")
print(f"{'Machine B':<15} | {stats_B['std']:>10.4f} | {defauts_B:>10} | {taux_defaut_B:>9.1f}%")

print("\n💡 CONCLUSION :")
print(f"Machine A (σ={stats_A['std']:.4f}) : ✅ Processus capable (faible dispersion)")
print(f"Machine B (σ={stats_B['std']:.4f}) : ⚠️ Processus non capable (forte dispersion)")
print(f"\n📊 Indice de capabilité Cp (simplifié) :")
print(f"  Cp_A = {(spec_sup - spec_inf) / (6 * stats_A['std']):.2f} (>1.33 = Excellent)")
print(f"  Cp_B = {(spec_sup - spec_inf) / (6 * stats_B['std']):.2f} (<1 = Inacceptable)")
```

**Interprétation** :
- **Écart-type faible** → Processus stable et prévisible
- **Cp > 1.33** = Standard "Six Sigma" (3.4 défauts par million)
- En production, on surveille en continu l'écart-type pour détecter dérives

---

### Exemple 2 : Comparaison Performance Modèles ML - Niveau Intermédiaire

**Contexte** :
Vous comparez 3 modèles de régression sur leurs erreurs de prédiction. Lequel est le plus stable ?

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(444)

# Simulation erreurs de 3 modèles sur 1000 prédictions
modeles = {
    'Linear Regression': np.random.normal(0, 20, 1000),
    'Random Forest': np.random.normal(0, 15, 1000),
    'XGBoost': np.concatenate([
        np.random.normal(0, 10, 950),
        np.random.uniform(-80, 80, 50)  # Quelques prédictions très mauvaises
    ])
}

# DataFrame pour analyse
df_erreurs = pd.DataFrame(modeles)

# Statistiques comparatives
print("\nCOMPARAISON STABILITÉ DES MODÈLES")
print("="*80)
print(f"{'Modèle':<20} | {'MAE':>8} | {'Std':>8} | {'IQR':>8} | {'CV%':>8} | {'Outliers':>10}")
print("-"*80)

for nom, erreurs in modeles.items():
    mae = np.mean(np.abs(erreurs))
    std = np.std(erreurs, ddof=1)
    iqr = np.percentile(erreurs, 75) - np.percentile(erreurs, 25)
    cv = (std / np.abs(np.mean(erreurs)) * 100) if np.mean(erreurs) != 0 else np.inf
    
    q1, q3 = np.percentile(erreurs, [25, 75])
    borne_inf, borne_sup = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_outliers = np.sum((erreurs < borne_inf) | (erreurs > borne_sup))
    
    print(f"{nom:<20} | {mae:>8.2f} | {std:>8.2f} | {iqr:>8.2f} | {cv:>8.1f} | {n_outliers:>10}")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plots comparatifs
axes[0, 0].boxplot([modeles[k] for k in modeles.keys()], labels=modeles.keys())
axes[0, 0].set_ylabel('Erreur de Prédiction')
axes[0, 0].set_title('Box Plots Comparatifs')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=15)

# Violin plots
positions = range(len(modeles))
for idx, (nom, erreurs) in enumerate(modeles.items()):
    parts = axes[0, 1].violinplot([erreurs], positions=[idx], showmeans=True, showmedians=True)
axes[0, 1].set_xticks(positions)
axes[0, 1].set_xticklabels(modeles.keys(), rotation=15)
axes[0, 1].set_ylabel('Erreur de Prédiction')
axes[0, 1].set_title('Violin Plots (densité + box plot)')
axes[0, 1].grid(alpha=0.3)

# Histogrammes superposés
for nom, erreurs in modeles.items():
    axes[1, 0].hist(erreurs, bins=50, alpha=0.5, label=nom, edgecolor='black')
axes[1, 0].set_xlabel('Erreur')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Distributions des Erreurs')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Écart-types comparatifs
stds = [np.std(modeles[k], ddof=1) for k in modeles.keys()]
axes[1, 1].bar(modeles.keys(), stds, color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Écart-type')
axes[1, 1].set_title('Stabilité (plus bas = mieux)')
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 RECOMMANDATION :")
print("Random Forest : ✅ Meilleur compromis (MAE faible + dispersion faible)")
print("XGBoost       : ⚠️ Quelques prédictions très mauvaises (outliers) → Investiguer")
```

**Interprétation** :
- **MAE** : Erreur moyenne (tendance centrale)
- **Std/IQR** : Stabilité (dispersion)
- **Outliers** : Prédictions catastrophiques ponctuelles
- **Choix final** : Minimiser **à la fois** erreur ET variance (biais-variance tradeoff)

---

### Exemple 3 : Analyse de Portefeuille Financier - Niveau Avancé

**Contexte** :
Vous comparez 3 actifs financiers sur leur rendement et risque (volatilité = écart-type).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(555)

# Simulation rendements quotidiens (%)
n_jours = 252  # 1 année de trading

actifs = {
    'Action Tech (risquée)': np.random.normal(0.08, 2.5, n_jours),  # Rendement moyen 0.08%, volatilité 2.5%
    'Obligation (stable)': np.random.normal(0.03, 0.5, n_jours),    # Rendement moyen 0.03%, volatilité 0.5%
    'Action Blue-Chip': np.random.normal(0.05, 1.2, n_jours)        # Rendement moyen 0.05%, volatilité 1.2%
}

df_rendements = pd.DataFrame(actifs)

# Calcul statistiques annualisées
print("\nANALYSE RISQUE-RENDEMENT (252 jours de trading)")
print("="*90)
print(f"{'Actif':<25} | {'Rend. Moy':>12} | {'Volatilité':>12} | {'Sharpe':>10} | {'VaR 95%':>12}")
print("-"*90)

taux_sans_risque = 0.02  # 2% annuel → ~0.008% par jour

for nom, rendements in actifs.items():
    rend_moyen_jour = np.mean(rendements)
    rend_annuel = rend_moyen_jour * 252  # Annualisation simple
    
    vol_jour = np.std(rendements, ddof=1)
    vol_annuelle = vol_jour * np.sqrt(252)  # Annualisation volatilité
    
    # Ratio de Sharpe = (rendement - taux sans risque) / volatilité
    sharpe = (rend_annuel - taux_sans_risque) / vol_annuelle if vol_annuelle != 0 else 0
    
    # Value at Risk 95% = percentile 5% (perte maximale dans 95% des cas)
    var_95 = np.percentile(rendements, 5)
    
    print(f"{nom:<25} | {rend_annuel:>11.2f}% | {vol_annuelle:>11.2f}% | {sharpe:>10.2f} | {var_95:>11.2f}%")

# Visualisation risque-rendement
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Séries temporelles
for nom, rendements in actifs.items():
    valeur_portefeuille = 100 * (1 + rendements/100).cumprod()  # Valeur initiale 100
    axes[0, 0].plot(valeur_portefeuille, label=nom, alpha=0.8)
axes[0, 0].set_xlabel('Jours')
axes[0, 0].set_ylabel('Valeur Portefeuille')
axes[0, 0].set_title('Évolution Temporelle (départ 100)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Distributions rendements
for nom, rendements in actifs.items():
    axes[0, 1].hist(rendements, bins=30, alpha=0.5, label=nom, edgecolor='black')
axes[0, 1].set_xlabel('Rendement Quotidien (%)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution des Rendements')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Diagramme Risque-Rendement
rends_annuels = [np.mean(actifs[k]) * 252 for k in actifs.keys()]
vols_annuelles = [np.std(actifs[k], ddof=1) * np.sqrt(252) for k in actifs.keys()]

axes[1, 0].scatter(vols_annuelles, rends_annuels, s=200, alpha=0.7, c=['red', 'blue', 'green'])
for idx, nom in enumerate(actifs.keys()):
    axes[1, 0].annotate(nom, (vols_annuelles[idx], rends_annuels[idx]), 
                        fontsize=8, ha='center', va='bottom')
axes[1, 0].set_xlabel('Risque (Volatilité Annuelle %)')
axes[1, 0].set_ylabel('Rendement Annuel (%)')
axes[1, 0].set_title('Espace Risque-Rendement')
axes[1, 0].grid(alpha=0.3)

# 4. Box plots
axes[1, 1].boxplot([actifs[k] for k in actifs.keys()], labels=actifs.keys())
axes[1, 1].set_ylabel('Rendement Quotidien (%)')
axes[1, 1].set_title('Box Plots Comparatifs')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

print("\n💡 INTERPRÉTATION FINANCIÈRE :")
print("- Volatilité (σ) = RISQUE")
print("- Rendement moyen = GAIN attendu")
print("- Ratio de Sharpe = Gain ajusté du risque (plus élevé = mieux)")
print("- VaR 95% = Perte maximale attendue dans 95% des cas")
print("\n✅ Choisir selon profil investisseur :")
print("  - Prudent : Obligation (faible σ)")
print("  - Équilibré : Blue-Chip (σ modéré, Sharpe bon)")
print("  - Agressif : Tech (σ élevé, rendement élevé)")
```

**Analyse critique** :
- **Écart-type = mesure standard du risque** en finance
- **Ratio de Sharpe** : Maximiser rendement PAR UNITÉ de risque
- **VaR** : Quantile 5% = "Dans 95% des cas, je ne perdrai pas plus que X%"
- **Diversification** : Combiner actifs pour réduire σ du portefeuille global

---

## ⚖️ Comparaisons et Choix de Design

### Écart-type vs IQR : Quelle mesure de dispersion utiliser ?

| Critère | Écart-type (σ) | IQR |
|---------|----------------|-----|
| **Sensibilité outliers** | ❌ Très sensible | ✅ Robuste |
| **Distribution** | Optimal pour normale | Fonctionne pour toutes |
| **Interprétation** | Règle 68-95-99.7 (si normale) | 50% centraux des données |
| **Usage ML** | Normalisation (Z-score) | Détection outliers |
| **Finance** | Volatilité, VaR paramétrique | VaR non paramétrique |
| **Visualisation** | Barre d'erreur | Box plot |
| **Calcul** | Requiert moyenne | Requiert médiane |

✅ **Utiliser Écart-type quand** :
- Distribution approximativement normale
- Peu ou pas d'outliers
- Modélisation théorique (régression, tests statistiques)
- Standardisation de features pour ML

✅ **Utiliser IQR quand** :
- Distribution asymétrique
- Présence d'outliers
- Détection d'anomalies
- Communication robuste (moins sensible aux extrêmes)

**Exemple comparatif** :

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

distributions = [
    ("Normale (σ approprié)", np.random.normal(50, 10, 1000)),
    ("Log-Normale (σ biaisé)", np.random.lognormal(3, 0.8, 1000)),
    ("Avec Outliers (σ biaisé)", np.concatenate([
        np.random.normal(50, 10, 990),
        np.array([200, 220, 250, 280, 300, 320, 350, 400, 450, 500])
    ]))
]

for idx, (titre, data) in enumerate(distributions):
    std = np.std(data, ddof=1)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    mediane = np.median(data)
    moyenne = np.mean(data)
    
    axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
    
    # Écart-type (barre rouge)
    axes[idx].axvline(moyenne - std, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[idx].axvline(moyenne + std, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'±1σ = {std:.1f}')
    
    # IQR (barre bleue)
    axes[idx].axvline(q1, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    axes[idx].axvline(q3, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'IQR = {iqr:.1f}')
    axes[idx].axvline(mediane, color='green', linestyle='-', linewidth=2, alpha=0.5)
    
    axes[idx].set_title(titre)
    axes[idx].set_xlabel('Valeur')
    axes[idx].set_ylabel('Fréquence')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 RECOMMANDATION :")
print("Distribution Normale     : σ et IQR donnent info similaire → Utiliser σ (plus standard)")
print("Distribution Asymétrique : IQR plus représentatif que σ")
print("Avec Outliers            : IQR inchangé, σ explosé → OBLIGATOIRE d'utiliser IQR")
```

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Oublier `ddof=1` pour Variance Échantillon

```python
# ❌ MAUVAIS
data = np.array([1, 2, 3, 4, 5])
var_fausse = np.var(data)  # ddof=0 par défaut → variance POPULATION
print(f"Variance (fausse) : {var_fausse}")  # 2.0

# ✅ BON
var_correcte = np.var(data, ddof=1)  # ddof=1 → variance ÉCHANTILLON
print(f"Variance (correcte) : {var_correcte}")  # 2.5

# Pourquoi c'est mieux : En pratique, on travaille presque toujours avec des ÉCHANTILLONS
```

**Impact** : Sous-estimation systématique de la variance, tests statistiques invalides.

---

### ❌ Erreur 2 : Comparer Écart-types d'Échelles Différentes

```python
# ❌ MAUVAIS
poids_souris_g = np.array([20, 22, 21, 23, 24, 25])
poids_elephants_kg = np.array([4800, 5000, 5200, 5100, 4900])

std_souris = np.std(poids_souris_g, ddof=1)
std_elephants = np.std(poids_elephants_kg, ddof=1)

print(f"Écart-type souris : {std_souris:.2f}g")
print(f"Écart-type éléphants : {std_elephants:.2f}kg")
print("❌ IMPOSSIBLE de comparer directement 1.87g vs 146kg !")

# ✅ BON - Utiliser Coefficient de Variation
cv_souris = (std_souris / np.mean(poids_souris_g)) * 100
cv_elephants = (std_elephants / np.mean(poids_elephants_kg)) * 100

print(f"\nCV souris : {cv_souris:.1f}%")
print(f"CV éléphants : {cv_elephants:.1f}%")
print("✅ Comparable : Les souris sont plus variables relativement à leur poids")
```

---

### ✅ Bonne Pratique 1 : Toujours Visualiser avec Box Plot

```python
def analyse_rapide(data, nom="Données"):
    """Quick check : statistiques + box plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Annotations
    q1, mediane, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    
    ax.text(1.15, mediane, f'Médiane: {mediane:.2f}', va='center')
    ax.text(1.15, q1, f'Q1: {q1:.2f}', va='center')
    ax.text(1.15, q3, f'Q3: {q3:.2f}', va='center')
    ax.text(1.15, q3 + 0.5*iqr, f'IQR: {iqr:.2f}', va='center', fontweight='bold')
    
    ax.set_ylabel('Valeur')
    ax.set_title(f'Box Plot : {nom}')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stats
    print(f"\n{nom}")
    print(f"  Moyenne : {np.mean(data):.2f}")
    print(f"  Médiane : {mediane:.2f}")
    print(f"  Std     : {np.std(data, ddof=1):.2f}")
    print(f"  IQR     : {iqr:.2f}")

# Test
test_data = np.concatenate([np.random.normal(100, 15, 95), np.array([200, 220, 250, 280, 300])])
analyse_rapide(test_data, "Données avec Outliers")
```

---

### 📋 Checklist de Validation

Avant de communiquer des mesures de dispersion :

- [ ] **Visualisation** : Box plot + histogramme créés ?
- [ ] **ddof** : Utilisé `ddof=1` pour variance échantillon ?
- [ ] **Échelle** : Si comparaison multi-échelles → CV calculé ?
- [ ] **Outliers** : Détectés avec IQR ou Z-score ?
- [ ] **Distribution** : Normale ou asymétrique ? (influence choix σ vs IQR)
- [ ] **Contexte** : Mesure choisie appropriée au domaine ?
  - Finance → σ (volatilité)
  - QC industriel → σ (capabilité processus)
  - Données sales → IQR (robustesse)
- [ ] **Communication** : Interlocuteur comprend-il σ vs IQR ?

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques

1. **"Robust Statistics: The Approach Based on Influence Functions"**
   - **Auteurs** : Hampel, Ronchetti, Rousseeuw, Stahel (1986)
   - **Contribution** : Fondements théoriques de la robustesse (MAD, IQR)

2. **"Exploratory Data Analysis"**
   - **Auteur** : John W. Tukey (1977)
   - **Contribution** : Invention du box plot, règle IQR outliers

3. **"Six Sigma and Process Capability"**
   - **Référence industrielle** : Contrôle qualité basé sur écart-type

---

### 📚 Ressources Complémentaires

- **StatQuest (YouTube)** - Josh Starmer : Variance, Std, IQR
- **[Seeing Theory](https://seeing-theory.brown.edu/)** - Visualisations interactives
- **Documentation NumPy/SciPy** : Fonctions statistiques

---

### 📖 Cours Connexes

**Suite directe** :
- [[data_visualization_principles]] - Box plots, violin plots en détail
- [[distribution_analysis]] - Skewness, kurtosis, formes
- [[normality_tests]] - Tester si distribution normale (Shapiro-Wilk)

**Applications** :
- [[confidence_intervals]] - IC basés sur écart-type
- [[hypothesis_testing]] - Tests basés sur variance
- [[sample_size_calculation]] - Rôle de σ dans calcul taille échantillon

---

## 📝 Résumé Rapide

### Formules Clés

| Mesure | Formule | Quand Utiliser |
|--------|---------|----------------|
| **Variance** | $$s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$$ | Théorie, calculs intermédiaires |
| **Écart-type** | $$s = \sqrt{s^2}$$ | Standard, même unité que données |
| **CV** | $$CV = \frac{s}{\bar{x}} \times 100\%$$ | Comparer échelles différentes |
| **IQR** | $$IQR = Q3 - Q1$$ | Robuste, distributions asymétriques |
| **MAD** | $$MAD = \text{médiane}(|x_i - \text{med}|)$$ | Très robuste, données contaminées |

### Code Minimal

```python
import numpy as np

data = np.array([...])

# Dispersion absolue
std = np.std(data, ddof=1)  # ⚠️ ddof=1 obligatoire !
var = np.var(data, ddof=1)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

# Dispersion relative
cv = (std / np.mean(data)) * 100

# Outliers (IQR)
q1, q3 = np.percentile(data, [25, 75])
outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
```

---

## 🔗 Intégration Repository

**Mise à jour INDEX** :

```markdown
| **Variance, Écart-type** | Mesurer la dispersion | [[measures_dispersion]] ✅ | ⭐🔥 |
```

**Prochaine étape** : [[data_visualization_principles]] 📊

---

**Cours créé le 2026-03-18** ✅
