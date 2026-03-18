# 📊 Principes de Visualisation de Données Statistiques

> **Résumé en une phrase** : La visualisation transforme des nombres en insights visuels, permettant de détecter patterns, outliers et distributions que les statistiques seules ne révèlent pas.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-18 |
| **Dernière mise à jour** | 2026-03-18 |
| **Domaine** | Statistiques Descriptives / Data Visualization |
| **Niveau** | Débutant |
| **Durée de lecture** | ~40 minutes |
| **Fichier** | `data_visualization_principles.md` |
| **Emplacement** | `/00_statistics_foundations/01_descriptive_statistics/` |
| **Tags** | `#statistics` `#visualization` `#eda` `#matplotlib` `#seaborn` `#charts` `#best-practices` |

### Prérequis

- [x] [[measures_central_tendency]] - Moyenne, médiane, mode
- [x] [[measures_dispersion]] - Variance, écart-type, quartiles
- [ ] Bases Python (matplotlib, seaborn - optionnel)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[measures_central_tendency]] - Mesures de tendance centrale
  - [[measures_dispersion]] - Mesures de dispersion
- **Complémentaires** : 
  - [[distribution_analysis]] - Skewness, kurtosis, formes
  - [[correlation_covariance]] - Visualiser relations bivariées
- **Suite recommandée** : 
  - [[probability_foundations]] - Module 2 : Théorie des probabilités
  - [[normality_tests]] - Vérifier normalité visuellement et statistiquement

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

*"Un graphique vaut mille statistiques."* Ce cours vous enseigne les **principes fondamentaux** de la visualisation statistique : **quoi** visualiser (distributions, comparaisons, relations), **comment** le faire correctement (choix du graphique, design efficace), et surtout **pourquoi** la visualisation est la première étape de toute analyse. Vous maîtriserez les visualisations essentielles (histogrammes, box plots, scatter plots) et éviterez les pièges fréquents qui mènent à des graphiques trompeurs.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Expliquer pourquoi visualiser avant toute analyse statistique (Quartet d'Anscombe)
2. **Appliquer** : Créer histogrammes, box plots, scatter plots avec Matplotlib/Seaborn
3. **Analyser** : Identifier distribution, outliers, asymétrie visuellement
4. **Évaluer** : Choisir le type de graphique approprié selon le type de données et la question posée
5. **Créer** : Produire visualisations publication-ready suivant les bonnes pratiques
6. **Critiquer** : Détecter visualisations trompeuses (axes manipulés, échelles inappropriées)

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

**Le Quartet d'Anscombe (1973)** est la démonstration ultime de l'importance de la visualisation :

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Les 4 datasets d'Anscombe (statistiques IDENTIQUES)
anscombe_data = {
    'I': {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
          'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]},
    'II': {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
           'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]},
    'III': {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]},
    'IV': {'x': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]}
}

# Calcul des statistiques pour chaque dataset
print("QUARTET D'ANSCOMBE - Statistiques Identiques !")
print("="*70)
print(f"{'Dataset':<10} | {'Moy(X)':>8} | {'Moy(Y)':>8} | {'Std(X)':>8} | {'Std(Y)':>8} | {'Corrélation':>12} | {'Régression':>15}")
print("-"*70)

for nom, data in anscombe_data.items():
    x = np.array(data['x'])
    y = np.array(data['y'])
    
    # Statistiques
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    correlation = np.corrcoef(x, y)[0, 1]
    
    # Régression linéaire y = a*x + b
    a, b = np.polyfit(x, y, 1)
    
    print(f"{nom:<10} | {mean_x:>8.2f} | {mean_y:>8.2f} | {std_x:>8.2f} | {std_y:>8.2f} | {correlation:>12.3f} | y={a:.2f}x+{b:.2f}")

print("\n💡 TOUTES les statistiques sont quasi-identiques !")
print("   Mais regardons les graphiques...")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (nom, data) in enumerate(anscombe_data.items()):
    x = np.array(data['x'])
    y = np.array(data['y'])
    
    # Scatter plot
    axes[idx].scatter(x, y, s=80, alpha=0.7, edgecolor='black')
    
    # Ligne de régression
    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b
    axes[idx].plot(x_line, y_line, 'r--', linewidth=2, label=f'y={a:.2f}x+{b:.2f}')
    
    axes[idx].set_xlabel('X', fontsize=12)
    axes[idx].set_ylabel('Y', fontsize=12)
    axes[idx].set_title(f'Dataset {nom}', fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
    axes[idx].set_xlim(2, 20)
    axes[idx].set_ylim(2, 14)

plt.suptitle('Quartet d\'Anscombe : Mêmes stats, distributions DIFFÉRENTES !', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Résultat** : Les 4 datasets ont :
- Même moyenne de X et Y
- Même écart-type de X et Y
- Même corrélation
- Même équation de régression linéaire

**MAIS** des distributions complètement différentes :
- **Dataset I** : Relation linéaire normale
- **Dataset II** : Relation curvilinéaire (non linéaire)
- **Dataset III** : Relation linéaire avec 1 outlier
- **Dataset IV** : Pas de relation, juste 1 point extrême créant une fausse corrélation

**Conclusion** : **Ne JAMAIS faire confiance aux statistiques seules sans visualiser !**

### Quel problème résout-il ?

**Problème** : Vous analysez les ventes mensuelles de 2 produits. Un collègue affirme : *"Les deux ont la même moyenne de ventes (1000 unités/mois) et le même écart-type (150 unités), donc performances identiques."*

**Vous visualisez** et découvrez :
- **Produit A** : Ventes stables autour de 1000 (distribution normale)
- **Produit B** : Alternance 500 unités / 1500 unités (distribution bimodale)

**Insight** : Produit B a un pattern saisonnier caché que les statistiques seules ne révèlent pas !

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Produit A : Ventes stables
mois = np.arange(1, 25)
ventes_A = np.random.normal(1000, 150, 24)

# Produit B : Alternance (pattern saisonnier)
ventes_B = np.where(mois % 2 == 0, 
                    np.random.normal(1500, 100, 24),  # Mois pairs : haute saison
                    np.random.normal(500, 100, 24))   # Mois impairs : basse saison

print(f"Produit A - Moyenne: {np.mean(ventes_A):.0f}, Std: {np.std(ventes_A, ddof=1):.0f}")
print(f"Produit B - Moyenne: {np.mean(ventes_B):.0f}, Std: {np.std(ventes_B, ddof=1):.0f}")
print("\n⚠️ Statistiques similaires, mais...")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Série temporelle
axes[0].plot(mois, ventes_A, 'o-', label='Produit A', linewidth=2, markersize=8, alpha=0.7)
axes[0].plot(mois, ventes_B, 's-', label='Produit B', linewidth=2, markersize=8, alpha=0.7)
axes[0].axhline(1000, color='gray', linestyle='--', alpha=0.5, label='Moyenne commune')
axes[0].set_xlabel('Mois')
axes[0].set_ylabel('Ventes (unités)')
axes[0].set_title('Séries Temporelles')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Distributions
axes[1].hist(ventes_A, bins=15, alpha=0.7, label='Produit A', edgecolor='black')
axes[1].hist(ventes_B, bins=15, alpha=0.7, label='Produit B', edgecolor='black')
axes[1].axvline(np.mean(ventes_A), color='blue', linestyle='--', linewidth=2)
axes[1].axvline(np.mean(ventes_B), color='orange', linestyle='--', linewidth=2)
axes[1].set_xlabel('Ventes (unités)')
axes[1].set_ylabel('Fréquence')
axes[1].set_title('Distributions')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Le graphique révèle : Produit B a un pattern saisonnier fort !")
print("   → Décision : Adapter stocks selon cycles pour Produit B")
```

### Applications dans le monde réel

1. **Analyse Exploratoire de Données (EDA)** :
   - Première étape de TOUT projet Data Science
   - Détecter patterns, outliers, distributions avant modélisation
   - Validation hypothèses (normalité, linéarité, homoscédasticité)

2. **Communication de Résultats** :
   - Rapports pour management non-technique
   - Dashboards business (Tableau, Power BI)
   - Publications scientifiques (figures)

3. **Debugging ML** :
   - Analyser résidus de régression (QQ-plots, scatter plots)
   - Visualiser frontières de décision (classification)
   - Feature importance (bar charts)

4. **Monitoring Production** :
   - Séries temporelles (métriques serveur, ventes)
   - Alerting visuel (déviations, anomalies)
   - A/B testing (distributions groupes A vs B)

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous structurons les visualisations selon le **type de données** (1 variable, 2 variables, etc.) et le **type de question** (distribution ? comparaison ? relation ?).

### 1. Taxonomie des Visualisations Statistiques

#### 1.1 Classification par Type de Données

**1 Variable (Univariée)** :
- **Numérique continue** : Histogramme, Box plot, Density plot, Violin plot
- **Catégorielle** : Bar chart, Pie chart (déconseillé)

**2 Variables (Bivariée)** :
- **Numérique × Numérique** : Scatter plot, Line plot (si temporel), Hexbin (si dense)
- **Numérique × Catégorielle** : Box plot groupé, Violin plot groupé, Strip plot
- **Catégorielle × Catégorielle** : Heatmap (table de contingence), Stacked bar chart

**3+ Variables (Multivariée)** :
- Scatter plot avec couleur/taille (3-4 variables)
- Pair plot / Scatter matrix
- Parallel coordinates
- Heatmap de corrélation

#### 1.2 Classification par Question Posée

| Question | Type de Viz Recommandé |
|----------|------------------------|
| **Quelle est la distribution ?** | Histogramme, Box plot, Violin plot |
| **Y a-t-il des outliers ?** | Box plot, Scatter plot |
| **Comparer 2+ groupes** | Box plot groupé, Violin plot, Strip plot |
| **Relation entre 2 variables ?** | Scatter plot, Line plot |
| **Évolution temporelle ?** | Line plot, Area chart |
| **Composition d'un tout ?** | Stacked bar, Treemap (PAS pie chart !) |
| **Distributions multiples ?** | Facet grid, Small multiples |

---

### 2. Histogrammes

#### 2.1 Définition et Construction

**Histogramme** : Graphique en barres représentant la **distribution de fréquences** d'une variable numérique continue.

**Construction** :
1. Diviser le range des données en **bins** (intervalles de même largeur)
2. Compter le nombre d'observations dans chaque bin
3. Dessiner barres dont la hauteur = fréquence (ou densité)

**Formule pour nombre de bins optimal** :

**Règle de Sturges** :

$$k = \lceil \log_2(n) + 1 \rceil$$

**Règle de Freedman-Diaconis** (plus robuste) :

$$\text{largeur bin} = 2 \times \frac{IQR}{n^{1/3}}$$

où $$IQR$$ = intervalle interquartile, $$n$$ = nombre d'observations.

**Pourquoi ces formules ?**

- **Sturges** : Basée sur distribution normale, simple
- **Freedman-Diaconis** : Robuste aux outliers, adaptative à la dispersion

#### 2.2 Implémentation et Variations

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(456)

# Génération données (distribution normale)
data = np.random.normal(100, 15, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Trop peu de bins (perte d'information)
axes[0, 0].hist(data, bins=5, edgecolor='black', alpha=0.7, color='lightcoral')
axes[0, 0].set_title('5 bins (Trop peu - Sous-lissage)')
axes[0, 0].set_xlabel('Valeur')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].grid(alpha=0.3)

# 2. Nombre optimal (Sturges)
n = len(data)
bins_sturges = int(np.ceil(np.log2(n) + 1))
axes[0, 1].hist(data, bins=bins_sturges, edgecolor='black', alpha=0.7, color='lightgreen')
axes[0, 1].set_title(f'Sturges: {bins_sturges} bins (Optimal)')
axes[0, 1].set_xlabel('Valeur')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].grid(alpha=0.3)

# 3. Trop de bins (bruit)
axes[0, 2].hist(data, bins=100, edgecolor='black', alpha=0.7, color='lightyellow')
axes[0, 2].set_title('100 bins (Trop - Sur-détail)')
axes[0, 2].set_xlabel('Valeur')
axes[0, 2].set_ylabel('Fréquence')
axes[0, 2].grid(alpha=0.3)

# 4. Fréquence vs Densité
axes[1, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue', density=False)
axes[1, 0].set_title('Fréquence (Count)')
axes[1, 0].set_xlabel('Valeur')
axes[1, 0].set_ylabel('Nombre d\'observations')
axes[1, 0].grid(alpha=0.3)

# 5. Densité normalisée (aire = 1)
axes[1, 1].hist(data, bins=30, edgecolor='black', alpha=0.7, color='lightblue', density=True)
axes[1, 1].set_title('Densité (Aire totale = 1)')
axes[1, 1].set_xlabel('Valeur')
axes[1, 1].set_ylabel('Densité')
axes[1, 1].grid(alpha=0.3)

# 6. Avec courbe de densité (KDE)
from scipy.stats import gaussian_kde
axes[1, 2].hist(data, bins=30, edgecolor='black', alpha=0.5, color='lightblue', density=True, label='Histogramme')
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 1000)
axes[1, 2].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE (Densité lissée)')
axes[1, 2].set_title('Histogramme + KDE')
axes[1, 2].set_xlabel('Valeur')
axes[1, 2].set_ylabel('Densité')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 2.3 Interprétation et Patterns

**Formes courantes** :

1. **Symétrique (Normale)** : En forme de cloche, moyenne ≈ médiane
2. **Asymétrique droite** : Queue longue à droite (ex: revenus, prix)
3. **Asymétrique gauche** : Queue longue à gauche
4. **Bimodale** : 2 pics → 2 sous-populations
5. **Uniforme** : Toutes valeurs aussi fréquentes

```python
# Génération distributions typiques
np.random.seed(789)

distributions = {
    'Normale': np.random.normal(50, 10, 1000),
    'Log-Normale (Asymétrie droite)': np.random.lognormal(3, 0.5, 1000),
    'Bimodale': np.concatenate([np.random.normal(30, 5, 500), 
                                np.random.normal(70, 5, 500)]),
    'Uniforme': np.random.uniform(0, 100, 1000),
    'Exponentielle': np.random.exponential(20, 1000),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (nom, data) in enumerate(distributions.items()):
    if idx < len(axes):
        axes[idx].hist(data, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Moy={np.mean(data):.1f}')
        axes[idx].axvline(np.median(data), color='blue', linestyle='--', linewidth=2, label=f'Méd={np.median(data):.1f}')
        axes[idx].set_title(nom, fontweight='bold')
        axes[idx].set_xlabel('Valeur')
        axes[idx].set_ylabel('Fréquence')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

# Masquer dernier subplot vide
axes[-1].axis('off')

plt.tight_layout()
plt.show()
```

**Sources académiques** :
- Freedman, D., & Diaconis, P. (1981). "On the histogram as a density estimator". *Zeitschrift für Wahrscheinlichkeitstheorie*.
- Scott, D. W. (1979). "On optimal and data-based histograms". *Biometrika*.

---

### 3. Box Plots (Boîtes à Moustaches)

#### 3.1 Anatomie du Box Plot

**Structure complète** :

```
            Outlier ●
                |
    ┌───────────┴───────────┐
    │    Whisker supérieur   │ → Q3 + 1.5×IQR (ou max si < borne)
    │                        │
    ├───────────────────────┤ ← Q3 (75ème percentile)
    │                        │
    │         BOX            │
    │     (IQR region)       │
    │                        │
    ├───────────────────────┤ ← Q2 (Médiane, 50ème percentile)
    │                        │
    ├───────────────────────┤ ← Q1 (25ème percentile)
    │                        │
    │    Whisker inférieur   │ → Q1 - 1.5×IQR (ou min si > borne)
    └───────────┬───────────┘
                |
            Outlier ●
```

**Composants** :
- **Box** : Contient 50% centraux des données (de Q1 à Q3)
- **Ligne médiane** : Q2 (médiane)
- **Whiskers** : Étendent jusqu'à dernière valeur dans [Q1-1.5×IQR, Q3+1.5×IQR]
- **Points isolés** : Outliers au-delà des whiskers

**Pourquoi cette structure ?**

Elle résume **5 statistiques clés** en 1 graphique compact :
1. Minimum (ou whisker bas)
2. Q1
3. Médiane
4. Q3
5. Maximum (ou whisker haut)
+ **Outliers** visuellement identifiés

#### 3.2 Implémentation et Variations

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(100)

# Données avec outliers
data_groups = {
    'Groupe A': np.concatenate([np.random.normal(50, 10, 100), np.array([5, 95, 100])]),
    'Groupe B': np.random.normal(60, 15, 100),
    'Groupe C': np.concatenate([np.random.normal(55, 8, 100), np.array([20, 90])])
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plot basique (Matplotlib)
axes[0, 0].boxplot([data_groups[k] for k in data_groups.keys()], 
                    labels=data_groups.keys(),
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7))
axes[0, 0].set_ylabel('Valeur')
axes[0, 0].set_title('Box Plot Classique (Matplotlib)')
axes[0, 0].grid(alpha=0.3, axis='y')

# 2. Box plot horizontal
data_list = [data_groups[k] for k in data_groups.keys()]
axes[0, 1].boxplot(data_list, 
                    labels=data_groups.keys(),
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', alpha=0.7))
axes[0, 1].set_xlabel('Valeur')
axes[0, 1].set_title('Box Plot Horizontal')
axes[0, 1].grid(alpha=0.3, axis='x')

# 3. Box plot avec Seaborn (plus élégant)
df_data = pd.DataFrame([(nom, val) for nom, vals in data_groups.items() for val in vals],
                       columns=['Groupe', 'Valeur'])
sns.boxplot(data=df_data, x='Groupe', y='Valeur', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Box Plot (Seaborn)')
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Violin plot (box plot + densité)
sns.violinplot(data=df_data, x='Groupe', y='Valeur', ax=axes[1, 1], palette='Set3', inner='quartile')
axes[1, 1].set_title('Violin Plot (Distribution + Quartiles)')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Annotation détaillée d'un box plot
fig, ax = plt.subplots(figsize=(8, 6))

data_exemple = data_groups['Groupe A']
bp = ax.boxplot([data_exemple], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')

# Calcul statistiques
q1 = np.percentile(data_exemple, 25)
q2 = np.percentile(data_exemple, 50)
q3 = np.percentile(data_exemple, 75)
iqr = q3 - q1
whisker_bas = max(data_exemple.min(), q1 - 1.5*iqr)
whisker_haut = min(data_exemple.max(), q3 + 1.5*iqr)

# Annotations
offset_x = 1.15
ax.text(offset_x, q1, f'Q1 = {q1:.1f}', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.text(offset_x, q2, f'Médiane = {q2:.1f}', va='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.text(offset_x, q3, f'Q3 = {q3:.1f}', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.text(offset_x, q2 + iqr/2, f'IQR = {iqr:.1f}', va='center', fontsize=10, style='italic')

# Whiskers
ax.annotate('', xy=(0.95, whisker_haut), xytext=(0.95, q3),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(0.85, (whisker_haut + q3)/2, 'Whisker', fontsize=9, color='blue', rotation=90, va='center')

ax.set_ylabel('Valeur', fontsize=12)
ax.set_title('Anatomie d\'un Box Plot', fontsize=14, fontweight='bold')
ax.set_xticks([1])
ax.set_xticklabels(['Groupe A'])
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

#### 3.3 Interprétation des Box Plots

**Comparaison visuelle rapide** :

| Observation | Interprétation |
|-------------|----------------|
| **Médiane décalée vers Q1** | Distribution asymétrique droite |
| **Médiane décalée vers Q3** | Distribution asymétrique gauche |
| **Médiane centrée dans box** | Distribution symétrique |
| **Box large** | Grande dispersion (IQR élevé) |
| **Whiskers longs** | Données étalées |
| **Nombreux outliers** | Problème qualité données OU queues lourdes |
| **Whisker supérieur >> inférieur** | Asymétrie droite |

**Sources académiques** :
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. - Inventeur du box plot

---

### 4. Scatter Plots (Nuages de Points)

#### 4.1 Objectif et Construction

**Scatter plot** : Visualise la **relation entre 2 variables numériques**.

**Axes** :
- **Axe X** : Variable indépendante (prédicteur, cause)
- **Axe Y** : Variable dépendante (réponse, effet)

**Patterns visuels** :

1. **Corrélation positive** : Points montent de gauche à droite
2. **Corrélation négative** : Points descendent de gauche à droite
3. **Pas de corrélation** : Nuage sans pattern
4. **Relation non linéaire** : Pattern courbe
5. **Outliers** : Points très éloignés du nuage principal

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.random.seed(200)

# Différents types de relations
n = 100

relations = {
    'Corrélation Positive Forte': {
        'x': np.linspace(0, 10, n),
        'y': lambda x: 2*x + np.random.normal(0, 1, n)
    },
    'Corrélation Négative': {
        'x': np.linspace(0, 10, n),
        'y': lambda x: -1.5*x + 20 + np.random.normal(0, 2, n)
    },
    'Pas de Corrélation': {
        'x': np.random.uniform(0, 10, n),
        'y': lambda x: np.random.uniform(0, 20, n)
    },
    'Relation Non Linéaire (Quadratique)': {
        'x': np.linspace(-5, 5, n),
        'y': lambda x: x**2 + np.random.normal(0, 2, n)
    },
    'Relation Exponentielle': {
        'x': np.linspace(0, 5, n),
        'y': lambda x: np.exp(x/2) + np.random.normal(0, 0.5, n)
    },
    'Avec Outliers': {
        'x': np.concatenate([np.linspace(0, 10, 95), np.array([2, 8, 5, 7, 3])]),
        'y': lambda x: np.concatenate([2*x[:95] + np.random.normal(0, 1, 95), 
                                       np.array([25, 25, 25, 0, 0])])
    }
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (nom, data) in enumerate(relations.items()):
    x = data['x']
    y = data['y'](x) if callable(data['y']) else data['y']
    
    # Scatter plot
    axes[idx].scatter(x, y, alpha=0.6, s=50, edgecolor='black')
    
    # Ligne de régression (si relation linéaire attendue)
    if 'Non Linéaire' not in nom and 'Exponentielle' not in nom:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[idx].plot(x, p(x), "r--", linewidth=2, alpha=0.7, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Corrélation de Pearson
    corr, _ = pearsonr(x, y)
    
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('Y')
    axes[idx].set_title(f'{nom}\nr = {corr:.3f}', fontweight='bold')
    axes[idx].grid(alpha=0.3)
    if 'Non Linéaire' not in nom and 'Exponentielle' not in nom:
        axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

#### 4.2 Variations Avancées

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(300)

# Données avec variables additionnelles
n = 200
x = np.random.uniform(0, 10, n)
y = 2*x + np.random.normal(0, 2, n)
taille = np.random.uniform(10, 200, n)  # 3ème variable
categorie = np.random.choice(['A', 'B', 'C'], n)  # 4ème variable

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Scatter plot basique
axes[0, 0].scatter(x, y, alpha=0.6, s=50, edgecolor='black')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_title('Scatter Plot Basique')
axes[0, 0].grid(alpha=0.3)

# 2. Avec couleur par catégorie
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
for cat in ['A', 'B', 'C']:
    mask = categorie == cat
    axes[0, 1].scatter(x[mask], y[mask], alpha=0.6, s=50, 
                       label=f'Catégorie {cat}', c=colors[cat], edgecolor='black')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].set_title('Scatter avec Couleur (Catégorie)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Avec taille proportionnelle (Bubble chart)
axes[1, 0].scatter(x, y, s=taille, alpha=0.5, edgecolor='black')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_title('Bubble Chart (Taille = 3ème Variable)')
axes[1, 0].grid(alpha=0.3)

# 4. Avec ligne de régression et intervalle de confiance (Seaborn)
df = pd.DataFrame({'X': x, 'Y': y})
sns.regplot(data=df, x='X', y='Y', ax=axes[1, 1], scatter_kws={'alpha':0.5, 's':50})
axes[1, 1].set_title('Régression avec IC 95%')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 4.3 Détection de Patterns

**Checklist visuelle** :

- [ ] **Linéarité** : Points alignés sur droite ?
- [ ] **Homoscédasticité** : Dispersion constante le long de X ?
- [ ] **Outliers** : Points très éloignés ?
- [ ] **Groupes** : Clusters distincts visibles ?
- [ ] **Non-linéarité** : Pattern courbe ?

```python
# Exemple : Diagnostic de régression
np.random.seed(400)

x = np.linspace(0, 10, 100)
y_hetero = x + np.random.normal(0, x*0.3, 100)  # Hétéroscédasticité

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter original
axes[0].scatter(x, y_hetero, alpha=0.6, s=50, edgecolor='black')
z = np.polyfit(x, y_hetero, 1)
p = np.poly1d(z)
axes[0].plot(x, p(x), "r--", linewidth=2)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Données avec Hétéroscédasticité')
axes[0].grid(alpha=0.3)

# Résidus vs Fitted (diagnostic)
residus = y_hetero - p(x)
axes[1].scatter(p(x), residus, alpha=0.6, s=50, edgecolor='black')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Valeurs Ajustées')
axes[1].set_ylabel('Résidus')
axes[1].set_title('Résidus vs Fitted (Détecte Hétéroscédasticité)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 Le graphique résidus montre que la dispersion AUGMENTE avec X")
print("   → Problème d'hétéroscédasticité → Transformer données ou utiliser régression robuste")
```

---

### 5. Autres Visualisations Essentielles

#### 5.1 Line Plots (Séries Temporelles)

**Usage** : Données ordonnées (temps, séquences)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(500)

# Série temporelle avec tendance + saisonnalité
t = np.arange(0, 100)
tendance = 0.5 * t
saisonnalite = 10 * np.sin(2 * np.pi * t / 12)
bruit = np.random.normal(0, 3, 100)
serie = 50 + tendance + saisonnalite + bruit

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Line plot classique
axes[0].plot(t, serie, linewidth=2, marker='o', markersize=4, alpha=0.7)
axes[0].set_xlabel('Temps')
axes[0].set_ylabel('Valeur')
axes[0].set_title('Série Temporelle (Line Plot)')
axes[0].grid(alpha=0.3)

# Décomposition visuelle
axes[1].plot(t, 50 + tendance, label='Tendance', linewidth=2)
axes[1].plot(t, saisonnalite, label='Saisonnalité', linewidth=2)
axes[1].plot(t, serie, label='Série Observée', alpha=0.5, linewidth=1)
axes[1].set_xlabel('Temps')
axes[1].set_ylabel('Valeur')
axes[1].set_title('Décomposition : Tendance + Saisonnalité')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 5.2 Heatmaps (Matrices de Corrélation)

**Usage** : Visualiser matrices, corrélations entre multiples variables

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(600)

# Génération features corrélées
n = 100
feature1 = np.random.normal(50, 10, n)
feature2 = 0.8 * feature1 + np.random.normal(0, 5, n)  # Corrélée avec feature1
feature3 = np.random.normal(30, 8, n)  # Indépendante
feature4 = -0.6 * feature1 + np.random.normal(0, 6, n)  # Anti-corrélée

df = pd.DataFrame({
    'Feature 1': feature1,
    'Feature 2': feature2,
    'Feature 3': feature3,
    'Feature 4': feature4
})

# Matrice de corrélation
corr_matrix = df.corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap avec Seaborn
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, ax=axes[0], 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[0].set_title('Matrice de Corrélation (Heatmap)')

# Pair plot (scatter matrix)
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.5, figsize=(10, 10), diagonal='kde', ax=axes[1])
axes[1].set_title('Pair Plot (Relations Bivariées)')

plt.tight_layout()
plt.show()

print("\n💡 INTERPRÉTATION :")
print("Feature 1 ↔ Feature 2 : Corrélation forte positive (0.8)")
print("Feature 1 ↔ Feature 4 : Corrélation forte négative (-0.6)")
print("Feature 3 : Indépendante des autres")
```

---

## 💡 Compréhension Intuitive

### L'Analogie de la Carte Géographique

**Statistiques = Coordonnées GPS** (précises mais abstraites)  
→ Latitude: 48.8566, Longitude: 2.3522

**Visualisation = Carte visuelle** (intuitive, patterns visibles)  
→ Vous voyez immédiatement : "Ah, c'est Paris ! Proche de la Seine, au centre de la France"

De même :
- **Moyenne, écart-type** = Coordonnées statistiques
- **Histogramme, box plot** = Carte de la distribution

**Sans carte (visualisation)**, vous ne verriez pas :
- Les clusters (villes groupées)
- Les outliers (îles isolées)
- Les patterns (côte, montagnes)

### Questions Rapides

1. **Q1** : Dataset A et B ont même moyenne et écart-type. Sont-ils identiques ?
   - *Réponse* : NON ! (Quartet d'Anscombe). Toujours visualiser.

2. **Q2** : Vous voyez 2 groupes distincts sur un histogramme. Quelle forme ?
   - *Réponse* : Distribution bimodale

3. **Q3** : Box plot avec médiane au ras de Q1. Distribution symétrique ?
   - *Réponse* : NON, asymétrique à droite

4. **Q4** : Scatter plot en forme de U. Corrélation de Pearson ?
   - *Réponse* : Proche de 0 (Pearson mesure LINÉARITÉ uniquement)

---

## 💻 Implémentation Pratique

### Dashboard Complet d'Analyse Exploratoire

```python
"""
Titre : Dashboard EDA automatique
Objectif : Fonction générique pour analyse visuelle complète d'un dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def eda_dashboard(df, target_col=None, figsize=(20, 12)):
    """
    Crée dashboard EDA complet pour un DataFrame.
    
    Args:
        df (DataFrame): Dataset à analyser
        target_col (str): Colonne cible si régression/classification
        figsize (tuple): Taille de la figure
    """
    # Séparation colonnes numériques/catégorielles
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_numeric = len(numeric_cols)
    n_cat = len(cat_cols)
    
    print(f"📊 DASHBOARD EDA")
    print(f"="*70)
    print(f"Nombre d'observations : {len(df)}")
    print(f"Nombre de features    : {len(df.columns)}")
    print(f"  - Numériques        : {n_numeric}")
    print(f"  - Catégorielles     : {n_cat}")
    print(f"\nValeurs manquantes :")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  Aucune ✅")
    
    # Création figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    
    # 1. Distributions des variables numériques (histogrammes)
    for idx, col in enumerate(numeric_cols[:4]):  # Max 4
        ax = fig.add_subplot(gs[0, idx])
        ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Moyenne')
        ax.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Médiane')
        ax.set_xlabel(col)
        ax.set_ylabel('Fréquence')
        ax.set_title(f'Distribution: {col}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 2. Box plots comparatifs
    ax = fig.add_subplot(gs[1, :2])
    df[numeric_cols[:min(5, n_numeric)]].boxplot(ax=ax, patch_artist=True)
    ax.set_title('Box Plots Comparatifs (Variables Numériques)')
    ax.set_ylabel('Valeur (normalisée)')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Matrice de corrélation
    if n_numeric > 1:
        ax = fig.add_subplot(gs[1, 2:])
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, square=True, linewidths=1, 
                    cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
        ax.set_title('Matrice de Corrélation')
    
    # 4. Scatter plots (si target définie)
    if target_col and target_col in numeric_cols:
        features_to_plot = [c for c in numeric_cols if c != target_col][:3]
        for idx, col in enumerate(features_to_plot):
            ax = fig.add_subplot(gs[2, idx])
            ax.scatter(df[col], df[target_col], alpha=0.5, s=30, edgecolor='black')
            
            # Régression linéaire
            mask = df[col].notna() & df[target_col].notna()
            if mask.sum() > 2:
                z = np.polyfit(df[col][mask], df[target_col][mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df[col].min(), df[col].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
                
                # Corrélation
                corr, _ = stats.pearsonr(df[col][mask], df[target_col][mask])
                ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel(col)
            ax.set_ylabel(target_col)
            ax.set_title(f'{col} vs {target_col}')
            ax.grid(alpha=0.3)
    
    # 5. Variables catégorielles (bar charts)
    for idx, col in enumerate(cat_cols[:3]):
        ax = fig.add_subplot(gs[3, idx])
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution: {col}')
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. QQ-plot (normalité) pour target
    if target_col and target_col in numeric_cols:
        ax = fig.add_subplot(gs[2, 3])
        stats.probplot(df[target_col].dropna(), dist="norm", plot=ax)
        ax.set_title(f'QQ-Plot: {target_col}')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Dashboard EDA Automatique', fontsize=18, fontweight='bold', y=0.995)
    plt.show()

# Test avec dataset exemple
np.random.seed(700)
n = 500

df_test = pd.DataFrame({
    'age': np.random.normal(40, 12, n).clip(18, 80),
    'revenu': np.random.lognormal(10, 0.5, n),
    'score': 50 + 0.3*np.random.normal(40, 12, n) + 0.0005*np.random.lognormal(10, 0.5, n) + np.random.normal(0, 10, n),
    'nb_achats': np.random.poisson(5, n),
    'categorie': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
    'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], n)
})

eda_dashboard(df_test, target_col='score')
```

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Axes Tronqués/Manipulés

```python
# Démonstration graphiques trompeurs
import numpy as np
import matplotlib.pyplot as plt

ventes_2023 = 100
ventes_2024 = 110  # +10%

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ❌ MAUVAIS : Axe Y commence à 95 (exagère différence)
axes[0].bar(['2023', '2024'], [ventes_2023, ventes_2024], color=['blue', 'green'], edgecolor='black', alpha=0.7)
axes[0].set_ylim(95, 111)
axes[0].set_ylabel('Ventes (M€)')
axes[0].set_title('❌ TROMPEUR : Axe Y tronqué\n(Donne impression de doublement)', color='red', fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# ✅ BON : Axe Y commence à 0
axes[1].bar(['2023', '2024'], [ventes_2023, ventes_2024], color=['blue', 'green'], edgecolor='black', alpha=0.7)
axes[1].set_ylim(0, 120)
axes[1].set_ylabel('Ventes (M€)')
axes[1].set_title('✅ HONNÊTE : Axe Y à zéro\n(Montre vraie proportion +10%)', color='green', fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("💡 RÈGLE : Pour bar charts, TOUJOURS commencer axe Y à zéro")
print("   Exception : Line plots de séries temporelles (variations relatives)")
```

---

### ❌ Erreur 2 : Pie Charts (Déconseillés)

```python
# Pourquoi éviter les pie charts
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D', 'E']
valeurs = [30, 25, 20, 15, 10]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ❌ Pie chart (difficile à comparer)
axes[0].pie(valeurs, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
axes[0].set_title('❌ PIE CHART\n(Difficile de comparer B vs C)', color='red', fontweight='bold')

# ✅ Bar chart (facile à comparer)
axes[1].barh(categories, valeurs, color=plt.cm.Set3.colors, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Valeur')
axes[1].set_title('✅ BAR CHART HORIZONTAL\n(Comparaison immédiate)', color='green', fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("💡 ÉVITER pie charts car :")
print("  - Cerveau humain mauvais pour comparer angles")
print("  - Impossible si >5 catégories")
print("  - Utiliser BAR CHART à la place")
```

---

### ✅ Bonne Pratique 1 : Choisir Palette de Couleurs Appropriée

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.random.randn(100, 5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Palette séquentielle (ordre/intensité)
sns.heatmap(data, cmap='Blues', ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title('Séquentielle (Blues)\nPour grandeurs ordonnées')

# Palette divergente (positif/négatif)
sns.heatmap(data, cmap='RdBu_r', center=0, ax=axes[1], cbar_kws={"shrink": 0.8})
axes[1].set_title('Divergente (RdBu)\nPour données centrées sur 0')

# Palette qualitative (catégories)
categories = np.random.choice([0, 1, 2, 3, 4], (100, 5))
sns.heatmap(categories, cmap='Set3', ax=axes[2], cbar_kws={"shrink": 0.8})
axes[2].set_title('Qualitative (Set3)\nPour catégories nominales')

plt.tight_layout()
plt.show()

print("💡 GUIDE PALETTES :")
print("  - Séquentielle : 1 couleur, intensité croissante (ex: densité)")
print("  - Divergente : 2 couleurs opposées, centre neutre (ex: corrélation)")
print("  - Qualitative : Couleurs distinctes sans ordre (ex: catégories)")
```

---

### 📋 Checklist Visualisation

Avant de publier un graphique :

- [ ] **Axes** : Labels clairs, unités spécifiées ?
- [ ] **Titre** : Descriptif et informatif ?
- [ ] **Légende** : Présente si plusieurs séries ?
- [ ] **Échelle** : Appropriée (log si nécessaire) ?
- [ ] **Axe Y** : Commence à 0 pour bar charts ?
- [ ] **Couleurs** : Palette appropriée, accessible (daltoniens) ?
- [ ] **Police** : Lisible (min 10pt) ?
- [ ] **Grille** : Présente pour faciliter lecture valeurs ?
- [ ] **Aspect ratio** : Approprié (pas trop étiré) ?
- [ ] **Message** : Le graphique répond-il à la question posée ?

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers & Livres Fondamentaux

1. **"The Visual Display of Quantitative Information"**
   - **Auteur** : Edward Tufte (1983)
   - **Contribution** : Bible de la visualisation, data-ink ratio
   - **Niveau** : Tous niveaux

2. **"Graphs in Statistical Analysis"**
   - **Auteur** : Francis Anscombe (1973)
   - **Contribution** : Quartet d'Anscombe, importance visualisation
   - **Niveau** : Accessible

3. **"Exploratory Data Analysis"**
   - **Auteur** : John Tukey (1977)
   - **Contribution** : Invention box plot, stem-and-leaf
   - **Niveau** : Intermédiaire

---

### 📚 Ressources

**Galeries d'inspiration** :
- [Python Graph Gallery](https://www.python-graph-gallery.com/) - Exemples code
- [From Data to Viz](https://www.data-to-viz.com/) - Arbre de décision choix graphiques
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

**Tutoriels** :
- **Matplotlib Tutorial** - Documentation officielle
- **Seaborn Tutorial** - Statistical visualization
- **Plotly** - Visualisations interactives

---

### 📖 Cours Connexes

**Suite naturelle** :
- [[distribution_analysis]] - Analyser formes (skewness, kurtosis)
- [[normality_tests]] - Tests statistiques de normalité
- [[correlation_covariance]] - Mesurer et visualiser corrélations

**Applications** :
- [[regression_diagnostics]] - Graphiques diagnostic (résidus, QQ-plots)
- [[time_series_basics]] - Visualisations temporelles avancées
- [[ab_testing]] - Visualiser résultats tests A/B

---

## 📝 Résumé Rapide

### Types de Graphiques par Usage

| Usage | Graphique Recommandé | Alternative |
|-------|---------------------|-------------|
| **Distribution 1 variable** | Histogramme, Box plot | Violin plot, KDE |
| **Comparer groupes** | Box plot groupé | Violin plot, Strip plot |
| **Relation 2 variables** | Scatter plot | Hexbin (si dense) |
| **Évolution temporelle** | Line plot | Area chart |
| **Composition** | Stacked bar | Treemap |
| **Corrélations multiples** | Heatmap | Pair plot |

### Code Minimal

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogramme
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Valeur')
plt.ylabel('Fréquence')
plt.title('Distribution')
plt.show()

# Box plot
plt.boxplot(data)
plt.ylabel('Valeur')
plt.show()

# Scatter plot
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Seaborn (plus élégant)
sns.set_style("whitegrid")
sns.histplot(data, kde=True)
sns.boxplot(data=df, x='categorie', y='valeur')
sns.scatterplot(data=df, x='x', y='y', hue='categorie')
```


---

**Cours créé le 2026-03-18** ✅
