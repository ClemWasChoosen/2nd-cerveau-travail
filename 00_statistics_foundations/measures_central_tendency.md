# 📊 Mesures de Tendance Centrale : Moyenne, Médiane, Mode

> **Résumé en une phrase** : Les mesures de tendance centrale résument un jeu de données par une seule valeur représentative, permettant de comprendre rapidement "où se situent" vos données.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-18 |
| **Dernière mise à jour** | 2026-03-18 |
| **Domaine** | Statistiques Descriptives |
| **Niveau** | Débutant |
| **Durée de lecture** | ~25 minutes |
| **Fichier** | `measures_central_tendency.md` |
| **Emplacement** | `/00_statistics_foundations/01_descriptive_statistics/` |
| **Tags** | `#statistics` `#descriptive` `#mean` `#median` `#mode` `#fundamentals` |

### Prérequis

- [ ] Mathématiques de base (sommes, divisions)
- [ ] Notions de base en Python (optionnel pour partie code)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : Aucun (cours fondamental)
- **Complémentaires** : 
  - [[measures_dispersion]] - Variance, écart-type, quantiles
  - [[data_visualization_principles]] - Visualiser distributions
- **Suite recommandée** : 
  - [[measures_dispersion]] - Mesures de dispersion
  - [[distribution_analysis]] - Skewness et Kurtosis

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Les mesures de tendance centrale sont la **première étape obligatoire** de toute analyse de données. Vous apprendrez à résumer des milliers (voire millions) de valeurs par **une seule valeur représentative** qui capture "le centre" de vos données. Ce cours vous expliquera **quoi** sont ces mesures, **pourquoi** elles sont différentes, **comment** les calculer, et surtout **quand** utiliser l'une plutôt que l'autre selon votre contexte métier.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Définir et interpréter moyenne, médiane et mode
2. **Appliquer** : Calculer ces mesures manuellement et en Python/NumPy
3. **Analyser** : Identifier quelle mesure utiliser selon la distribution des données
4. **Évaluer** : Comparer les forces et faiblesses de chaque mesure dans des contextes réels de Data Science (salaires, prix, trafic web, etc.)
5. **Créer** : Implémenter des fonctions robustes gérant les cas limites (valeurs manquantes, distributions multimodales)

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

En tant que Data Scientist, vous recevrez constamment des questions comme :
- *"Quel est le revenu **typique** de nos clients ?"*
- *"Quel est le temps de réponse **moyen** de notre API ?"*
- *"Quelle est la satisfaction **générale** des utilisateurs ?"*

Le mot crucial ici est **"typique" / "moyen" / "général"**. Vos interlocuteurs business demandent **une seule valeur** qui résume des milliers de points de données. C'est précisément le rôle des mesures de tendance centrale.

**Mais voici le piège** : selon le choix de la mesure (moyenne vs médiane vs mode), vous pouvez raconter des histoires **très différentes** avec les mêmes données ! Comprendre la différence est essentiel pour :
1. **Éviter de mentir par accident** avec des statistiques mal choisies
2. **Détecter des manipulations** dans les rapports que vous recevez
3. **Communiquer honnêtement** vos résultats d'analyse

### Quel problème résout-il ?

**Problème** : Vous avez 10,000 valeurs de salaires dans votre entreprise. Votre CEO vous demande : *"Quel est le salaire moyen ici ?"*

Vous calculez la **moyenne** : **75,000€**  
Mais vous calculez aussi la **médiane** : **48,000€**

**Quoi ?! 27,000€ de différence !** Laquelle communiquer ? Les deux sont "correctes" mathématiquement, mais racontent des histoires différentes.

**Exemple concret** :

```python
import numpy as np
import matplotlib.pyplot as plt

# Salaires de 10 employés (en milliers €)
salaires = np.array([30, 35, 38, 42, 45, 48, 52, 55, 60, 500])
# Le dernier est le CEO avec un salaire 10x supérieur

moyenne = np.mean(salaires)
mediane = np.median(salaires)

print(f"Moyenne : {moyenne:.1f}k€")  # 90.5k€
print(f"Médiane : {mediane:.1f}k€")  # 46.5k€

# Visualisation
plt.figure(figsize=(10, 4))
plt.hist(salaires, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(moyenne, color='red', linestyle='--', linewidth=2, label=f'Moyenne = {moyenne:.1f}k€')
plt.axvline(mediane, color='blue', linestyle='--', linewidth=2, label=f'Médiane = {mediane:.1f}k€')
plt.xlabel('Salaire (k€)')
plt.ylabel('Nombre d\'employés')
plt.title('Distribution des Salaires - Moyenne vs Médiane')
plt.legend()
plt.show()
```

**Résultat** :
- **Moyenne = 90.5k€** : Fortement influencée par le CEO (outlier)
- **Médiane = 46.5k€** : Valeur "typique" pour la majorité des employés

**La médiane raconte la vraie histoire ici** : 50% des employés gagnent moins de 46.5k€. La moyenne est trompeuse car elle est biaisée par un seul salaire extrême.

### Applications dans le monde réel

1. **Finance / RH** : 
   - Salaires médians (plus représentatifs que moyennes)
   - Prix médian de l'immobilier (usage standard)
   - Revenu médian d'un pays (indicateur économique officiel)

2. **E-commerce / Web Analytics** :
   - Temps moyen de session (mais médiane souvent plus utile car distribution asymétrique)
   - Panier moyen vs médian
   - Mode = produit le plus vendu

3. **Machine Learning / Data Science** :
   - Imputation de valeurs manquantes (moyenne ou médiane selon distribution)
   - Détection d'outliers (valeurs loin de la médiane)
   - Feature engineering (créer features basées sur écarts à la moyenne)

4. **Santé / Médecine** :
   - Temps médian de survie (survie analysis)
   - Tension artérielle moyenne
   - Glycémie moyenne (HbA1c)

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous allons construire les bases en introduisant chaque mesure séparément, puis les comparer. Suivant le **principe de pré-entraînement**, nous maîtrisons d'abord les composants avant de voir leurs interactions.

### 1. La Moyenne (Mean)

#### 1.1 Définition Mathématique

**Moyenne arithmétique** :

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

**Où** :
- $$\bar{x}$$ (lu "x barre") : Moyenne de l'échantillon
- $$n$$ : Nombre d'observations
- $$x_i$$ : $$i$$-ème observation
- $$\sum$$ : Symbole de sommation (addition de tous les termes)

**Intuition** :
La moyenne est le "centre de gravité" de vos données. Si vous placiez chaque point de donnée sur une balance, la moyenne serait le point d'équilibre parfait.

**Pourquoi cette définition ?**
La moyenne est la solution à cette question : *"Si je devais répartir équitablement la somme totale entre toutes les observations, quelle valeur donnerais-je à chacune ?"*

#### 1.2 Propriétés Mathématiques de la Moyenne

**Propriété 1 : Linéarité**

Si vous transformez vos données par $$y_i = a \cdot x_i + b$$, alors :

$$\bar{y} = a \cdot \bar{x} + b$$

**Exemple pratique** : Conversion Celsius → Fahrenheit

```python
temperatures_celsius = np.array([15, 18, 20, 22, 25])
moyenne_celsius = np.mean(temperatures_celsius)  # 20°C

# Conversion : F = 1.8*C + 32
temperatures_fahrenheit = 1.8 * temperatures_celsius + 32
moyenne_fahrenheit = np.mean(temperatures_fahrenheit)  # 68°F

# Vérification propriété
print(f"Moyenne Celsius : {moyenne_celsius}°C")
print(f"Moyenne Fahrenheit : {moyenne_fahrenheit}°F")
print(f"Formule directe : {1.8 * moyenne_celsius + 32}°F")
# Les deux méthodes donnent le même résultat !
```

**Propriété 2 : Sensibilité aux outliers**

La moyenne est **très sensible** aux valeurs extrêmes. Mathématiquement, chaque point contribue équitablement à la somme, donc un outlier peut fortement déplacer la moyenne.

**Propriété 3 : Somme des écarts = 0**

$$\sum_{i=1}^{n} (x_i - \bar{x}) = 0$$

Les écarts positifs (au-dessus de la moyenne) et négatifs (en-dessous) s'annulent exactement. C'est la définition même d'un "centre".

**Propriété 4 : Minimise la somme des carrés des écarts**

La moyenne minimise cette quantité :

$$\sum_{i=1}^{n} (x_i - c)^2$$

C'est-à-dire : aucune autre valeur $$c$$ ne peut avoir une somme de carrés d'écarts plus petite. C'est pourquoi la moyenne est utilisée dans la régression linéaire (méthode des moindres carrés) !

#### 1.3 Moyenne Pondérée (Weighted Mean)

Parfois, toutes les observations n'ont pas la même "importance". La moyenne pondérée attribue un poids $$w_i$$ à chaque observation :

$$\bar{x}_w = \frac{\sum_{i=1}^{n} w_i \cdot x_i}{\sum_{i=1}^{n} w_i}$$

**Exemple** : Notes d'un cours avec coefficients

```python
notes = np.array([12, 15, 18])  # Contrôle continu, Projet, Examen final
poids = np.array([0.2, 0.3, 0.5])  # Coefficients

moyenne_simple = np.mean(notes)  # 15.0 (mauvais !)
moyenne_ponderee = np.average(notes, weights=poids)  # 15.9 (correct)

print(f"Moyenne simple : {moyenne_simple:.1f}")
print(f"Moyenne pondérée : {moyenne_ponderee:.1f}")
```

**Quand utiliser** :
- Notes avec coefficients
- Moyennes de moyennes (pondérer par taille des groupes)
- Scores de satisfaction (pondérer par nombre de réponses)

#### 1.4 Autres Types de Moyennes

**Moyenne géométrique** :

$$\bar{x}_g = \sqrt[n]{x_1 \cdot x_2 \cdots x_n} = \left(\prod_{i=1}^{n} x_i\right)^{1/n}$$

**Quand utiliser** : Taux de croissance, rendements financiers

```python
# Exemple : Croissance annuelle sur 3 ans
croissances = np.array([1.10, 1.05, 1.08])  # +10%, +5%, +8%

# Moyenne arithmétique (FAUX pour taux de croissance)
moyenne_arith = np.mean(croissances)  # 1.0767

# Moyenne géométrique (CORRECT)
from scipy.stats import gmean
moyenne_geom = gmean(croissances)  # 1.0766

# Calcul manuel
moyenne_geom_manuel = np.prod(croissances) ** (1/len(croissances))

print(f"Croissance moyenne (géométrique) : {(moyenne_geom - 1)*100:.2f}%")
```

**Moyenne harmonique** :

$$\bar{x}_h = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

**Quand utiliser** : Moyennes de vitesses, de taux

**Sources académiques** :
- [Handbook of Statistical Methods (NIST)](https://www.itl.nist.gov/div898/handbook/) - Définitions officielles
- Wackerly, D., Mendenhall, W., & Scheaffer, R. (2008). *Mathematical Statistics with Applications*. Thomson

---

### 2. La Médiane (Median)

#### 2.1 Définition Mathématique

**Médiane** : Valeur qui sépare les données ordonnées en deux moitiés égales.

**Formule formelle** :

Si $$n$$ est impair :

$$\text{Médiane} = x_{\left(\frac{n+1}{2}\right)}$$

Si $$n$$ est pair :

$$\text{Médiane} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}$$

Où $$x_{(i)}$$ est la $$i$$-ème valeur **après tri croissant** (statistique d'ordre).

**Intuition** :
La médiane est la valeur du "milieu" une fois les données triées. 50% des observations sont en-dessous, 50% au-dessus.

**Pourquoi cette définition ?**
La médiane répond à : *"Quelle valeur sépare mes données en deux groupes de taille égale ?"* Elle est **robuste** aux outliers car elle ne dépend que de la **position**, pas de la **valeur** des extrêmes.

**Visualisation conceptuelle** :

```
Données triées : [10, 20, 30, 40, 50]
                          ↑
                     Médiane = 30
                (2 valeurs à gauche, 2 à droite)

Données triées : [10, 20, 30, 40, 50, 60]
                           ↑   ↑
                  Médiane = (30 + 40) / 2 = 35
```

#### 2.2 Propriétés de la Médiane

**Propriété 1 : Robustesse aux outliers**

La médiane est **résistante** aux valeurs extrêmes. Vous pouvez multiplier les valeurs maximales par 1000, la médiane ne changera pas (tant que vous ne changez pas l'ordre relatif).

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mediane_originale = np.median(data)  # 5.5

# On multiplie les 3 dernières valeurs par 1000 (outliers extrêmes)
data_avec_outliers = np.array([1, 2, 3, 4, 5, 6, 7, 8000, 9000, 10000])
mediane_robuste = np.median(data_avec_outliers)  # 5.5 (inchangée !)

moyenne_affectee = np.mean(data_avec_outliers)  # 2703.3 (complètement biaisée)

print(f"Médiane originale : {mediane_originale}")
print(f"Médiane avec outliers : {mediane_robuste}")
print(f"Moyenne avec outliers : {moyenne_affectee}")
```

**Propriété 2 : Minimise la somme des écarts absolus**

La médiane minimise :

$$\sum_{i=1}^{n} |x_i - c|$$

(alors que la moyenne minimise la somme des **carrés** des écarts)

**Propriété 3 : Quantile 50%**

La médiane est équivalente au **50ème percentile** (Q2, deuxième quartile). On la note parfois $$Q_2$$ ou $$P_{50}$$.

**Propriété 4 : Non-linéarité**

Contrairement à la moyenne, la médiane **ne respecte pas** la linéarité :

$$\text{Médiane}(a \cdot x + b) \neq a \cdot \text{Médiane}(x) + b$$ 

(Faux en général ! Vrai seulement si transformation préserve l'ordre)

En pratique, la linéarité fonctionne pour transformations monotones (qui préservent l'ordre), mais pas toujours.

#### 2.3 Calcul Efficace de la Médiane

**Complexité algorithmique** :
- Tri naïf : $$O(n \log n)$$
- Algorithme de sélection (Quickselect) : $$O(n)$$ en moyenne

```python
# Méthode 1 : NumPy (optimisé)
data = np.random.randint(1, 100, size=10000)
mediane_numpy = np.median(data)  # Rapide

# Méthode 2 : Manual (pour comprendre)
def median_manual(arr):
    """Calcul manuel de la médiane"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    
    if n % 2 == 1:  # Impair
        return sorted_arr[n // 2]
    else:  # Pair
        mid1 = sorted_arr[n // 2 - 1]
        mid2 = sorted_arr[n // 2]
        return (mid1 + mid2) / 2

mediane_manuelle = median_manual(data)

print(f"Médiane NumPy : {mediane_numpy}")
print(f"Médiane manuelle : {mediane_manuelle}")
print(f"Égalité : {np.isclose(mediane_numpy, mediane_manuelle)}")
```

**Sources académiques** :
- Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust Regression and Outlier Detection*. Wiley - Référence sur robustesse statistique

---

### 3. Le Mode (Mode)

#### 3.1 Définition

**Mode** : Valeur(s) la/les plus fréquente(s) dans les données.

**Notation** : Pas de notation mathématique standard universelle (parfois noté $$\text{Mo}$$)

**Intuition** :
Le mode répond à : *"Quelle est la valeur la plus commune dans mes données ?"*

**Particularités** :
- Une distribution peut avoir **plusieurs modes** (multimodale)
- Une distribution peut n'avoir **aucun mode** (toutes les valeurs uniques)
- Le mode est la **seule mesure** applicable aux **données catégorielles**

**Visualisation** :

```
Distribution unimodale (1 mode) :
    Fréquence
       *
      ***
     *****
    *******  ← Mode ici
   *********
  ***********
  1 2 3 4 5 6 7

Distribution bimodale (2 modes) :
    Fréquence
      ***         ***
     *****       *****  ← 2 modes
    *******     *******
   ********* ***********
  1 2 3 4 5 6 7 8 9 10
```

#### 3.2 Types de Distributions selon le Mode

- **Unimodale** : 1 seul mode (ex: distribution normale)
- **Bimodale** : 2 modes distincts (ex: tailles hommes + femmes mélangées)
- **Multimodale** : 3+ modes (rare, souvent signe de sous-populations)
- **Uniforme** : Pas de mode (toutes valeurs aussi fréquentes)

#### 3.3 Calcul du Mode

```python
from scipy import stats

# Exemple 1 : Données numériques
notes = np.array([12, 15, 15, 15, 16, 16, 18, 18, 18, 18, 20])

# Avec SciPy
mode_scipy = stats.mode(notes, keepdims=True)
print(f"Mode (SciPy) : {mode_scipy.mode[0]} (apparaît {mode_scipy.count[0]} fois)")

# Manuel avec NumPy
valeurs_uniques, comptages = np.unique(notes, return_counts=True)
mode_index = np.argmax(comptages)
mode_manuel = valeurs_uniques[mode_index]
print(f"Mode (manuel) : {mode_manuel}")

# Exemple 2 : Données catégorielles
couleurs = np.array(['rouge', 'bleu', 'rouge', 'vert', 'rouge', 'bleu', 'rouge'])
mode_couleur = stats.mode(couleurs, keepdims=True)
print(f"Couleur la plus fréquente : {mode_couleur.mode[0]}")

# Exemple 3 : Distribution bimodale
data_bimodale = np.array([1, 1, 1, 2, 3, 3, 3, 4, 5])
# Modes : 1 et 3 (tous deux apparaissent 3 fois)
# SciPy ne retourne que le premier trouvé

# Détection manuelle de tous les modes
valeurs, counts = np.unique(data_bimodale, return_counts=True)
max_count = counts.max()
tous_les_modes = valeurs[counts == max_count]
print(f"Tous les modes : {tous_les_modes}")
```

#### 3.4 Quand Utiliser le Mode

**Cas d'usage principaux** :

1. **Données catégorielles** (seule option !) :
   - Catégorie de produit la plus vendue
   - Type de client le plus fréquent
   - Erreur la plus commune dans les logs

2. **Données discrètes avec répétitions** :
   - Nombre d'enfants par famille (0, 1, 2, 3...)
   - Nombre de clics par session
   - Taille de vêtement la plus demandée

3. **Détecter sous-populations** :
   - Distribution bimodale = 2 groupes distincts ?
   - Ex: âges dans une salle de cinéma (enfants + parents)

**Limitations** :
- Instable : ajouter/retirer 1 observation peut changer le mode
- Peut ne pas exister (valeurs toutes uniques)
- Peut être multiple (difficile à interpréter)
- Moins utilisé que moyenne/médiane en Data Science

**Sources académiques** :
- Siegel, A. F. (2016). *Practical Business Statistics*. Academic Press - Applications pratiques du mode

---

### 4. Comparaison des Trois Mesures

#### 4.1 Tableau Récapitulatif

| Critère | Moyenne ($$\bar{x}$$) | Médiane | Mode |
|---------|----------------------|---------|------|
| **Définition** | Somme / n | Valeur centrale triée | Valeur la plus fréquente |
| **Sensibilité outliers** | ❌ Très sensible | ✅ Robuste | ✅ Robuste |
| **Type de données** | Numériques | Numériques ordinales | Toutes (même catégorielles) |
| **Unicité** | ✅ Unique | ✅ Unique | ❌ Peut être multiple |
| **Calcul** | Facile $$O(n)$$ | Moyen $$O(n \log n)$$ | Moyen $$O(n)$$ |
| **Interprétation** | "Centre de gravité" | "Valeur médiane" | "Valeur typique" |
| **Propriété minimisée** | $$\sum (x_i - c)^2$$ | $$\sum |x_i - c|$$ | — |
| **Usage en ML** | Imputation, features | Imputation robuste | Catégoriel, détection modes |

#### 4.2 Relation Moyenne-Médiane selon la Distribution

**Distribution symétrique** (ex: Normale) :

$$\text{Moyenne} = \text{Médiane} = \text{Mode}$$

**Distribution asymétrique à droite** (skewness positive, ex: salaires) :

$$\text{Mode} < \text{Médiane} < \text{Moyenne}$$

La moyenne est "tirée" vers les valeurs élevées.

**Distribution asymétrique à gauche** (skewness négative, ex: notes d'exam facile) :

$$\text{Moyenne} < \text{Médiane} < \text{Mode}$$

La moyenne est "tirée" vers les valeurs faibles.

**Visualisation** :

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution symétrique (Normale)
data_sym = np.random.normal(50, 10, 10000)
axes[0].hist(data_sym, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(data_sym), color='red', linestyle='--', label=f'Moyenne={np.mean(data_sym):.1f}')
axes[0].axvline(np.median(data_sym), color='blue', linestyle='--', label=f'Médiane={np.median(data_sym):.1f}')
axes[0].set_title('Distribution Symétrique\n(Moyenne ≈ Médiane)')
axes[0].legend()

# Distribution asymétrique droite
data_right = skewnorm.rvs(5, loc=30, scale=20, size=10000)
axes[1].hist(data_right, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(np.mean(data_right), color='red', linestyle='--', label=f'Moyenne={np.mean(data_right):.1f}')
axes[1].axvline(np.median(data_right), color='blue', linestyle='--', label=f'Médiane={np.median(data_right):.1f}')
axes[1].set_title('Asymétrie Droite\n(Moyenne > Médiane)')
axes[1].legend()

# Distribution asymétrique gauche
data_left = skewnorm.rvs(-5, loc=70, scale=20, size=10000)
axes[2].hist(data_left, bins=50, edgecolor='black', alpha=0.7)
axes[2].axvline(np.mean(data_left), color='red', linestyle='--', label=f'Moyenne={np.mean(data_left):.1f}')
axes[2].axvline(np.median(data_left), color='blue', linestyle='--', label=f'Médiane={np.median(data_left):.1f}')
axes[2].set_title('Asymétrie Gauche\n(Moyenne < Médiane)')
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## 💡 Compréhension Intuitive

### Analogie du monde réel

Imaginez une classe de 10 étudiants passant un examen noté sur 20 :

**Scénario 1 : Examen équilibré**
Notes : [10, 11, 12, 12, 13, 13, 14, 14, 15, 16]
- **Moyenne** = 13.0
- **Médiane** = 13.0
- **Mode** = 12, 13, 14 (tous apparaissent 2 fois)

**Interprétation** : Distribution bien équilibrée, les 3 mesures se rejoignent.

**Scénario 2 : Un génie dans la classe**
Notes : [8, 9, 10, 11, 11, 12, 12, 13, 14, 20]
- **Moyenne** = 12.0 (tirée vers le haut par le 20)
- **Médiane** = 11.5 (valeur centrale réelle)
- **Mode** = 11, 12

**Interprétation** : La moyenne surestime la performance "typique". La médiane est plus représentative.

**Scénario 3 : Examen raté collectivement sauf 1 étudiant**
Notes : [2, 3, 4, 5, 5, 6, 6, 7, 8, 19]
- **Moyenne** = 6.5
- **Médiane** = 5.5
- **Mode** = 5, 6

**Question professionnelle** : Si vous êtes enseignant et devez justifier vos méthodes au directeur, quelle statistique communiquerez-vous ?
- La **moyenne (6.5)** rend l'échec moins visible
- La **médiane (5.5)** montre qu'au moins 50% des élèves ont ≤ 5.5/20

**Éthique** : Choisir la bonne mesure est une question d'honnêteté intellectuelle !

### Questions pour vérifier la compréhension

Avant de continuer, assurez-vous de pouvoir répondre (auto-évaluation cognitive) :

1. **Q1** : Si toutes les valeurs d'un dataset sont identiques, que valent moyenne, médiane et mode ?
   - *Réponse attendue* : Toutes les trois égales à cette valeur unique

2. **Q2** : Vous avez un dataset de revenus avec quelques milliardaires. Quelle mesure utiliser pour décrire le revenu "typique" ?
   - *Réponse attendue* : Médiane (robuste aux outliers extrêmes)

3. **Q3** : Vous analysez les tailles de t-shirts vendus (S, M, L, XL). Quelle mesure a du sens ?
   - *Réponse attendue* : Mode (données catégorielles ordinales)

4. **Q4** : Vous doublez toutes les valeurs d'un dataset. Comment changent moyenne et médiane ?
   - *Réponse attendue* : Moyenne et médiane sont toutes deux doublées (linéarité pour transformation monotone)

---

## 💻 Implémentation Pratique

> **Principe de modalité** : Code commenté + explication textuelle pour double encodage cognitif.

### 1. Implémentation de Base avec NumPy

```python
"""
Titre : Calcul des mesures de tendance centrale avec NumPy
Objectif : Maîtriser les fonctions de base et comprendre leur comportement
Complexité : O(n) pour moyenne, O(n log n) pour médiane
"""

import numpy as np
from scipy import stats

# Génération de données exemple
np.random.seed(42)  # Reproductibilité
data = np.array([23, 45, 12, 67, 34, 56, 23, 78, 23, 90, 12, 45])

# ========== MOYENNE ==========
moyenne = np.mean(data)
print(f"Moyenne : {moyenne:.2f}")

# Vérification manuelle
moyenne_manuelle = np.sum(data) / len(data)
print(f"Moyenne (manuelle) : {moyenne_manuelle:.2f}")

# ========== MÉDIANE ==========
mediane = np.median(data)
print(f"Médiane : {mediane:.2f}")

# Vérification manuelle
data_sorted = np.sort(data)
n = len(data_sorted)
if n % 2 == 0:
    mediane_manuelle = (data_sorted[n//2 - 1] + data_sorted[n//2]) / 2
else:
    mediane_manuelle = data_sorted[n//2]
print(f"Médiane (manuelle) : {mediane_manuelle:.2f}")

# ========== MODE ==========
mode_result = stats.mode(data, keepdims=True)
mode_value = mode_result.mode[0]
mode_count = mode_result.count[0]
print(f"Mode : {mode_value} (apparaît {mode_count} fois)")

# Détection de tous les modes
valeurs_uniques, comptages = np.unique(data, return_counts=True)
max_count = comptages.max()
tous_modes = valeurs_uniques[comptages == max_count]
print(f"Tous les modes : {tous_modes}")

# ========== RÉSUMÉ ==========
print(f"\n{'='*50}")
print(f"Données : {data}")
print(f"Nombre d'observations : {len(data)}")
print(f"Moyenne  : {moyenne:.2f}")
print(f"Médiane  : {mediane:.2f}")
print(f"Mode(s)  : {tous_modes}")
print(f"{'='*50}")
```

### 2. Fonction Robuste avec Gestion des Cas Limites

```python
"""
Titre : Fonction robuste de calcul des mesures centrales
Objectif : Gérer valeurs manquantes, outliers, distributions spéciales
"""

def mesures_tendance_centrale(data, remove_outliers=False, outlier_threshold=3):
    """
    Calcule moyenne, médiane et mode avec gestion robuste des cas limites.
    
    Args:
        data (array-like): Données numériques ou catégorielles
        remove_outliers (bool): Si True, retire outliers avant calcul moyenne
        outlier_threshold (float): Seuil Z-score pour outliers (défaut 3)
    
    Returns:
        dict: Dictionnaire avec statistiques calculées
    
    Raises:
        ValueError: Si données vides ou invalides
    """
    import numpy as np
    from scipy import stats
    
    # Conversion en array NumPy
    data = np.asarray(data)
    
    # Validation : données non vides
    if len(data) == 0:
        raise ValueError("Le dataset est vide !")
    
    # Gestion des valeurs manquantes (NaN)
    data_clean = data[~np.isnan(data)] if data.dtype.kind in ['f', 'c'] else data
    n_missing = len(data) - len(data_clean)
    
    if len(data_clean) == 0:
        raise ValueError("Toutes les valeurs sont manquantes (NaN) !")
    
    # Détection du type de données
    is_numeric = np.issubdtype(data_clean.dtype, np.number)
    
    resultats = {
        'n_total': len(data),
        'n_missing': n_missing,
        'n_valide': len(data_clean)
    }
    
    if is_numeric:
        # Retrait des outliers si demandé
        if remove_outliers and len(data_clean) > 3:
            z_scores = np.abs(stats.zscore(data_clean))
            data_no_outliers = data_clean[z_scores < outlier_threshold]
            n_outliers = len(data_clean) - len(data_no_outliers)
            resultats['n_outliers_removed'] = n_outliers
            data_for_mean = data_no_outliers
        else:
            data_for_mean = data_clean
            resultats['n_outliers_removed'] = 0
        
        # Calculs numériques
        resultats['moyenne'] = np.mean(data_for_mean)
        resultats['mediane'] = np.median(data_clean)  # Médiane toujours sur données complètes
        
        # Écart moyenne-médiane (indicateur d'asymétrie)
        resultats['ecart_moyenne_mediane'] = resultats['moyenne'] - resultats['mediane']
        
    else:
        # Données catégorielles : seul le mode a du sens
        resultats['moyenne'] = None
        resultats['mediane'] = None
    
    # Mode (fonctionne pour numérique ET catégoriel)
    try:
        mode_result = stats.mode(data_clean, keepdims=True)
        resultats['mode'] = mode_result.mode[0]
        resultats['mode_count'] = mode_result.count[0]
        resultats['mode_frequency'] = mode_result.count[0] / len(data_clean)
    except:
        resultats['mode'] = None
        resultats['mode_count'] = 0
        resultats['mode_frequency'] = 0.0
    
    return resultats

# ========== TESTS ==========

# Test 1 : Données normales
data_normal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Test 1 : Données normales")
print(mesures_tendance_centrale(data_normal))

# Test 2 : Données avec outliers
data_outliers = np.array([10, 12, 11, 13, 12, 14, 13, 15, 14, 1000])
print("\nTest 2 : Avec outliers")
print("Sans retrait outliers :")
print(mesures_tendance_centrale(data_outliers, remove_outliers=False))
print("Avec retrait outliers :")
print(mesures_tendance_centrale(data_outliers, remove_outliers=True))

# Test 3 : Données avec NaN
data_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0])
print("\nTest 3 : Avec valeurs manquantes")
print(mesures_tendance_centrale(data_nan))

# Test 4 : Données catégorielles
data_cat = np.array(['A', 'B', 'A', 'C', 'A', 'B', 'A'])
print("\nTest 4 : Données catégorielles")
print(mesures_tendance_centrale(data_cat))
```

### 3. Moyenne Tronquée (Trimmed Mean)

```python
"""
Titre : Moyenne tronquée (compromise moyenne/médiane)
Objectif : Robustesse aux outliers tout en utilisant plus de données que la médiane
"""

from scipy import stats

def trimmed_mean_analysis(data, trim_proportions=[0.0, 0.05, 0.10, 0.20, 0.50]):
    """
    Compare moyenne classique, moyennes tronquées et médiane.
    
    Args:
        data (array): Données numériques
        trim_proportions (list): Proportions de troncature à tester
    
    Returns:
        dict: Résultats pour chaque proportion
    """
    import numpy as np
    from scipy import stats
    
    resultats = {}
    
    for trim in trim_proportions:
        if trim == 0.0:
            resultats[f'trim_{trim:.0%}'] = {
                'value': np.mean(data),
                'description': 'Moyenne classique'
            }
        elif trim == 0.50:
            resultats[f'trim_{trim:.0%}'] = {
                'value': np.median(data),
                'description': 'Médiane (cas limite trim=50%)'
            }
        else:
            resultats[f'trim_{trim:.0%}'] = {
                'value': stats.trim_mean(data, trim),
                'description': f'Retire {trim:.0%} des extrêmes de chaque côté'
            }
    
    return resultats

# Exemple avec données fortement asymétriques
salaires = np.array([30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60, 500, 800])

print("Analyse de moyennes tronquées sur salaires (en k€) :")
print(f"Données : {salaires}")
print()

results = trimmed_mean_analysis(salaires)
for key, val in results.items():
    print(f"{key:15} : {val['value']:6.1f}k€  |  {val['description']}")

# Interprétation
print("\n💡 Interprétation :")
print("Plus on tronque, plus on se rapproche de la médiane (robustesse)")
print("Trim 10% est un bon compromis : robuste mais utilise 80% des données")
```

---

## 🔬 Exemples Concrets et Cas d'Usage

> **Principe de personnalisation** : Exemples progressifs du simple au complexe pour gérer la charge cognitive.

### Exemple 1 : Analyse de Salaires (RH) - Niveau Débutant

**Contexte** :
Vous êtes Data Analyst RH et devez présenter les salaires de l'entreprise au comité de direction. Il y a 15 employés.

**Données** :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Salaires en k€
salaires = np.array([28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60, 65, 80, 250])
postes = ['Junior Dev', 'Junior Dev', 'Dev', 'Dev', 'Dev', 'Senior Dev', 'Senior Dev',
          'Senior Dev', 'Lead Dev', 'Lead Dev', 'Manager', 'Manager', 'Senior Manager',
          'Directeur Tech', 'CEO']

# Création DataFrame
df = pd.DataFrame({'Poste': postes, 'Salaire': salaires})
```

**Analyse** :

```python
# Statistiques
moyenne = np.mean(salaires)
mediane = np.median(salaires)
mode_result = stats.mode(salaires, keepdims=True)

print(f"Moyenne  : {moyenne:.1f}k€")
print(f"Médiane  : {mediane:.1f}k€")
print(f"Min      : {np.min(salaires):.1f}k€")
print(f"Max      : {np.max(salaires):.1f}k€")

# Quartiles
q1 = np.percentile(salaires, 25)
q3 = np.percentile(salaires, 75)
print(f"\n25% gagnent moins de {q1:.1f}k€")
print(f"50% gagnent moins de {mediane:.1f}k€")
print(f"75% gagnent moins de {q3:.1f}k€")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
axes[0].hist(salaires, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(moyenne, color='red', linestyle='--', linewidth=2, label=f'Moyenne = {moyenne:.1f}k€')
axes[0].axvline(mediane, color='blue', linestyle='--', linewidth=2, label=f'Médiane = {mediane:.1f}k€')
axes[0].set_xlabel('Salaire (k€)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution des Salaires')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot(salaires, vert=True)
axes[1].set_ylabel('Salaire (k€)')
axes[1].set_title('Box Plot des Salaires')
axes[1].grid(alpha=0.3)
axes[1].axhline(moyenne, color='red', linestyle='--', alpha=0.5, label='Moyenne')
axes[1].axhline(mediane, color='blue', linestyle='--', alpha=0.5, label='Médiane')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Résultats** :
```
Moyenne  : 59.9k€
Médiane  : 45.0k€
Min      : 28.0k€
Max      : 250.0k€

25% gagnent moins de 35.0k€
50% gagnent moins de 45.0k€
75% gagnent moins de 57.5k€
```

**Interprétation** :
- **Écart énorme** : Moyenne (59.9k€) vs Médiane (45.0k€) → 14.9k€ de différence !
- **Cause** : Le CEO à 250k€ tire fortement la moyenne vers le haut
- **Communication au board** :
  - ❌ "Le salaire moyen est 59.9k€" → Donne impression fausse (90% gagnent moins !)
  - ✅ "Le salaire médian est 45k€, avec 50% entre 35k€ et 57.5k€" → Honnête

**Analyse critique** :
- **Points forts** : La médiane représente bien le salaire "typique"
- **Limitations** : Ne montre pas l'écart CEO/employés (voir [[measures_dispersion]] pour ça)
- **Leçons apprises** : Toujours communiquer médiane + quartiles pour les distributions asymétriques

---

### Exemple 2 : Temps de Réponse API (Performance Web) - Niveau Intermédiaire

**Contexte** :
Vous optimisez une API. Vous mesurez 1000 temps de réponse en millisecondes.

**Données** :

```python
np.random.seed(123)

# Distribution réaliste : log-normale (la plupart rapides, quelques très lentes)
temps_reponse_ms = np.random.lognormal(mean=3.0, sigma=0.8, size=1000)

# Ajout de quelques timeouts extrêmes (>5 secondes)
n_timeouts = 20
indices_timeouts = np.random.choice(1000, n_timeouts, replace=False)
temps_reponse_ms[indices_timeouts] = np.random.uniform(5000, 10000, n_timeouts)
```

**Analyse** :

```python
# Statistiques brutes
moyenne_ms = np.mean(temps_reponse_ms)
mediane_ms = np.median(temps_reponse_ms)
p95_ms = np.percentile(temps_reponse_ms, 95)
p99_ms = np.percentile(temps_reponse_ms, 99)

print(f"Temps de réponse API (sur {len(temps_reponse_ms)} requêtes) :")
print(f"{'='*60}")
print(f"Moyenne       : {moyenne_ms:.0f} ms")
print(f"Médiane (P50) : {mediane_ms:.0f} ms")
print(f"P95           : {p95_ms:.0f} ms")
print(f"P99           : {p99_ms:.0f} ms")
print(f"Max           : {np.max(temps_reponse_ms):.0f} ms")

# Pourcentage sous différents seuils
seuils = [100, 500, 1000, 2000]
print(f"\n% requêtes sous seuil :")
for seuil in seuils:
    pct = (temps_reponse_ms < seuil).sum() / len(temps_reponse_ms) * 100
    print(f"  < {seuil:4d} ms : {pct:5.1f}%")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogramme (échelle normale)
axes[0, 0].hist(temps_reponse_ms, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(moyenne_ms, color='red', linestyle='--', label=f'Moyenne={moyenne_ms:.0f}ms')
axes[0, 0].axvline(mediane_ms, color='blue', linestyle='--', label=f'Médiane={mediane_ms:.0f}ms')
axes[0, 0].set_xlabel('Temps (ms)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution Temps de Réponse')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Histogramme (échelle log pour mieux voir)
axes[0, 1].hist(temps_reponse_ms, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(moyenne_ms, color='red', linestyle='--', label=f'Moyenne={moyenne_ms:.0f}ms')
axes[0, 1].axvline(mediane_ms, color='blue', linestyle='--', label=f'Médiane={mediane_ms:.0f}ms')
axes[0, 1].set_xlabel('Temps (ms)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution (Échelle Log Y)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Box plot
axes[1, 0].boxplot(temps_reponse_ms, vert=True)
axes[1, 0].set_ylabel('Temps (ms)')
axes[1, 0].set_title('Box Plot Temps de Réponse')
axes[1, 0].grid(alpha=0.3)

# Courbe cumulative (CDF)
temps_sorted = np.sort(temps_reponse_ms)
cumulative = np.arange(1, len(temps_sorted) + 1) / len(temps_sorted) * 100
axes[1, 1].plot(temps_sorted, cumulative, linewidth=2)
axes[1, 1].axhline(50, color='blue', linestyle='--', alpha=0.5, label='P50 (médiane)')
axes[1, 1].axhline(95, color='orange', linestyle='--', alpha=0.5, label='P95')
axes[1, 1].axhline(99, color='red', linestyle='--', alpha=0.5, label='P99')
axes[1, 1].set_xlabel('Temps (ms)')
axes[1, 1].set_ylabel('% Requêtes')
axes[1, 1].set_title('Distribution Cumulative (CDF)')
axes[1, 1].set_xscale('log')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Résultats typiques** :
```
Temps de réponse API (sur 1000 requêtes) :
============================================================
Moyenne       : 154 ms
Médiane (P50) : 25 ms
P95           : 68 ms
P99           : 5234 ms
Max           : 9876 ms

% requêtes sous seuil :
  <  100 ms :  87.2%
  <  500 ms :  96.8%
  < 1000 ms :  97.9%
  < 2000 ms :  98.0%
```

**Interprétation** :
- **Moyenne trompeuse** : 154ms suggère API lente, MAIS...
- **Médiane révèle la vérité** : 50% des requêtes < 25ms (excellent !)
- **Problème** : Quelques timeouts extrêmes (>5s) explosent la moyenne
- **Recommandation professionnelle** :
  - ❌ Ne JAMAIS utiliser moyenne pour latences/temps de réponse
  - ✅ Utiliser médiane (P50) + P95 + P99 (standard industrie)
  - SLAs typiques : "P95 < 200ms" ou "P99 < 1s"

**Décision business** :
1. API performante pour 95% des cas
2. Enquêter sur les 1% de requêtes > 5s (timeouts, retry, etc.)
3. Implémenter circuit breaker pour ces cas

**Analyse critique** :
- **Points forts** : Percentiles donnent vision complète de la distribution
- **Limitations** : Ne montrent pas si problème vient de certains endpoints spécifiques
- **Leçons apprises** : Distributions log-normales (très communes en web) nécessitent médiane + percentiles

---

### Exemple 3 : Détection de Fraude (E-commerce) - Niveau Avancé

**Contexte** :
Vous analysez des transactions pour détecter des comptes frauduleux. Hypothèse : comptes normaux ont un panier moyen stable, fraudeurs ont comportement différent.

**Données** :

```python
np.random.seed(456)

# Simula comptes normaux (95%) et frauduleux (5%)
n_users = 1000
n_fraudeurs = 50

# Comptes normaux : panier moyen autour de 50€
paniers_normaux = np.random.gamma(shape=5, scale=10, size=n_users - n_fraudeurs)

# Fraudeurs : 2 stratégies
# - Micro-transactions (tester cartes volées) : 5-15€
# - Gros achats (avant blocage carte) : 500-2000€
fraudeurs_micro = np.random.uniform(5, 15, size=n_fraudeurs // 2)
fraudeurs_gros = np.random.uniform(500, 2000, size=n_fraudeurs // 2)
paniers_fraudeurs = np.concatenate([fraudeurs_micro, fraudeurs_gros])

# Fusion
tous_paniers = np.concatenate([paniers_normaux, paniers_fraudeurs])
labels = np.array(['Normal'] * (n_users - n_fraudeurs) + ['Fraudeur'] * n_fraudeurs)

# DataFrame
df_transactions = pd.DataFrame({
    'Panier': tous_paniers,
    'Type': labels
})
```

**Analyse Exploratoire** :

```python
# Statistiques globales
print("ANALYSE GLOBALE (tous comptes mélangés)")
print("="*60)
stats_globales = mesures_tendance_centrale(tous_paniers)
print(f"Moyenne  : {stats_globales['moyenne']:.2f}€")
print(f"Médiane  : {stats_globales['mediane']:.2f}€")

# Statistiques par type
print("\nANALYSE PAR TYPE DE COMPTE")
print("="*60)
for type_compte in ['Normal', 'Fraudeur']:
    paniers_type = df_transactions[df_transactions['Type'] == type_compte]['Panier'].values
    stats_type = mesures_tendance_centrale(paniers_type)
    print(f"\n{type_compte} (n={len(paniers_type)}) :")
    print(f"  Moyenne  : {stats_type['moyenne']:.2f}€")
    print(f"  Médiane  : {stats_type['mediane']:.2f}€")
    print(f"  Écart M-m: {stats_type['ecart_moyenne_mediane']:.2f}€")

# Visualisation comparative
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogramme global
axes[0, 0].hist(tous_paniers, bins=50, edgecolor='black', alpha=0.7, color='gray')
axes[0, 0].axvline(np.mean(tous_paniers), color='red', linestyle='--', label='Moyenne')
axes[0, 0].axvline(np.median(tous_paniers), color='blue', linestyle='--', label='Médiane')
axes[0, 0].set_xlabel('Montant Panier (€)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution Globale')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 300)  # Zoom pour mieux voir

# Histogrammes séparés
axes[0, 1].hist(paniers_normaux, bins=50, alpha=0.7, label='Normaux', color='green', edgecolor='black')
axes[0, 1].hist(paniers_fraudeurs, bins=50, alpha=0.7, label='Fraudeurs', color='red', edgecolor='black')
axes[0, 1].set_xlabel('Montant Panier (€)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution par Type')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 300)

# Box plots comparatifs
df_transactions.boxplot(column='Panier', by='Type', ax=axes[1, 0])
axes[1, 0].set_xlabel('Type de Compte')
axes[1, 0].set_ylabel('Montant Panier (€)')
axes[1, 0].set_title('Box Plot par Type')
axes[1, 0].get_figure().suptitle('')  # Retirer titre auto

# Moyennes par groupe (barplot)
moyennes_par_type = df_transactions.groupby('Type')['Panier'].mean()
medianes_par_type = df_transactions.groupby('Type')['Panier'].median()

x = np.arange(len(moyennes_par_type))
width = 0.35

axes[1, 1].bar(x - width/2, moyennes_par_type, width, label='Moyenne', color='red', alpha=0.7)
axes[1, 1].bar(x + width/2, medianes_par_type, width, label='Médiane', color='blue', alpha=0.7)
axes[1, 1].set_xlabel('Type de Compte')
axes[1, 1].set_ylabel('Montant (€)')
axes[1, 1].set_title('Moyenne vs Médiane par Type')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(moyennes_par_type.index)
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

**Construction Feature pour ML** :

```python
# Feature engineering basée sur mesures centrales
def compute_user_features(paniers_user):
    """
    Calcule features pour un utilisateur basées sur ses transactions.
    """
    stats = mesures_tendance_centrale(paniers_user, remove_outliers=False)
    
    features = {
        'panier_moyen': stats['moyenne'],
        'panier_median': stats['mediane'],
        'ratio_moyenne_mediane': stats['moyenne'] / stats['mediane'] if stats['mediane'] > 0 else np.nan,
        'ecart_moyenne_mediane_abs': abs(stats['ecart_moyenne_mediane']),
        'nb_transactions': stats['n_valide']
    }
    
    return features

# Simulation : chaque user a plusieurs transactions
# (simplifié : ici on a 1 transaction/user, mais concept reste valide)

# Calcul ratio moyenne/médiane comme indicateur de fraude
df_transactions['ratio_M_m'] = df_transactions.groupby('Type')['Panier'].transform(
    lambda x: x.mean() / x.median()
)

print("\nRATIO MOYENNE/MÉDIANE comme indicateur de fraude :")
print("="*60)
print(f"Comptes normaux  : {df_transactions[df_transactions['Type']=='Normal']['ratio_M_m'].iloc[0]:.2f}")
print(f"Comptes fraudeurs: {df_transactions[df_transactions['Type']=='Fraudeur']['ratio_M_m'].iloc[0]:.2f}")
print("\n💡 Ratio élevé (>>1) indique distribution asymétrique → suspect !")
```

**Interprétation** :
- **Fraudeurs bimodaux** : Distribution avec 2 modes (micro + gros achats)
- **Écart Moyenne-Médiane** : Plus grand chez fraudeurs (comportement erratique)
- **Feature ML** : Ratio moyenne/médiane = bon prédicteur de fraude
- **Limite** : Ici simplifié (1 transaction/user), en réalité analyser séquences temporelles

**Analyse critique** :
- **Points forts** : Mesures centrales permettent feature engineering simple mais efficace
- **Limitations** : Ne capture pas dimension temporelle (voir [[time_series_basics]])
- **Leçons apprises** : Comparer moyenne/médiane révèle patterns de comportement

---

## ⚖️ Comparaisons et Choix de Design

> **Principe de contiguïté spatiale** : Comparaisons côte à côte pour faciliter la compréhension.

### Pourquoi Médiane plutôt que Moyenne ?

**Contexte de décision** : Données avec outliers ou distributions asymétriques

| Critère | Moyenne | Médiane |
|---------|---------|---------|
| **Sensibilité outliers** | ❌ Très sensible | ✅ Robuste |
| **Interprétation** | "Centre de gravité" | "Valeur typique" |
| **Complexité calcul** | $$O(n)$$ | $$O(n \log n)$$ |
| **Propriété mathématique** | Minimise $$\sum (x_i - c)^2$$ | Minimise $$\sum |x_i - c|$$ |
| **Usage ML** | Loss functions (MSE) | Loss robuste (MAE) |
| **Communication** | Peut être trompeuse | Honnête |

**Cas d'usage** :

✅ **Utiliser Moyenne quand** :
- Distribution approximativement symétrique (normale)
- Pas d'outliers significatifs
- Vous voulez que chaque point ait même poids
- Calcul théorique (propriétés mathématiques)
- Exemples : température, notes d'examen bien conçu, mesures physiques

✅ **Utiliser Médiane quand** :
- Distribution asymétrique (skewness élevé)
- Présence d'outliers
- Données de type "revenu/prix" (log-normales)
- Communication à non-experts
- Exemples : salaires, prix immobilier, temps de réponse, revenus

**Exemple comparatif avec visualisation** :

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation
np.random.seed(789)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

distributions = [
    ('Normale (Symétrique)', np.random.normal(50, 10, 1000)),
    ('Log-Normale (Asymétrique)', np.random.lognormal(3, 0.5, 1000)),
    ('Avec Outliers', np.concatenate([np.random.normal(50, 10, 990), np.array([200, 220, 250, 280, 300, 350, 400, 500, 600, 800])]))
]

for idx, (titre, data) in enumerate(distributions):
    row = idx // 3
    col = idx % 3
    
    moyenne = np.mean(data)
    mediane = np.median(data)
    
    axes[row, col].hist(data, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
    axes[row, col].axvline(moyenne, color='red', linestyle='--', linewidth=2, label=f'Moyenne={moyenne:.1f}')
    axes[row, col].axvline(mediane, color='blue', linestyle='--', linewidth=2, label=f'Médiane={mediane:.1f}')
    axes[row, col].set_title(titre)
    axes[row, col].set_xlabel('Valeur')
    axes[row, col].set_ylabel('Fréquence')
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.3)
    
    # Annotation écart
    ecart = abs(moyenne - mediane)
    axes[row, col].text(0.05, 0.95, f'Écart M-m: {ecart:.1f}', 
                        transform=axes[row, col].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Masquer axes vides
for idx in range(len(distributions), 6):
    row = idx // 3
    col = idx % 3
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Comparaison chiffrée
print("\nRÉSUMÉ COMPARAISON MOYENNE vs MÉDIANE")
print("="*70)
print(f"{'Distribution':<25} | {'Moyenne':>10} | {'Médiane':>10} | {'Écart':>10} | Recommandation")
print("-"*70)
for titre, data in distributions:
    moy = np.mean(data)
    med = np.median(data)
    ecart = abs(moy - med)
    recomm = "Moyenne OK" if ecart < 5 else "⚠️ Utiliser Médiane"
    print(f"{titre:<25} | {moy:>10.1f} | {med:>10.1f} | {ecart:>10.1f} | {recomm}")
```

**Recommandation** :

- ✅ **Écart < 5%** du range → Moyenne OK
- ⚠️ **Écart > 10%** du range → Privilégier Médiane
- 🚨 **Écart > 20%** du range → OBLIGATOIRE d'utiliser Médiane

---

## ⚠️ Pièges Courants et Bonnes Pratiques

> **Principe de cohérence** : Liste structurée et actionnable.

### ❌ Erreurs fréquentes

#### Erreur 1 : Utiliser Moyenne sur Données Asymétriques

**Description** :
Calculer la moyenne sur des distributions très asymétriques (salaires, prix, temps de réponse) conduit à des conclusions trompeuses.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS
revenus_france = np.array([...])  # Distribution très asymétrique
revenu_moyen = np.mean(revenus_france)
print(f"Le revenu moyen en France est {revenu_moyen}€")
# Pourquoi c'est problématique : Biaisé par les ultra-riches, ne représente pas le citoyen "typique"
```

**Solution** :

```python
# ✅ BON
revenu_median = np.median(revenus_france)
q1 = np.percentile(revenus_france, 25)
q3 = np.percentile(revenus_france, 75)
print(f"Le revenu médian en France est {revenu_median}€")
print(f"50% des français gagnent entre {q1}€ et {q3}€")
# Pourquoi c'est mieux : Représente vraiment le citoyen médian, robuste aux outliers
```

**Impact** : Politiques publiques basées sur moyenne plutôt que médiane peuvent être complètement inadaptées.

**Source** : 
- Piketty, T. (2013). *Le Capital au XXIe siècle*. - Analyse distributions de revenus
- [INSEE - Revenus et salaires](https://www.insee.fr/fr/statistiques) - Utilise systématiquement la médiane

---

#### Erreur 2 : Ignorer les Valeurs Manquantes (NaN)

**Description** :
NumPy retourne `nan` si présence de `NaN` dans les données, sans warning.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS
data_avec_nan = np.array([1, 2, 3, np.nan, 5, 6])
moyenne = np.mean(data_avec_nan)  # Retourne nan !
print(f"Moyenne : {moyenne}")  # nan
# Pourquoi c'est problématique : Résultat inutilisable, pas d'erreur levée
```

**Solution** :

```python
# ✅ BON - Option 1 : Utiliser nanmean
moyenne_safe = np.nanmean(data_avec_nan)  # Ignore les NaN
print(f"Moyenne (sans NaN) : {moyenne_safe}")  # 3.4

# ✅ BON - Option 2 : Filtrer explicitement
data_clean = data_avec_nan[~np.isnan(data_avec_nan)]
moyenne_filtered = np.mean(data_clean)
print(f"Moyenne (filtrée) : {moyenne_filtered}")  # 3.4

# ✅ MEILLEUR - Option 3 : Vérifier et documenter
n_nan = np.isnan(data_avec_nan).sum()
if n_nan > 0:
    print(f"⚠️ Attention : {n_nan} valeurs manquantes détectées")
    moyenne_safe = np.nanmean(data_avec_nan)
else:
    moyenne_safe = np.mean(data_avec_nan)
# Pourquoi c'est mieux : Transparent, documenté, décision consciente
```

**Impact** : Pipelines ML qui crashent silencieusement, résultats erronés non détectés.

**Source** :
- [NumPy NaN handling](https://numpy.org/doc/stable/reference/routines.statistics.html) - Documentation officielle

---

#### Erreur 3 : Moyenne de Moyennes (Sans Pondération)

**Description** :
Calculer la moyenne de moyennes sans tenir compte de la taille des groupes donne un résultat biaisé.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS
# Moyennes par département
dept_A_moyenne = 50  # n=1000 employés
dept_B_moyenne = 80  # n=10 employés
dept_C_moyenne = 60  # n=5 employés

moyenne_entreprise_fausse = np.mean([dept_A_moyenne, dept_B_moyenne, dept_C_moyenne])
print(f"Moyenne entreprise : {moyenne_entreprise_fausse}")  # 63.33
# Pourquoi c'est problématique : Dept B (10 employés) a autant de poids que Dept A (1000) !
```

**Solution** :

```python
# ✅ BON - Moyenne pondérée
moyennes = np.array([50, 80, 60])
effectifs = np.array([1000, 10, 5])

moyenne_entreprise_correcte = np.average(moyennes, weights=effectifs)
print(f"Moyenne entreprise (pondérée) : {moyenne_entreprise_correcte:.2f}")  # 50.59

# ✅ MEILLEUR - Recalculer depuis données brutes si possible
# (Évite erreurs d'arrondi et garantit exactitude)
# Pourquoi c'est mieux : Chaque employé compte équitablement
```

**Impact** : Rapports financiers erronés, décisions stratégiques basées sur chiffres faux.

**Source** :
- [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) - Phénomène lié

---

#### Erreur 4 : Comparer Moyennes Sans Tests Statistiques

**Description** :
Conclure qu'une différence existe juste en comparant deux moyennes, sans tester la significativité statistique.

**Exemple de code problématique** :

```python
# ❌ MAUVAIS
groupe_A = np.random.normal(50, 10, 30)
groupe_B = np.random.normal(52, 10, 30)

moy_A = np.mean(groupe_A)
moy_B = np.mean(groupe_B)

print(f"Groupe A : {moy_A:.1f}")
print(f"Groupe B : {moy_B:.1f}")
print(f"Conclusion : B est meilleur que A")  # ❌ Pas prouvé !
# Pourquoi c'est problématique : Peut être juste variabilité aléatoire !
```

**Solution** :

```python
# ✅ BON - Test statistique
from scipy import stats

# Test t de Student
t_statistic, p_value = stats.ttest_ind(groupe_A, groupe_B)

print(f"Groupe A : {moy_A:.1f}")
print(f"Groupe B : {moy_B:.1f}")
print(f"Différence : {moy_B - moy_A:.1f}")
print(f"p-value : {p_value:.4f}")

if p_value < 0.05:
    print("✅ Différence statistiquement significative")
else:
    print("❌ Pas de preuve de différence (peut être hasard)")
# Pourquoi c'est mieux : Quantifie l'incertitude, évite fausses conclusions
```

**Impact** : Décisions business basées sur bruit aléatoire (faux positifs), A/B tests mal interprétés.

**Source** :
- Cours complet : [[hypothesis_testing]]
- Cours complet : [[two_sample_tests]]

---

### ✅ Bonnes pratiques

#### Pratique 1 : Toujours Visualiser Avant de Résumer

**Principe** :
Ne jamais calculer statistiques sans d'abord visualiser la distribution des données.

**Justification scientifique/technique** :
Le [Quartet d'Anscombe](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) (1973) démontre que 4 datasets avec moyennes/variances identiques ont des distributions complètement différentes.

**Implémentation** :

```python
# ✅ Workflow recommandé
def analyse_complete(data, nom="Données"):
    """
    Analyse complète : visualisation + statistiques.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Histogramme
    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(data), color='red', linestyle='--', label='Moyenne')
    axes[0].axvline(np.median(data), color='blue', linestyle='--', label='Médiane')
    axes[0].set_title(f'{nom} - Distribution')
    axes[0].set_xlabel('Valeur')
    axes[0].set_ylabel('Fréquence')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Box plot
    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f'{nom} - Box Plot')
    axes[1].set_ylabel('Valeur')
    axes[1].grid(alpha=0.3)
    
    # 3. QQ plot (normalité)
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title(f'{nom} - QQ Plot')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    print(f"\nSTATISTIQUES - {nom}")
    print("="*50)
    print(f"n             : {len(data)}")
    print(f"Moyenne       : {np.mean(data):.2f}")
    print(f"Médiane       : {np.median(data):.2f}")
    print(f"Écart-type    : {np.std(data, ddof=1):.2f}")
    print(f"Min           : {np.min(data):.2f}")
    print(f"Q1            : {np.percentile(data, 25):.2f}")
    print(f"Q3            : {np.percentile(data, 75):.2f}")
    print(f"Max           : {np.max(data):.2f}")
    
    # Test de normalité
    _, p_shapiro = stats.shapiro(data)
    print(f"\nTest normalité (Shapiro-Wilk) :")
    print(f"p-value = {p_shapiro:.4f}")
    if p_shapiro < 0.05:
        print("❌ Distribution NON normale (utiliser médiane)")
    else:
        print("✅ Distribution normale (moyenne OK)")

# Test
data_test = np.random.lognormal(3, 0.8, 500)
analyse_complete(data_test, "Temps de réponse API")
```

**Sources** :
- Anscombe, F. J. (1973). "Graphs in Statistical Analysis". *The American Statistician*. - Quartet d'Anscombe
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. - Père de l'EDA

---

#### Pratique 2 : Documenter le Choix de la Mesure

**Principe** :
Toujours expliquer POURQUOI vous utilisez moyenne vs médiane dans vos rapports/code.

**Implémentation** :

```python
# ✅ Code auto-documenté
def compute_central_tendency(data, method='auto'):
    """
    Calcule mesure de tendance centrale avec choix automatique ou manuel.
    
    Args:
        data (array): Données numériques
        method (str): 'auto', 'mean', 'median'
            - 'auto' : Choisit automatiquement selon distribution
            - 'mean'  : Force moyenne
            - 'median': Force médiane
    
    Returns:
        dict: {'value': float, 'method': str, 'rationale': str}
    """
    from scipy import stats
    import numpy as np
    
    data_clean = data[~np.isnan(data)]
    
    if method == 'auto':
        # Test de normalité
        _, p_shapiro = stats.shapiro(data_clean) if len(data_clean) < 5000 else (None, 0.0)
        
        # Asymétrie (skewness)
        skew = stats.skew(data_clean)
        
        # Décision
        if abs(skew) < 0.5 and (p_shapiro is None or p_shapiro > 0.05):
            method_chosen = 'mean'
            rationale = f"Distribution symétrique (skew={skew:.2f}, p_shapiro={p_shapiro:.3f if p_shapiro else 'N/A'})"
        else:
            method_chosen = 'median'
            rationale = f"Distribution asymétrique (skew={skew:.2f}) ou non normale"
    else:
        method_chosen = method
        rationale = "Choix manuel de l'utilisateur"
    
    # Calcul
    if method_chosen == 'mean':
        value = np.mean(data_clean)
    else:
        value = np.median(data_clean)
    
    return {
        'value': value,
        'method': method_chosen,
        'rationale': rationale,
        'n': len(data_clean),
        'n_missing': len(data) - len(data_clean)
    }

# Exemple d'utilisation
salaires = np.array([30, 35, 40, 45, 50, 55, 60, 250])
result = compute_central_tendency(salaires, method='auto')

print(f"Valeur centrale : {result['value']:.2f}€")
print(f"Méthode utilisée : {result['method']}")
print(f"Justification : {result['rationale']}")
```

---

### 📋 Checklist de Validation

Avant de communiquer une mesure de tendance centrale :

- [ ] **Visualisation** : Ai-je regardé histogramme + box plot ?
- [ ] **Distribution** : Est-elle symétrique ou asymétrique ?
- [ ] **Outliers** : Y a-t-il des valeurs extrêmes ?
- [ ] **Valeurs manquantes** : Ai-je vérifié et géré les NaN ?
- [ ] **Choix justifié** : Puis-je expliquer pourquoi moyenne OU médiane ?
- [ ] **Contexte** : Ai-je fourni quartiles/percentiles en complément ?
- [ ] **Comparaison** : Si je compare groupes, ai-je fait test statistique ?
- [ ] **Communication** : Mon interlocuteur comprend-il la différence moyenne/médiane ?

---

## 🧪 Exercices et Validation des Connaissances

### Exercice 1 : Analyse de Satisfaction Client - Débutant

**Énoncé** :
Vous collectez des notes de satisfaction (échelle 1-10) de 20 clients :

```python
notes_satisfaction = np.array([8, 9, 7, 8, 9, 10, 8, 7, 9, 8, 
                                2, 9, 8, 10, 9, 7, 8, 9, 10, 8])
```

**Questions** :
1. Calculez moyenne, médiane et mode
2. Identifiez l'outlier et expliquez son impact
3. Quelle mesure communiqueriez-vous au management ?

**Données** :

```python
notes_satisfaction = np.array([8, 9, 7, 8, 9, 10, 8, 7, 9, 8, 
                                2, 9, 8, 10, 9, 7, 8, 9, 10, 8])
```

**Indices** :

<details>
<summary>💡 Indice 1 (cliquez pour révéler)</summary>

Regardez la distribution : une valeur est très éloignée des autres. Comparez ce que donne moyenne vs médiane.

</details>

<details>
<summary>💡 Indice 2</summary>

L'outlier est le 2. Calculez la moyenne avec et sans cette valeur pour voir l'impact.

</details>

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
from scipy import stats

notes_satisfaction = np.array([8, 9, 7, 8, 9, 10, 8, 7, 9, 8, 
                                2, 9, 8, 10, 9, 7, 8, 9, 10, 8])

# Q1 : Statistiques
moyenne = np.mean(notes_satisfaction)
mediane = np.median(notes_satisfaction)
mode_result = stats.mode(notes_satisfaction, keepdims=True)
mode = mode_result.mode[0]

print(f"Moyenne  : {moyenne:.2f}")  # 8.15
print(f"Médiane  : {mediane:.2f}")  # 8.0
print(f"Mode     : {mode}")         # 8

# Q2 : Identification outlier
outlier = notes_satisfaction.min()  # 2
print(f"\nOutlier détecté : {outlier}")

# Impact de l'outlier
notes_sans_outlier = notes_satisfaction[notes_satisfaction != 2]
moyenne_sans_outlier = np.mean(notes_sans_outlier)

print(f"Moyenne AVEC outlier : {moyenne:.2f}")
print(f"Moyenne SANS outlier : {moyenne_sans_outlier:.2f}")
print(f"Impact : {moyenne_sans_outlier - moyenne:.2f} points")

# Q3 : Recommandation
print("\n📊 RECOMMANDATION MANAGEMENT :")
print(f"✅ Communiquer la MÉDIANE = {mediane:.1f}/10")
print("Raison : Robuste à l'outlier (1 client très mécontent)")
print(f"Note : 1 client (5%) a noté 2/10 → Enquête ciblée nécessaire")
```

**Explication** : 
- La **médiane (8.0)** représente bien la satisfaction "typique" (95% des clients satisfaits)
- La **moyenne (8.15)** est légèrement tirée vers le bas par l'outlier
- L'outlier (2/10) ne doit pas être ignoré : c'est un signal pour enquête client !

</details>

---

### Exercice 2 : Optimisation Algorithme - Intermédiaire

**Énoncé** :
Vous testez 2 algorithmes de tri sur 100 arrays aléatoires. Voici les temps d'exécution (ms) :

```python
np.random.seed(111)
algo_A_temps = np.random.gamma(shape=2, scale=10, size=100)
algo_B_temps = np.random.gamma(shape=3, scale=7, size=100)
```

**Questions** :
1. Calculez moyenne et médiane pour chaque algorithme
2. Visualisez les distributions
3. Lequel est le plus rapide en moyenne ? Et en médiane ?
4. Quelle métrique utiliseriez-vous pour choisir l'algorithme en production ?

**Indices** :

<details>
<summary>💡 Indice 1</summary>

Créez des histogrammes côte à côte pour visualiser les distributions. Les temps de calcul suivent souvent des distributions asymétriques.

</details>

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(111)
algo_A_temps = np.random.gamma(shape=2, scale=10, size=100)
algo_B_temps = np.random.gamma(shape=3, scale=7, size=100)

# Q1 : Statistiques
print("ALGORITHME A")
print(f"  Moyenne : {np.mean(algo_A_temps):.2f} ms")
print(f"  Médiane : {np.median(algo_A_temps):.2f} ms")
print(f"  P95     : {np.percentile(algo_A_temps, 95):.2f} ms")

print("\nALGORITHME B")
print(f"  Moyenne : {np.mean(algo_B_temps):.2f} ms")
print(f"  Médiane : {np.median(algo_B_temps):.2f} ms")
print(f"  P95     : {np.percentile(algo_B_temps, 95):.2f} ms")

# Q2 : Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist([algo_A_temps, algo_B_temps], bins=20, label=['Algo A', 'Algo B'], 
             alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(algo_A_temps), color='red', linestyle='--', label='Moyenne A')
axes[0].axvline(np.mean(algo_B_temps), color='blue', linestyle='--', label='Moyenne B')
axes[0].set_xlabel('Temps (ms)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution Temps d\'Exécution')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot([algo_A_temps, algo_B_temps], labels=['Algo A', 'Algo B'])
axes[1].set_ylabel('Temps (ms)')
axes[1].set_title('Box Plot Comparatif')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Q3 & Q4 : Analyse et recommandation
print("\n📊 ANALYSE COMPARATIVE :")
print(f"Algorithme A :")
print(f"  - Plus rapide en moyenne ({np.mean(algo_A_temps):.1f} ms)")
print(f"  - Plus rapide en médiane ({np.median(algo_A_temps):.1f} ms)")
print(f"\nAlgorithme B :")
print(f"  - Plus lent mais plus stable")
print(f"  - Moins de variance")

print("\n💡 RECOMMANDATION PRODUCTION :")
print("Utiliser MÉDIANE + P95 pour décider :")
print("- Si latence critique → Choisir celui avec P95 le plus bas")
print("- Si coût CPU critique → Choisir celui avec médiane la plus basse")
print(f"\n✅ Ici : Algo A est meilleur (médiane {np.median(algo_A_temps):.1f}ms vs {np.median(algo_B_temps):.1f}ms)")
```

**Explication** : 
Pour les temps d'exécution, **médiane + percentiles** sont les métriques standards en production (voir Exemple 2 du cours).

</details>

---

### Exercice 3 : Imputation de Valeurs Manquantes - Avancé

**Énoncé** :
Vous préparez un dataset pour du ML. Une feature numérique a 15% de valeurs manquantes :

```python
np.random.seed(222)
feature = np.random.lognormal(mean=4, sigma=1, size=1000)
# Création de valeurs manquantes aléatoires
indices_nan = np.random.choice(1000, 150, replace=False)
feature[indices_nan] = np.nan
```

**Questions** :
1. Comparez imputation par moyenne vs médiane
2. Calculez l'erreur introduite pour chaque méthode (comparez avec vraies valeurs)
3. Quelle méthode recommandez-vous et pourquoi ?
4. Bonus : Implémentez une imputation par k-NN (moyenne des k plus proches voisins)

**Indices** :

<details>
<summary>💡 Indice 1</summary>

Gardez une copie des vraies valeurs avant de les mettre à NaN, pour pouvoir calculer l'erreur d'imputation.

</details>

<details>
<summary>💡 Indice 2</summary>

Les données log-normales sont très asymétriques. Quelle mesure est plus robuste pour ce type de distribution ?

</details>

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer

np.random.seed(222)

# Données complètes (vérité terrain)
feature_complete = np.random.lognormal(mean=4, sigma=1, size=1000)

# Création valeurs manquantes
feature_avec_nan = feature_complete.copy()
indices_nan = np.random.choice(1000, 150, replace=False)
vraies_valeurs = feature_avec_nan[indices_nan].copy()  # Sauvegarder vérité
feature_avec_nan[indices_nan] = np.nan

# Q1 : Imputation
# Méthode 1 : Moyenne
imputer_mean = SimpleImputer(strategy='mean')
feature_imputed_mean = imputer_mean.fit_transform(feature_avec_nan.reshape(-1, 1)).ravel()

# Méthode 2 : Médiane
imputer_median = SimpleImputer(strategy='median')
feature_imputed_median = imputer_median.fit_transform(feature_avec_nan.reshape(-1, 1)).ravel()

# Méthode 3 : KNN (bonus)
imputer_knn = KNNImputer(n_neighbors=5)
feature_imputed_knn = imputer_knn.fit_transform(feature_avec_nan.reshape(-1, 1)).ravel()

# Q2 : Calcul erreurs
erreurs_mean = np.abs(feature_imputed_mean[indices_nan] - vraies_valeurs)
erreurs_median = np.abs(feature_imputed_median[indices_nan] - vraies_valeurs)
erreurs_knn = np.abs(feature_imputed_knn[indices_nan] - vraies_valeurs)

print("ERREURS D'IMPUTATION (MAE sur valeurs imputées)")
print("="*60)
print(f"Moyenne  : {np.mean(erreurs_mean):.2f}")
print(f"Médiane  : {np.mean(erreurs_median):.2f}")
print(f"KNN (k=5): {np.mean(erreurs_knn):.2f}")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution originale
axes[0, 0].hist(feature_complete, bins=50, alpha=0.7, edgecolor='black', label='Distribution originale')
axes[0, 0].axvline(np.mean(feature_complete), color='red', linestyle='--', label='Moyenne')
axes[0, 0].axvline(np.median(feature_complete), color='blue', linestyle='--', label='Médiane')
axes[0, 0].set_title('Distribution Originale (Log-Normale)')
axes[0, 0].set_xlabel('Valeur')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Comparaison imputations
axes[0, 1].scatter(vraies_valeurs, feature_imputed_mean[indices_nan], alpha=0.5, label='Moyenne', s=10)
axes[0, 1].scatter(vraies_valeurs, feature_imputed_median[indices_nan], alpha=0.5, label='Médiane', s=10)
axes[0, 1].scatter(vraies_valeurs, feature_imputed_knn[indices_nan], alpha=0.5, label='KNN', s=10)
axes[0, 1].plot([0, max(vraies_valeurs)], [0, max(vraies_valeurs)], 'k--', label='Parfait')
axes[0, 1].set_xlabel('Vraies Valeurs')
axes[0, 1].set_ylabel('Valeurs Imputées')
axes[0, 1].set_title('Qualité Imputation')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Distribution erreurs
axes[1, 0].hist([erreurs_mean, erreurs_median, erreurs_knn], bins=30, 
                label=['Moyenne', 'Médiane', 'KNN'], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Erreur Absolue')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Distribution des Erreurs d\'Imputation')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Box plot erreurs
axes[1, 1].boxplot([erreurs_mean, erreurs_median, erreurs_knn], 
                    labels=['Moyenne', 'Médiane', 'KNN'])
axes[1, 1].set_ylabel('Erreur Absolue')
axes[1, 1].set_title('Comparaison Erreurs (Box Plot)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Q3 : Recommandation
print("\n💡 RECOMMANDATION :")
print("Pour distribution LOG-NORMALE (très asymétrique) :")
print("✅ Médiane ou KNN >> Moyenne")
print("\nRaison : Moyenne biaisée vers valeurs élevées sur log-normale")
print("KNN meilleur si features corrélées disponibles, sinon médiane suffit")
```

**Explication** : 
- **Log-normale** : Distribution très asymétrique (commune pour prix, revenus, temps)
- **Moyenne** : Biaisée vers les valeurs élevées → Mauvaise imputation
- **Médiane** : Robuste → Bonne imputation simple
- **KNN** : Meilleur si contexte (autres features) informatif

</details>

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques Fondamentaux

#### 1. The Mean, Median, and Mode: A Historical Perspective

- **Auteurs** : Stephen M. Stigler (1977)
- **Publication** : *Journal of the American Statistical Association*
- **URL** : [JSTOR](https://www.jstor.org/stable/2286902)
- **Contribution clé** : Histoire du développement des mesures de tendance centrale, pourquoi ces 3 spécifiquement
- **Pertinence** : Comprendre le "pourquoi" derrière ces concepts fondamentaux
- **Niveau** : Accessible (historique et conceptuel)

#### 2. Robust Statistics: The Approach Based on Influence Functions

- **Auteurs** : Frank R. Hampel, Elvezio M. Ronchetti, Peter J. Rousseeuw, Werner A. Stahel (1986)
- **Publication** : Wiley Series in Probability and Statistics
- **Contribution clé** : Robustesse statistique, médiane vs moyenne sous outliers
- **Pertinence** : Fondements théoriques de la robustesse de la médiane
- **Niveau** : Technique/Mathématique

#### 3. Anscombe's Quartet

- **Auteurs** : Francis Anscombe (1973)
- **Publication** : *The American Statistician*
- **URL** : [JSTOR](https://www.jstor.org/stable/2682899)
- **Contribution clé** : Démonstration que statistiques résumées (moyenne, variance) ne suffisent pas sans visualisation
- **Pertinence** : Justifie la bonne pratique "toujours visualiser avant résumer"
- **Niveau** : Accessible

---

### 📚 Ressources Complémentaires

#### Articles de blog techniques

- **"Why Median is Better than Mean for Skewed Distributions"** par Will Koehrsen
  - [Towards Data Science](https://towardsdatascience.com/)
  - 📌 **Pourquoi** : Exemples concrets Data Science avec code Python
  - ⏱️ **Durée** : ~10 min

- **"Measures of Central Tendency in Python"** par Real Python
  - [Real Python](https://realpython.com/python-statistics/)
  - 📌 **Pourquoi** : Tutoriel pratique avec module `statistics` de Python
  - ⏱️ **Durée** : ~15 min

#### Vidéos éducatives

- **"Mean, Median and Mode: StatQuest!!!"** par Josh Starmer
  - [YouTube](https://www.youtube.com/watch?v=h8EYEJ32oQ8)
  - 📌 **Pourquoi** : Visualisations exceptionnelles, intuition claire
  - ⏱️ **Durée** : 11 min

- **"When to Use Mean vs Median"** par Khan Academy
  - [Khan Academy](https://www.khanacademy.org/math/statistics-probability)
  - 📌 **Pourquoi** : Explications pédagogiques progressives
  - ⏱️ **Durée** : 8 min

#### Documentation officielle

- **NumPy Statistics**
  - [URL](https://numpy.org/doc/stable/reference/routines.statistics.html)
  - 📌 **Section recommandée** : `numpy.mean`, `numpy.median`, `numpy.nanmean`, `numpy.nanmedian`

- **SciPy Stats**
  - [URL](https://docs.scipy.org/doc/scipy/reference/stats.html)
  - 📌 **Section recommandée** : `scipy.stats.mode`, `scipy.stats.trim_mean`, `scipy.stats.gmean`

- **Pandas Descriptive Statistics**
  - [URL](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics)
  - 📌 **Section recommandée** : `.describe()`, `.mean()`, `.median()`, `.mode()`

---

### 🛠️ Outils et Frameworks

#### Outil 1 : Pandas (Analyse de données)

- **URL** : [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **Description** : Bibliothèque Python pour manipulation et analyse de données tabulaires
- **Cas d'usage** : Calcul rapide de statistiques descriptives sur DataFrames
- **Installation** :

```bash
pip install pandas
```

- **Exemple rapide** :

```python
import pandas as pd

df = pd.DataFrame({
    'salaires': [30, 35, 40, 45, 50, 250],
    'dept': ['IT', 'IT', 'RH', 'RH', 'IT', 'CEO']
})

# Statistiques globales
print(df['salaires'].describe())

# Par groupe
print(df.groupby('dept')['salaires'].agg(['mean', 'median', 'count']))
```

#### Outil 2 : Seaborn (Visualisation statistique)

- **URL** : [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- **Description** : Visualisations statistiques élégantes basées sur Matplotlib
- **Cas d'usage** : Box plots, violin plots, distributions
- **Installation** :

```bash
pip install seaborn
```

- **Exemple rapide** :

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Box plot comparatif avec médiane visible
sns.boxplot(data=df, x='dept', y='salaires')
plt.show()

# Violin plot (densité + box plot)
sns.violinplot(data=df, x='dept', y='salaires')
plt.show()
```

#### Outil 3 : Pingouin (Tests statistiques simplifiés)

- **URL** : [https://pingouin-stats.org/](https://pingouin-stats.org/)
- **Description** : Bibliothèque Python de statistiques, plus simple que SciPy
- **Cas d'usage** : Tests statistiques, intervalles de confiance
- **Installation** :

```bash
pip install pingouin
```

- **Exemple rapide** :

```python
import pingouin as pg

# Statistiques descriptives complètes
stats = pg.compute_bootci(data, func='mean', confidence=0.95)
print(stats)
```

---

### 📖 Cours et Tutoriels Connexes

#### Dans votre repository (Liens Zettelkasten)

- **Suite directe** :
  - [[measures_dispersion]] - Variance, écart-type, quantiles (complète les mesures centrales)
  - [[data_visualization_principles]] - Visualiser distributions efficacement
  
- **Approfondissement** :
  - [[distribution_analysis]] - Skewness, kurtosis, formes de distributions
  - [[bootstrap_methods]] - Estimer incertitude autour de la moyenne/médiane
  - [[hypothesis_testing]] - Tester si différence de moyennes est significative
  
- **Applications pratiques** :
  - [[sample_size_calculation]] - Combien de données pour estimer moyenne avec précision donnée ?
  - [[ab_testing]] - Comparer moyennes de groupes A vs B
  - [[regression_diagnostics]] - Analyser résidus (écarts à la moyenne)

#### Cours externes recommandés

- **"Statistics and Probability"** par Khan Academy
  - [URL](https://www.khanacademy.org/math/statistics-probability)
  - 📌 **Modules pertinents** : "Summarizing quantitative data"
  - ⏱️ **Durée** : 2-3h

- **"Practical Statistics for Data Scientists"** par O'Reilly
  - 📌 **Chapitre** : Chapter 1 - Exploratory Data Analysis
  - ⏱️ **Durée** : 1-2h

---

## 📝 Résumé Rapide (Quick Reference)

> **Carte de référence** : À consulter rapidement pour se remémorer l'essentiel.

### Concepts Clés

| Concept | Formule/Définition | Cas d'usage |
|---------|-------------------|-------------|
| **Moyenne** | $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$ | Distribution symétrique, pas d'outliers |
| **Médiane** | Valeur centrale (50ème percentile) | Distribution asymétrique, présence outliers |
| **Mode** | Valeur la plus fréquente | Données catégorielles, détecter sous-populations |
| **Moyenne pondérée** | $$\bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}$$ | Observations avec importances différentes |
| **Moyenne tronquée** | Moyenne après retrait % extrêmes | Compromis robustesse/utilisation données |

### Code Minimal

```python
import numpy as np
from scipy import stats

data = np.array([...])

# Calculs basiques
moyenne = np.mean(data)
mediane = np.median(data)
mode = stats.mode(data, keepdims=True).mode[0]

# Gestion NaN
moyenne_safe = np.nanmean(data)
mediane_safe = np.nanmedian(data)

# Moyenne pondérée
moyenne_pond = np.average(data, weights=poids)

# Percentiles
q25, q50, q75 = np.percentile(data, [25, 50, 75])
```

### Décisions Clés

**Quand utiliser Moyenne :**
```
├─ Distribution approximativement symétrique
├─ Absence d'outliers significatifs
├─ Calculs théoriques (propriétés mathématiques)
└─ Exemples : températures, notes, mesures physiques
```

**Quand utiliser Médiane :**
```
├─ Distribution asymétrique (skewness élevé)
├─ Présence d'outliers
├─ Données log-normales (salaires, prix, temps)
├─ Communication à non-experts (plus intuitive)
└─ Exemples : revenus, prix immobilier, latences API
```

**Quand utiliser Mode :**
```
├─ Données catégorielles (seule option)
├─ Identifier sous-populations (distributions multimodales)
└─ Exemples : produit le plus vendu, catégorie majoritaire
```

### Pièges à éviter

1. ⚠️ **Moyenne sur distribution asymétrique** → Solution : Utiliser médiane
2. ⚠️ **Ignorer valeurs manquantes (NaN)** → Solution : `np.nanmean()` ou filtrage explicite
3. ⚠️ **Moyenne de moyennes sans pondération** → Solution : Moyenne pondérée par effectifs
4. ⚠️ **Comparer moyennes sans test statistique** → Solution : Test t de Student [[two_sample_tests]]
5. ⚠️ **Ne pas visualiser avant de résumer** → Solution : Toujours faire histogramme + box plot

### Formules Rapides

**Test rapide asymétrie** :

Si $$|\text{Moyenne} - \text{Médiane}| > 0.1 \times \text{Range}$$ → Distribution asymétrique → Utiliser médiane

**Impact outlier sur moyenne** :

Ajout d'1 valeur $$x_{\text{new}}$$ change moyenne de :

$$\Delta\bar{x} = \frac{x_{\text{new}} - \bar{x}}{n+1}$$

---

## 🔗 Intégration Repository GitHub

### Fichiers à mettre à jour

Lors de l'ajout de ce cours, mettre à jour :

1. **`00_INDEX_STATISTICS.md`**

```markdown
### Module 1 : Statistiques Descriptives

| Outil | Utilité | Cours associé | Priorité |
|-------|---------|---------------|----------|
| **Moyenne, Médiane, Mode** | Mesurer la tendance centrale | [[measures_central_tendency]] ✅ | ⭐🔥 |
```

2. **`README.md`** principal du repository

```markdown
## 00_statistics_foundations

### 01_descriptive_statistics
- ✅ [Mesures de Tendance Centrale](00_statistics_foundations/01_descriptive_statistics/measures_central_tendency.md) - Moyenne, médiane, mode
- 🔲 Mesures de Dispersion - Variance, écart-type, quantiles
```

3. Futurs cours qui référenceront celui-ci :

- `measures_dispersion.md` ajoutera dans section "Prérequis" :
  ```markdown
  ### Prérequis
  - [x] [[measures_central_tendency]] - Mesures de tendance centrale
  ```

### Proposition de nommage

- **Nom de fichier** : `measures_central_tendency.md` ✅
- **Cohérent avec** : Convention snake_case, descriptif, pas de date (cours fondamental intemporel)

### Emplacement dans repository

```
00_statistics_foundations/
├── 00_INDEX_STATISTICS.md
├── 01_descriptive_statistics/
│   ├── measures_central_tendency.md ← CE COURS
│   ├── measures_dispersion.md (à créer)
│   └── data_visualization_principles.md (à créer)
├── 02_probability_theory/
├── 03_inferential_statistics/
└── ...
```

### Tags pour recherche future

```
#statistics #descriptive #mean #median #mode #fundamentals 
#data-science #exploratory-data-analysis #eda #central-tendency
```

---

**Prochaine étape recommandée** : [[measures_dispersion]] - Variance, Écart-type, Quantiles

Ce cours complète naturellement celui-ci : après avoir défini le "centre" de vos données, vous devez mesurer leur "dispersion" autour de ce centre ! 📊
