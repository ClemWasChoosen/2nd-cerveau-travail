# 🎲 Variables Aléatoires et Espérance

> **Résumé en une phrase** : Les variables aléatoires transforment les résultats d'expériences aléatoires en valeurs numériques manipulables mathématiquement, permettant de calculer espérances, variances et de modéliser des phénomènes incertains.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-20 |
| **Dernière mise à jour** | 2026-03-20 |
| **Domaine** | Théorie des Probabilités |
| **Niveau** | Intermédiaire |
| **Durée de lecture** | ~55 minutes |
| **Fichier** | `random_variables.md` |
| **Emplacement** | `/00_statistics_foundations/02_probability_theory/` |
| **Tags** | `#probability` `#random-variables` `#expectation` `#variance` `#PMF` `#PDF` `#CDF` |

### Prérequis

- [x] [[probability_foundations]] - Axiomes, probabilités conditionnelles, Bayes (ESSENTIEL)
- [x] [[measures_central_tendency]] - Moyenne (pour comprendre espérance)
- [x] [[measures_dispersion]] - Variance (pour variance de VA)
- [ ] Calcul intégral de base (pour VA continues)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[probability_foundations]] - Fondements probabilités
- **Complémentaires** : 
  - [[data_visualization_principles]] - Visualiser distributions
- **Suite recommandée** : 
  - [[common_distributions]] - Lois de probabilité classiques
  - [[joint_distributions]] - Variables aléatoires multiples
  - [[law_large_numbers]] - Théorèmes limites
  - [[central_limit_theorem]] - Théorème central limite

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Jusqu'ici, nous avons travaillé avec des **événements** abstraits (A, B, C). Les **variables aléatoires** sont le pont entre probabilités et statistiques : elles **assignent des nombres** aux résultats d'expériences aléatoires, permettant de faire des **calculs mathématiques**. Vous apprendrez à définir variables aléatoires discrètes et continues, calculer leurs **espérances** (valeur moyenne), **variances** (dispersion), et comprendre leurs **distributions** via PMF, PDF et CDF. Ce cours est la **fondation** pour comprendre toutes les lois de probabilité classiques et le machine learning probabiliste.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Définir variable aléatoire, distinguer discrète/continue, PMF/PDF/CDF
2. **Appliquer** : Calculer espérance, variance, écart-type pour VA discrètes et continues
3. **Analyser** : Interpréter PMF/PDF/CDF graphiquement, identifier type de distribution
4. **Évaluer** : Choisir entre espérance et médiane selon contexte
5. **Créer** : Modéliser problèmes réels avec variables aléatoires
6. **Synthétiser** : Utiliser propriétés linéarité espérance/variance dans calculs complexes

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

**Sans variables aléatoires, pas de modélisation statistique !**

Toutes ces notions fondamentales reposent sur les VA :
- **Lois de probabilité** : Bernoulli, Binomiale, Normale, Poisson, etc.
- **Statistiques inférentielles** : Estimateurs, intervalles de confiance
- **Machine Learning** : Fonctions de perte, régularisation probabiliste
- **Processus stochastiques** : Séries temporelles, chaînes de Markov
- **Deep Learning** : Dropout, VAE, diffusion models

**Exemple concret** : Prix d'une action boursière

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation prix action sur 100 jours
# Prix_t = Prix_{t-1} × exp(r_t)
# où r_t ~ N(μ, σ²) (rendement quotidien)

prix_initial = 100
n_jours = 100
mu_rendement = 0.001  # Rendement moyen 0.1% par jour
sigma_rendement = 0.02  # Volatilité 2% par jour

# Rendements quotidiens (Variable Aléatoire Continue)
rendements = np.random.normal(mu_rendement, sigma_rendement, n_jours)

# Prix (transformation de VA)
prix = prix_initial * np.exp(np.cumsum(rendements))

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Évolution prix
axes[0, 0].plot(prix, linewidth=2, color='blue')
axes[0, 0].axhline(prix_initial, color='red', linestyle='--', linewidth=2, label='Prix initial')
axes[0, 0].set_xlabel('Jours')
axes[0, 0].set_ylabel('Prix (€)')
axes[0, 0].set_title('Évolution Prix Action (Processus Aléatoire)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Distribution rendements (VA)
axes[0, 1].hist(rendements, bins=30, density=True, alpha=0.7, 
                edgecolor='black', color='lightblue', label='Données')
x_range = np.linspace(rendements.min(), rendements.max(), 1000)
from scipy.stats import norm
axes[0, 1].plot(x_range, norm.pdf(x_range, mu_rendement, sigma_rendement), 
                'r-', linewidth=2, label=f'N({mu_rendement}, {sigma_rendement}²)')
axes[0, 1].axvline(mu_rendement, color='green', linestyle='--', linewidth=2, label='Espérance')
axes[0, 1].set_xlabel('Rendement quotidien')
axes[0, 1].set_ylabel('Densité')
axes[0, 1].set_title('Distribution des Rendements (VA Continue)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Distribution prix final (transformation de VA)
n_simulations = 10000
rendements_sim = np.random.normal(mu_rendement, sigma_rendement, (n_simulations, n_jours))
prix_final_sim = prix_initial * np.exp(np.sum(rendements_sim, axis=1))

axes[1, 0].hist(prix_final_sim, bins=50, density=True, alpha=0.7, 
                edgecolor='black', color='lightgreen')
axes[1, 0].axvline(np.mean(prix_final_sim), color='red', linestyle='--', 
                   linewidth=2, label=f'Espérance = {np.mean(prix_final_sim):.1f}€')
axes[1, 0].axvline(np.median(prix_final_sim), color='blue', linestyle='--', 
                   linewidth=2, label=f'Médiane = {np.median(prix_final_sim):.1f}€')
axes[1, 0].set_xlabel('Prix après 100 jours (€)')
axes[1, 0].set_ylabel('Densité')
axes[1, 0].set_title('Distribution Prix Final (10k simulations)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Statistiques
axes[1, 1].axis('off')
stats_text = f"""
STATISTIQUES DE LA VARIABLE ALÉATOIRE "Prix"

Rendement quotidien (VA) :
  • Type : Continue (Normale)
  • Espérance : {mu_rendement:.3f} ({mu_rendement*100:.1f}%)
  • Écart-type : {sigma_rendement:.3f} ({sigma_rendement*100:.1f}%)
  
Prix après 100 jours (VA) :
  • Type : Continue (Log-Normale)
  • Espérance : {np.mean(prix_final_sim):.2f}€
  • Écart-type : {np.std(prix_final_sim):.2f}€
  • Médiane : {np.median(prix_final_sim):.2f}€
  • P(Prix > 100€) = {np.mean(prix_final_sim > 100):.1%}
  • P(Prix > 120€) = {np.mean(prix_final_sim > 120):.1%}
  
💡 Grâce aux VA, on peut :
  - Modéliser incertitude (rendements aléatoires)
  - Calculer probabilités (P(Prix > seuil))
  - Estimer risque (écart-type, VaR)
  - Optimiser portefeuille (max rendement, min risque)
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("💡 Le prix est une VARIABLE ALÉATOIRE :")
print("   - On ne connaît pas sa valeur future avec certitude")
print("   - On connaît sa DISTRIBUTION (Normale pour rendements)")
print("   - On calcule ESPÉRANCE et VARIANCE pour mesurer rendement et risque")
```

**Sans comprendre variables aléatoires, impossible de faire de la finance quantitative, du ML probabiliste, ou des statistiques avancées !**

### Quel problème résout-il ?

**Problème** : Vous lancez 3 pièces de monnaie. Combien de Piles obtenez-vous en moyenne ? Quelle est la probabilité d'obtenir exactement 2 Piles ?

**Sans VA** : Analyser tous les cas $$\{PPP, PPF, PFP, FPP, PFF, FPF, FFP, FFF\}$$ → Fastidieux !

**Avec VA** : Définir $$X$$ = "Nombre de Piles" → $$X$$ peut valoir 0, 1, 2, ou 3 avec certaines probabilités.

```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Énumération complète (sans VA)
print("SANS VARIABLE ALÉATOIRE (approche naïve)")
print("="*60)

resultats = list(product(['P', 'F'], repeat=3))
print(f"Tous les résultats possibles ({len(resultats)}) :")
for r in resultats:
    n_piles = r.count('P')
    print(f"  {''.join(r)} → {n_piles} Pile(s)")

# Comptage manuel
comptage = {}
for r in resultats:
    n = r.count('P')
    comptage[n] = comptage.get(n, 0) + 1

print(f"\nComptage :")
for k, v in sorted(comptage.items()):
    print(f"  {k} Piles : {v} cas → P(X={k}) = {v}/{len(resultats)} = {v/len(resultats):.3f}")

print("\n" + "="*60)
print("AVEC VARIABLE ALÉATOIRE (approche élégante)")
print("="*60)

# Définition VA : X = nombre de Piles
# X peut prendre valeurs {0, 1, 2, 3}
# Distribution : X ~ Binomiale(n=3, p=0.5)

from scipy.stats import binom

n = 3  # Nombre de lancers
p = 0.5  # Probabilité de Pile

# PMF (Probability Mass Function)
x_vals = np.arange(0, 4)
pmf_vals = binom.pmf(x_vals, n, p)

print("Variable Aléatoire X ~ Binomiale(n=3, p=0.5)")
print("PMF (Fonction de Masse) :")
for x, prob in zip(x_vals, pmf_vals):
    print(f"  P(X={x}) = {prob:.3f}")

# Espérance et variance (formules directes !)
esperance = binom.mean(n, p)
variance = binom.var(n, p)
ecart_type = binom.std(n, p)

print(f"\nEspérance (valeur moyenne) : E[X] = {esperance:.2f} Piles")
print(f"Variance : Var(X) = {variance:.2f}")
print(f"Écart-type : σ(X) = {ecart_type:.2f}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PMF
axes[0].bar(x_vals, pmf_vals, alpha=0.7, edgecolor='black', color='skyblue')
axes[0].axvline(esperance, color='red', linestyle='--', linewidth=2, 
                label=f'Espérance = {esperance}')
axes[0].set_xlabel('Nombre de Piles (X)')
axes[0].set_ylabel('Probabilité P(X=k)')
axes[0].set_title('PMF de X (Distribution de Probabilité)')
axes[0].set_xticks(x_vals)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

for x, prob in zip(x_vals, pmf_vals):
    axes[0].text(x, prob + 0.02, f'{prob:.3f}', ha='center', fontweight='bold')

# CDF
cdf_vals = binom.cdf(x_vals, n, p)
axes[1].step(x_vals, cdf_vals, where='post', linewidth=2, color='blue', label='CDF')
axes[1].scatter(x_vals, cdf_vals, s=100, color='blue', zorder=3)
axes[1].set_xlabel('Nombre de Piles (x)')
axes[1].set_ylabel('Probabilité P(X ≤ x)')
axes[1].set_title('CDF de X (Fonction de Répartition)')
axes[1].set_xticks(x_vals)
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 AVANTAGES de l'approche VA :")
print("   1. Notation compacte : X au lieu de lister tous les cas")
print("   2. Formules directes : E[X] = np, Var(X) = np(1-p)")
print("   3. Visualisation : PMF, CDF")
print("   4. Généralisable : Marche pour n'importe quel n")
```

**Les variables aléatoires transforment un problème combinatoire fastidieux en calcul probabiliste élégant !**

### Applications dans le monde réel

1. **Machine Learning** :
   - Features = VA (hauteur, poids, revenu)
   - Prédiction = Espérance conditionnelle $$E[Y|X]$$
   - Incertitude = Variance prédictive

2. **Finance** :
   - Rendement d'actif = VA
   - Portefeuille optimal = Maximiser $$E[\text{Rendement}]$$, minimiser $$\text{Var}(\text{Rendement})$$

3. **Assurance** :
   - Nombre de sinistres = VA de Poisson
   - Prime = Espérance des coûts + marge

4. **Télécommunications** :
   - Nombre d'appels par heure = VA
   - Dimensionnement réseau selon distribution

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous commençons par définir rigoureusement les VA, puis introduisons les **distributions** (PMF/PDF/CDF), et enfin les **caractéristiques** (espérance, variance).

### 1. Définition Formelle de Variable Aléatoire

#### 1.1 Définition Mathématique

**Variable aléatoire** (VA) : Fonction $$X: \Omega \to \mathbb{R}$$ qui associe un **nombre réel** à chaque résultat d'une expérience aléatoire.

**Notation** : Lettres majuscules $$X, Y, Z$$ pour la VA, minuscules $$x, y, z$$ pour les valeurs qu'elle prend.

**Pourquoi cette définition ?**

Une VA **n'est pas** une variable classique (avec valeur fixe). C'est une **fonction** qui transforme des résultats aléatoires abstraits en nombres manipulables.

**Exemple 1 : Lancer de dé**

```python
# Espace des possibles Ω
Omega = {1, 2, 3, 4, 5, 6}

# Variable aléatoire X : "Valeur du dé"
# X(ω) = ω pour ω ∈ Ω
# X est l'identité ici

# Variable aléatoire Y : "Le résultat est-il pair ?"
def Y(omega):
    return 1 if omega % 2 == 0 else 0

print("VARIABLES ALÉATOIRES sur lancer de dé")
print("="*60)
print("\nX : Valeur du dé")
for omega in Omega:
    print(f"  ω={omega} → X(ω)={omega}")

print("\nY : Indicateur 'pair' (1 si pair, 0 sinon)")
for omega in Omega:
    print(f"  ω={omega} → Y(ω)={Y(omega)}")

print("\n💡 X et Y sont 2 VA différentes sur le MÊME espace Ω")
```

**Exemple 2 : Lancer 2 pièces**

```python
from itertools import product

# Espace des possibles
Omega = list(product(['P', 'F'], repeat=2))

# VA : X = "Nombre de Piles"
def X(omega):
    return omega.count('P')

print("\nVARIABLE ALÉATOIRE : Nombre de Piles")
print("="*60)
for omega in Omega:
    print(f"  ω={''.join(omega)} → X(ω)={X(omega)}")

print("\nValeurs possibles de X : {0, 1, 2}")
print("X transforme 4 résultats abstraits en 3 valeurs numériques")
```

#### 1.2 Types de Variables Aléatoires

**Variable aléatoire discrète** : Prend un ensemble **fini** ou **dénombrable** de valeurs.

**Exemples** :
- Nombre de Piles (0, 1, 2, ...)
- Nombre de clients par jour (0, 1, 2, ...)
- Résultat d'un dé (1, 2, 3, 4, 5, 6)

**Variable aléatoire continue** : Prend un ensemble **continu** de valeurs (intervalle).

**Exemples** :
- Taille d'une personne (tout réel dans [0, 3] mètres)
- Temps d'attente (tout réel ≥ 0)
- Température (tout réel)

**Distinction cruciale** :

| Critère | Discrète | Continue |
|---------|----------|----------|
| **Valeurs** | Dénombrables (liste) | Continues (intervalle) |
| **Distribution** | PMF (Probability Mass Function) | PDF (Probability Density Function) |
| **P(X=x)** | $$> 0$$ possible | $$= 0$$ toujours ! |
| **Somme/Intégrale** | $$\sum_x P(X=x) = 1$$ | $$\int_{-\infty}^{\infty} f(x)dx = 1$$ |
| **Exemples** | Binomiale, Poisson | Normale, Exponentielle |

**⚠️ Piège** : Pour VA continue, $$P(X = x) = 0$$ pour toute valeur $$x$$ !

On calcule $$P(a < X \leq b) = \int_a^b f(x)dx$$ (probabilité d'un intervalle).

---

### 2. Fonction de Masse (PMF) et Densité (PDF)

#### 2.1 PMF (Probability Mass Function) - VA Discrète

**PMF** : Fonction $$p_X(x) = P(X = x)$$ qui donne la probabilité que $$X$$ prenne la valeur $$x$$.

**Propriétés** :
1. $$p_X(x) \geq 0$$ pour tout $$x$$
2. $$\sum_{x} p_X(x) = 1$$ (somme sur toutes les valeurs possibles)

**Exemple : Lancer un dé équilibré**

```python
import numpy as np
import matplotlib.pyplot as plt

# VA : X = valeur du dé
x_vals = np.arange(1, 7)
pmf_vals = np.ones(6) / 6  # Équilibré → 1/6 pour chaque face

# Vérification propriétés
print("PMF : Dé équilibré")
print("="*50)
for x, p in zip(x_vals, pmf_vals):
    print(f"  P(X={x}) = {p:.4f}")

print(f"\nSomme des probabilités : {np.sum(pmf_vals):.4f} (doit = 1)")

# Visualisation
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x_vals, pmf_vals, alpha=0.7, edgecolor='black', color='skyblue', width=0.6)
ax.set_xlabel('Valeur x', fontsize=12)
ax.set_ylabel('P(X=x)', fontsize=12)
ax.set_title('PMF : Dé Équilibré', fontsize=14, fontweight='bold')
ax.set_xticks(x_vals)
ax.set_ylim(0, 0.25)
ax.grid(alpha=0.3, axis='y')

for x, p in zip(x_vals, pmf_vals):
    ax.text(x, p + 0.01, f'{p:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Exemple : Lancer 2 dés, somme**

```python
# VA : X = somme de 2 dés
# Valeurs possibles : 2, 3, ..., 12

# Comptage des cas
from collections import Counter
resultats = [(i, j) for i in range(1, 7) for j in range(1, 7)]
sommes = [i+j for i, j in resultats]
comptage = Counter(sommes)

x_vals = np.arange(2, 13)
pmf_vals = np.array([comptage[x] / 36 for x in x_vals])

# Visualisation
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x_vals, pmf_vals, alpha=0.7, edgecolor='black', color='lightgreen', width=0.6)
ax.set_xlabel('Somme (x)', fontsize=12)
ax.set_ylabel('P(X=x)', fontsize=12)
ax.set_title('PMF : Somme de 2 Dés', fontsize=14, fontweight='bold')
ax.set_xticks(x_vals)
ax.grid(alpha=0.3, axis='y')

for x, p in zip(x_vals, pmf_vals):
    ax.text(x, p + 0.005, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("PMF : Somme de 2 dés")
print("="*50)
for x, p in zip(x_vals, pmf_vals):
    print(f"  P(X={x:2d}) = {comptage[x]:2d}/36 = {p:.4f}")

print(f"\n💡 Distribution NON uniforme : 7 est la somme la plus probable")
```

#### 2.2 PDF (Probability Density Function) - VA Continue

**PDF** : Fonction $$f_X(x) \geq 0$$ telle que :

$$P(a < X \leq b) = \int_a^b f_X(x) dx$$

**⚠️ Attention** : $$f_X(x)$$ **n'est PAS** une probabilité ! C'est une **densité**.

**Propriétés** :
1. $$f_X(x) \geq 0$$ pour tout $$x$$
2. $$\int_{-\infty}^{\infty} f_X(x) dx = 1$$
3. $$P(X = x) = 0$$ pour tout $$x$$ (probabilité d'un point = 0)
4. $$f_X(x)$$ peut être > 1 ! (c'est une densité, pas une probabilité)

**Exemple : Loi Uniforme Continue**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# VA : X ~ Uniforme[0, 1]
# PDF : f(x) = 1 si x ∈ [0,1], 0 sinon

x = np.linspace(-0.5, 1.5, 1000)
pdf = uniform.pdf(x, loc=0, scale=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF
axes[0].plot(x, pdf, linewidth=3, color='blue', label='PDF : f(x)')
axes[0].fill_between(x, 0, pdf, where=(x >= 0.3) & (x <= 0.7), 
                      alpha=0.3, color='red', label='P(0.3 < X ≤ 0.7)')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('f(x)', fontsize=12)
axes[0].set_title('PDF : Loi Uniforme [0,1]', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1.5)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Calcul probabilité
prob = uniform.cdf(0.7, loc=0, scale=1) - uniform.cdf(0.3, loc=0, scale=1)
axes[0].text(0.5, 0.5, f'P(0.3 < X ≤ 0.7) = {prob:.2f}', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Comparaison : PDF peut être > 1
# Exemple : Uniforme[0, 0.5]
x2 = np.linspace(-0.2, 0.7, 1000)
pdf2 = uniform.pdf(x2, loc=0, scale=0.5)

axes[1].plot(x2, pdf2, linewidth=3, color='purple', label='PDF : Uniforme[0, 0.5]')
axes[1].axhline(1, color='red', linestyle='--', linewidth=2, label='y=1 (ref)')
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('f(x)', fontsize=12)
axes[1].set_title('PDF peut être > 1 !', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 3)
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[1].text(0.25, 2.2, 'f(x) = 2 sur [0, 0.5]\n(densité, pas probabilité !)', 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.show()

print("PDF : Loi Uniforme")
print("="*50)
print("X ~ Uniforme[0, 1]")
print("  f(x) = 1 pour x ∈ [0,1], 0 ailleurs")
print(f"  P(0.3 < X ≤ 0.7) = ∫[0.3 to 0.7] f(x)dx = {prob:.2f}")
print(f"\n⚠️ f(x) = 1 n'est PAS une probabilité !")
print(f"   Aire sous la courbe = 1 (probabilité totale)")
```

**Exemple : Loi Normale (Gaussienne)**

```python
from scipy.stats import norm

# X ~ N(μ=0, σ²=1) : Normale standard
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, mu, sigma)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x, pdf, linewidth=3, color='blue', label=f'PDF : N({mu}, {sigma}²)')

# Zone pour P(-1 < X ≤ 1)
x_zone = x[(x >= -1) & (x <= 1)]
pdf_zone = pdf[(x >= -1) & (x <= 1)]
ax.fill_between(x_zone, 0, pdf_zone, alpha=0.3, color='red', 
                label='P(-1 < X ≤ 1) ≈ 68%')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('PDF : Loi Normale Standard N(0, 1)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Annotations
prob = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
ax.text(0, 0.2, f'P(-1 < X ≤ 1) = {prob:.4f}', 
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
plt.show()

print("PDF : Loi Normale N(0, 1)")
print("="*50)
print(f"f(x) = (1/√(2π)) × exp(-x²/2)")
print(f"\nProbabilités courantes :")
print(f"  P(-1 < X ≤ 1) = {prob:.4f} (≈ 68%)")
print(f"  P(-2 < X ≤ 2) = {norm.cdf(2) - norm.cdf(-2):.4f} (≈ 95%)")
print(f"  P(-3 < X ≤ 3) = {norm.cdf(3) - norm.cdf(-3):.4f} (≈ 99.7%)")
```

**Sources académiques** :
- Ross, S. (2014). *A First Course in Probability*. Pearson - Définitions rigoureuses PMF/PDF

---

### 3. Fonction de Répartition (CDF)

#### 3.1 Définition

**CDF (Cumulative Distribution Function)** : Fonction $$F_X(x) = P(X \leq x)$$

**Valable pour VA discrètes ET continues.**

**Propriétés** :
1. $$0 \leq F_X(x) \leq 1$$
2. $$F_X$$ est **non décroissante** : si $$x_1 < x_2$$ alors $$F_X(x_1) \leq F_X(x_2)$$
3. $$\lim_{x \to -\infty} F_X(x) = 0$$
4. $$\lim_{x \to +\infty} F_X(x) = 1$$
5. $$F_X$$ est **continue à droite**

**Relations avec PMF/PDF** :

**Discrète** :

$$F_X(x) = \sum_{k \leq x} p_X(k)$$

**Continue** :

$$F_X(x) = \int_{-\infty}^{x} f_X(t) dt$$

**Inverse** (pour continues) :

$$f_X(x) = \frac{d F_X(x)}{dx}$$

#### 3.2 Visualisation CDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== DISCRÈTE : Binomiale ==========
n, p = 10, 0.5
x_disc = np.arange(0, 11)
pmf = binom.pmf(x_disc, n, p)
cdf = binom.cdf(x_disc, n, p)

# PMF
axes[0, 0].bar(x_disc, pmf, alpha=0.7, edgecolor='black', color='skyblue', width=0.6)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(X=x)')
axes[0, 0].set_title('PMF : Binomiale(n=10, p=0.5)', fontweight='bold')
axes[0, 0].set_xticks(x_disc)
axes[0, 0].grid(alpha=0.3, axis='y')

# CDF
axes[0, 1].step(x_disc, cdf, where='post', linewidth=2, color='blue', label='CDF')
axes[0, 1].scatter(x_disc, cdf, s=80, color='blue', zorder=3)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('P(X ≤ x)')
axes[0, 1].set_title('CDF : Binomiale(n=10, p=0.5)', fontweight='bold')
axes[0, 1].set_xticks(x_disc)
axes[0, 1].set_ylim(-0.1, 1.1)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Annotations
axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Médiane (F=0.5)')
axes[0, 1].axvline(5, color='red', linestyle='--', alpha=0.5)

# ========== CONTINUE : Normale ==========
mu, sigma = 0, 1
x_cont = np.linspace(-4, 4, 1000)
pdf_cont = norm.pdf(x_cont, mu, sigma)
cdf_cont = norm.cdf(x_cont, mu, sigma)

# PDF
axes[1, 0].plot(x_cont, pdf_cont, linewidth=2, color='blue', label='PDF')
axes[1, 0].fill_between(x_cont, 0, pdf_cont, alpha=0.3, color='lightblue')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].set_title('PDF : Normale(μ=0, σ=1)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# CDF
axes[1, 1].plot(x_cont, cdf_cont, linewidth=2, color='blue', label='CDF')
axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Médiane (F=0.5)')
axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('P(X ≤ x)')
axes[1, 1].set_title('CDF : Normale(μ=0, σ=1)', fontweight='bold')
axes[1, 1].set_ylim(-0.1, 1.1)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("FONCTION DE RÉPARTITION (CDF)")
print("="*60)
print("\nDiscrète (Binomiale) :")
print("  • Escalier (sauts aux valeurs possibles)")
print("  • F(x) = Σ P(X=k) pour k ≤ x")
print("  • Médiane : plus petit x tel que F(x) ≥ 0.5")

print("\nContinue (Normale) :")
print("  • Courbe lisse (en S)")
print("  • F(x) = ∫[-∞ to x] f(t)dt")
print("  • Médiane : x tel que F(x) = 0.5")
```

#### 3.3 Utilisation de la CDF

**Calcul de probabilités** :

$$P(a < X \leq b) = F_X(b) - F_X(a)$$

$$P(X > a) = 1 - F_X(a)$$

```python
from scipy.stats import norm

# X ~ N(100, 15²) : QI
mu, sigma = 100, 15

# Questions
print("EXEMPLE : QI ~ N(100, 15²)")
print("="*60)

# Q1 : P(X ≤ 115) ?
prob_1 = norm.cdf(115, mu, sigma)
print(f"P(QI ≤ 115) = {prob_1:.4f} = {prob_1:.1%}")

# Q2 : P(X > 130) ?
prob_2 = 1 - norm.cdf(130, mu, sigma)
print(f"P(QI > 130) = {prob_2:.4f} = {prob_2:.1%}")

# Q3 : P(85 < X ≤ 115) ?
prob_3 = norm.cdf(115, mu, sigma) - norm.cdf(85, mu, sigma)
print(f"P(85 < QI ≤ 115) = {prob_3:.4f} = {prob_3:.1%}")

# Q4 : Trouver seuil pour top 10%
# P(X > x) = 0.10 → P(X ≤ x) = 0.90
seuil = norm.ppf(0.90, mu, sigma)  # ppf = inverse CDF
print(f"\nTop 10% : QI > {seuil:.1f}")
```

---

### 4. Espérance (Expected Value)

#### 4.1 Définition

**Espérance** : "Valeur moyenne" que prendrait $$X$$ si on répétait l'expérience un grand nombre de fois.

**Notation** : $$E[X]$$ ou $$\mu_X$$

**Formules** :

**Discrète** :

$$E[X] = \sum_{x} x \cdot P(X = x) = \sum_{x} x \cdot p_X(x)$$

**Continue** :

$$E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**Intuition** : Moyenne **pondérée** par les probabilités.

**Pourquoi "espérance" ?**

C'est la valeur qu'on "espère" obtenir en moyenne. Loi des Grands Nombres garantit que la moyenne empirique converge vers $$E[X]$$.

#### 4.2 Calculs d'Espérance

**Exemple 1 : Dé équilibré**

```python
# X = valeur du dé
x_vals = np.arange(1, 7)
pmf = np.ones(6) / 6

esperance = np.sum(x_vals * pmf)

print("ESPÉRANCE : Dé équilibré")
print("="*50)
print("E[X] = Σ x × P(X=x)")
for x, p in zip(x_vals, pmf):
    print(f"     + {x} × {p:.4f} = {x*p:.4f}")
print(f"     ─────────────────")
print(f"E[X] = {esperance:.2f}")

# Vérification par simulation
n_lancers = 100000
simulations = np.random.randint(1, 7, n_lancers)
esperance_empirique = np.mean(simulations)

print(f"\nVérification simulation ({n_lancers:,} lancers) :")
print(f"  Moyenne empirique = {esperance_empirique:.4f}")
print(f"  Espérance théorique = {esperance:.4f}")
print(f"  Différence = {abs(esperance - esperance_empirique):.4f}")
```

**Exemple 2 : Gain jeu de hasard**

```python
# Jeu : Lancer dé
# • Si 6 → Gain 10€
# • Sinon → Perte 1€

# VA : X = gain
# X peut valoir {10, -1}

p_gagner = 1/6
p_perdre = 5/6

esperance_gain = 10 * p_gagner + (-1) * p_perdre

print("\nESPÉRANCE : Jeu de dé")
print("="*50)
print("Règles :")
print("  • Si dé = 6 → Gain 10€")
print("  • Sinon → Perte 1€")
print()
print("E[Gain] = 10 × P(6) + (-1) × P(non-6)")
print(f"        = 10 × {p_gagner:.4f} + (-1) × {p_perdre:.4f}")
print(f"        = {esperance_gain:.4f}€")

if esperance_gain > 0:
    print(f"\n✅ Espérance positive : jeu FAVORABLE (gain moyen {esperance_gain:.2f}€)")
else:
    print(f"\n❌ Espérance négative : jeu DÉFAVORABLE (perte moyenne {-esperance_gain:.2f}€)")

# Simulation
n_jeux = 100000
des = np.random.randint(1, 7, n_jeux)
gains = np.where(des == 6, 10, -1)
gain_moyen = np.mean(gains)

print(f"\nSimulation ({n_jeux:,} parties) :")
print(f"  Gain moyen = {gain_moyen:.4f}€")
print(f"  Espérance théorique = {esperance_gain:.4f}€")
```

**Exemple 3 : Loi Uniforme Continue**

```python
# X ~ Uniforme[a, b]
# E[X] = (a + b) / 2

a, b = 0, 10

esperance_theo = (a + b) / 2

print("\nESPÉRANCE : Loi Uniforme Continue [0, 10]")
print("="*50)
print(f"E[X] = ∫ x × f(x)dx")
print(f"     = ∫[{a} to {b}] x × (1/{b-a})dx")
print(f"     = (a + b) / 2")
print(f"     = {esperance_theo:.2f}")

# Simulation
n_sims = 100000
simulations = np.random.uniform(a, b, n_sims)
esperance_emp = np.mean(simulations)

print(f"\nSimulation ({n_sims:,} tirages) :")
print(f"  Moyenne empirique = {esperance_emp:.4f}")
print(f"  Espérance théorique = {esperance_theo:.4f}")
```

#### 4.3 Propriétés de l'Espérance

**Propriété 1 : Linéarité**

$$E[aX + b] = a \cdot E[X] + b$$

$$E[X + Y] = E[X] + E[Y]$$ **(toujours, même si dépendantes !)**

```python
# Vérification linéarité
X = np.random.randint(1, 7, 10000)
a, b = 2, 5

# E[aX + b] = a*E[X] + b ?
esperance_X = np.mean(X)
esperance_aX_plus_b = np.mean(a*X + b)
esperance_calculee = a * esperance_X + b

print("PROPRIÉTÉ : Linéarité de l'espérance")
print("="*50)
print(f"E[X] = {esperance_X:.4f}")
print(f"E[2X + 5] (empirique) = {esperance_aX_plus_b:.4f}")
print(f"2×E[X] + 5 (théorique) = {esperance_calculee:.4f}")
print(f"Égalité vérifiée : {np.isclose(esperance_aX_plus_b, esperance_calculee)}")
```

**Propriété 2 : Espérance d'une fonction de X**

$$E[g(X)] = \sum_x g(x) \cdot P(X=x)$$ (discrète)

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) dx$$ (continue)

```python
# Exemple : E[X²] pour dé
X_vals = np.arange(1, 7)
pmf = np.ones(6) / 6

esperance_X = np.sum(X_vals * pmf)
esperance_X_carre = np.sum(X_vals**2 * pmf)

print("\nESPÉRANCE d'une fonction : E[X²]")
print("="*50)
print(f"E[X] = {esperance_X:.4f}")
print(f"E[X²] = Σ x² × P(X=x)")

for x, p in zip(X_vals, pmf):
    print(f"      + {x}² × {p:.4f} = {x**2 * p:.4f}")

print(f"      ─────────────────")
print(f"E[X²] = {esperance_X_carre:.4f}")

print(f"\n⚠️ E[X²] ≠ (E[X])² en général !")
print(f"  E[X²] = {esperance_X_carre:.4f}")
print(f"  (E[X])² = {esperance_X**2:.4f}")
```

---

### 5. Variance et Écart-type

#### 5.1 Définition

**Variance** : Mesure de la **dispersion** autour de l'espérance.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Écart-type** :

$$\sigma_X = \sqrt{\text{Var}(X)}$$

**Formules de calcul** :

**Discrète** :

$$\text{Var}(X) = \sum_x (x - \mu)^2 \cdot P(X=x)$$

où $$\mu = E[X]$$

**Continue** :

$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f_X(x) dx$$

**Formule computationnelle** (plus stable numériquement) :

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

#### 5.2 Calculs de Variance

```python
# Dé équilibré
X_vals = np.arange(1, 7)
pmf = np.ones(6) / 6

esperance = np.sum(X_vals * pmf)
esperance_carre = np.sum(X_vals**2 * pmf)

# Méthode 1 : E[(X - μ)²]
variance_methode1 = np.sum((X_vals - esperance)**2 * pmf)

# Méthode 2 : E[X²] - (E[X])²
variance_methode2 = esperance_carre - esperance**2

ecart_type = np.sqrt(variance_methode1)

print("VARIANCE : Dé équilibré")
print("="*50)
print(f"E[X] = {esperance:.4f}")
print(f"E[X²] = {esperance_carre:.4f}")
print()
print("Méthode 1 : Var(X) = E[(X - μ)²]")
for x, p in zip(X_vals, pmf):
    contrib = (x - esperance)**2 * p
    print(f"  + ({x} - {esperance:.2f})² × {p:.4f} = {contrib:.4f}")
print(f"  ──────────────────────")
print(f"  Var(X) = {variance_methode1:.4f}")

print(f"\nMéthode 2 : Var(X) = E[X²] - (E[X])²")
print(f"  = {esperance_carre:.4f} - {esperance:.4f}²")
print(f"  = {variance_methode2:.4f}")

print(f"\nÉcart-type : σ(X) = √Var(X) = {ecart_type:.4f}")

# Simulation
n_lancers = 100000
simulations = np.random.randint(1, 7, n_lancers)
variance_emp = np.var(simulations)
ecart_type_emp = np.std(simulations)

print(f"\nSimulation ({n_lancers:,} lancers) :")
print(f"  Variance empirique = {variance_emp:.4f}")
print(f"  Écart-type empirique = {ecart_type_emp:.4f}")
```

#### 5.3 Propriétés de la Variance

**Propriété 1 : Variance d'une constante**

$$\text{Var}(c) = 0$$

**Propriété 2 : Multiplication par constante**

$$\text{Var}(aX) = a^2 \cdot \text{Var}(X)$$

$$\text{Var}(X + b) = \text{Var}(X)$$ (ajouter constante ne change pas dispersion)

**Propriété 3 : Somme de VA indépendantes**

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$ **(SI $$X$$ et $$Y$$ indépendantes)**

**⚠️ Faux en général si dépendantes !**

```python
# Vérification propriétés
print("PROPRIÉTÉS DE LA VARIANCE")
print("="*60)

# Données
X = np.random.randint(1, 7, 10000)
var_X = np.var(X)

# Propriété : Var(aX) = a² Var(X)
a = 3
var_aX = np.var(a * X)
var_theo = a**2 * var_X

print(f"Var(X) = {var_X:.4f}")
print(f"Var(3X) = {var_aX:.4f}")
print(f"3² × Var(X) = {var_theo:.4f}")
print(f"Égalité vérifiée : {np.isclose(var_aX, var_theo)}")

# Propriété : Var(X + b) = Var(X)
b = 10
var_X_plus_b = np.var(X + b)

print(f"\nVar(X + 10) = {var_X_plus_b:.4f}")
print(f"Var(X) = {var_X:.4f}")
print(f"Égalité vérifiée : {np.isclose(var_X_plus_b, var_X)}")

# Propriété : Var(X + Y) si indépendants
Y = np.random.randint(1, 7, 10000)  # Indépendant de X
var_Y = np.var(Y)
var_X_plus_Y = np.var(X + Y)
var_theo_somme = var_X + var_Y

print(f"\nVar(X) = {var_X:.4f}")
print(f"Var(Y) = {var_Y:.4f}")
print(f"Var(X + Y) = {var_X_plus_Y:.4f}")
print(f"Var(X) + Var(Y) = {var_theo_somme:.4f}")
print(f"Égalité vérifiée : {np.isclose(var_X_plus_Y, var_theo_somme, rtol=0.01)}")
```

**Sources académiques** :
- Wackerly, D., Mendenhall, W., & Scheaffer, R. (2008). *Mathematical Statistics with Applications*. Thomson

---

## 💡 Compréhension Intuitive

### L'Analogie du Tireur à l'Arc

**Variable Aléatoire** = Position où la flèche atterrit (aléatoire)

**PMF/PDF** = Carte de chaleur montrant où le tireur vise (zones probables)

**Espérance** = Centre de la cible (point moyen visé)

**Variance** = Dispersion des tirs (précision du tireur)

**Tireur A** : $$E[X] = \text{centre}$$, $$\text{Var}(X) = \text{petite}$$ → Précis et juste

**Tireur B** : $$E[X] = \text{centre}$$, $$\text{Var}(X) = \text{grande}$$ → Juste mais imprécis

**Tireur C** : $$E[X] \neq \text{centre}$$, $$\text{Var}(X) = \text{petite}$$ → Précis mais biaisé

### Questions Rapides

1. **Q1** : $$E[2X + 3] = ?$$ (si $$E[X] = 5$$)
   - *Réponse* : $$2 \times 5 + 3 = 13$$

2. **Q2** : $$\text{Var}(2X) = ?$$ (si $$\text{Var}(X) = 4$$)
   - *Réponse* : $$2^2 \times 4 = 16$$

3. **Q3** : Pour VA continue, $$P(X = 3) = ?$$
   - *Réponse* : $$0$$ (toujours !)

4. **Q4** : CDF peut-elle décroître ?
   - *Réponse* : NON (toujours non décroissante)

---

## 💻 Implémentation Pratique

### Classe Variable Aléatoire en Python

```python
"""
Titre : Classe générique pour manipuler Variables Aléatoires discrètes
Objectif : Outil pour calculer PMF, CDF, espérance, variance
"""

import numpy as np
import matplotlib.pyplot as plt

class VariableAleatoireDiscrete:
    """
    Représente une variable aléatoire discrète.
    
    Attributes:
        valeurs (array): Valeurs possibles
        probabilites (array): Probabilités associées (PMF)
    """
    
    def __init__(self, valeurs, probabilites):
        """
        Initialise VA discrète.
        
        Args:
            valeurs (array-like): Valeurs que peut prendre X
            probabilites (array-like): P(X=x) pour chaque valeur
        """
        self.valeurs = np.array(valeurs)
        self.probabilites = np.array(probabilites)
        
        # Validation
        if not np.isclose(self.probabilites.sum(), 1.0):
            raise ValueError(f"Somme probabilités = {self.probabilites.sum():.4f} ≠ 1")
        
        if np.any(self.probabilites < 0):
            raise ValueError("Probabilités doivent être ≥ 0")
    
    def pmf(self, x):
        """Probability Mass Function : P(X=x)"""
        idx = np.where(self.valeurs == x)[0]
        return self.probabilites[idx[0]] if len(idx) > 0 else 0.0
    
    def cdf(self, x):
        """Cumulative Distribution Function : P(X ≤ x)"""
        return self.probabilites[self.valeurs <= x].sum()
    
    def esperance(self):
        """E[X] = Σ x × P(X=x)"""
        return np.sum(self.valeurs * self.probabilites)
    
    def variance(self):
        """Var(X) = E[X²] - (E[X])²"""
        esperance_X = self.esperance()
        esperance_X_carre = np.sum(self.valeurs**2 * self.probabilites)
        return esperance_X_carre - esperance_X**2
    
    def ecart_type(self):
        """σ(X) = √Var(X)"""
        return np.sqrt(self.variance())
    
    def quantile(self, q):
        """Calcule quantile d'ordre q (0 < q < 1)"""
        cdf_vals = np.array([self.cdf(x) for x in self.valeurs])
        idx = np.where(cdf_vals >= q)[0]
        return self.valeurs[idx[0]] if len(idx) > 0 else self.valeurs[-1]
    
    def simuler(self, n):
        """Génère n échantillons selon PMF"""
        return np.random.choice(self.valeurs, size=n, p=self.probabilites)
    
    def visualiser(self):
        """Affiche PMF et CDF"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # PMF
        axes[0].bar(self.valeurs, self.probabilites, alpha=0.7, 
                    edgecolor='black', color='skyblue', width=0.6)
        esperance = self.esperance()
        axes[0].axvline(esperance, color='red', linestyle='--', linewidth=2, 
                        label=f'E[X]={esperance:.2f}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('P(X=x)')
        axes[0].set_title('PMF (Fonction de Masse)')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        for x, p in zip(self.valeurs, self.probabilites):
            axes[0].text(x, p + 0.01, f'{p:.3f}', ha='center', fontsize=9)
        
        # CDF
        cdf_vals = np.array([self.cdf(x) for x in self.valeurs])
        axes[1].step(self.valeurs, cdf_vals, where='post', linewidth=2, 
                     color='blue', label='CDF')
        axes[1].scatter(self.valeurs, cdf_vals, s=80, color='blue', zorder=3)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('P(X ≤ x)')
        axes[1].set_title('CDF (Fonction de Répartition)')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self):
        return f"VA Discrète: E[X]={self.esperance():.2f}, Var(X)={self.variance():.2f}"

# ========== EXEMPLES D'UTILISATION ==========

print("EXEMPLE 1 : Dé équilibré")
print("="*60)

de = VariableAleatoireDiscrete(
    valeurs=[1, 2, 3, 4, 5, 6],
    probabilites=[1/6]*6
)

print(de)
print(f"E[X] = {de.esperance():.4f}")
print(f"Var(X) = {de.variance():.4f}")
print(f"σ(X) = {de.ecart_type():.4f}")
print(f"P(X=3) = {de.pmf(3):.4f}")
print(f"P(X≤4) = {de.cdf(4):.4f}")
print(f"Médiane (Q2) = {de.quantile(0.5)}")

de.visualiser()

# Simulation
echantillon = de.simuler(10000)
print(f"\nSimulation (10k lancers) :")
print(f"  Moyenne empirique = {np.mean(echantillon):.4f}")
print(f"  Variance empirique = {np.var(echantillon):.4f}")

print("\n" + "="*60)
print("EXEMPLE 2 : Jeu de hasard biaisé")
print("="*60)

# Gain : 50€ avec prob 0.1, perte 5€ avec prob 0.9
jeu = VariableAleatoireDiscrete(
    valeurs=[50, -5],
    probabilites=[0.1, 0.9]
)

print(jeu)
print(f"E[Gain] = {jeu.esperance():.2f}€")
print(f"σ(Gain) = {jeu.ecart_type():.2f}€")

if jeu.esperance() > 0:
    print(f"✅ Espérance positive : jeu FAVORABLE")
else:
    print(f"❌ Espérance négative : jeu DÉFAVORABLE")

jeu.visualiser()
```

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Confondre $$E[X^2]$$ et $$(E[X])^2$$

```python
# Démonstration
X_vals = np.array([1, 2, 3, 4, 5])
probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

E_X = np.sum(X_vals * probs)
E_X_squared = np.sum(X_vals**2 * probs)
squared_E_X = E_X**2

print("ERREUR FRÉQUENTE : E[X²] vs (E[X])²")
print("="*50)
print(f"E[X] = {E_X:.4f}")
print(f"E[X²] = {E_X_squared:.4f}")
print(f"(E[X])² = {squared_E_X:.4f}")
print(f"\n⚠️ E[X²] ≠ (E[X])² en général !")
print(f"   E[X²] - (E[X])² = Var(X) = {E_X_squared - squared_E_X:.4f}")
```

---

### ✅ Bonne Pratique 1 : Toujours Vérifier Somme PMF = 1

```python
def valider_pmf(valeurs, probabilites):
    """Valide qu'une PMF est bien définie"""
    somme = np.sum(probabilites)
    
    print("VALIDATION PMF")
    print("="*50)
    print(f"Somme des probabilités : {somme:.6f}")
    
    if not np.isclose(somme, 1.0):
        print(f"❌ ERREUR : Somme ≠ 1 (écart = {abs(somme - 1):.6f})")
        return False
    
    if np.any(probabilites < 0):
        print(f"❌ ERREUR : Probabilités négatives détectées")
        return False
    
    print("✅ PMF valide")
    return True

# Test
valeurs = [1, 2, 3, 4]
probs_fausses = [0.2, 0.3, 0.3, 0.1]  # Somme = 0.9 ≠ 1
probs_correctes = [0.25, 0.25, 0.25, 0.25]

valider_pmf(valeurs, probs_fausses)
print()
valider_pmf(valeurs, probs_correctes)
```

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Fondamentaux

1. **"A First Course in Probability"**
   - **Auteur** : Sheldon Ross
   - **Contribution** : Référence pédagogique sur VA
   - **Niveau** : Accessible

2. **"Mathematical Statistics with Applications"**
   - **Auteurs** : Wackerly, Mendenhall, Scheaffer
   - **Contribution** : Traitement rigoureux espérance/variance
   - **Niveau** : Intermédiaire

---

### 📚 Ressources

**Cours en ligne** :
- [MIT 6.041 - Random Variables](https://ocw.mit.edu/)
- [Khan Academy - Random Variables](https://www.khanacademy.org/)

**Visualisations interactives** :
- [Seeing Theory - Random Variables](https://seeing-theory.brown.edu/probability-distributions/)

---

### 📖 Cours Connexes

**Suite naturelle** :
- [[common_distributions]] - Bernoulli, Binomiale, Poisson, Normale, Exponentielle
- [[joint_distributions]] - Distributions jointes, covariance, corrélation
- [[moment_generating_functions]] - Fonctions génératrices des moments

**Applications** :
- [[law_large_numbers]] - Convergence moyenne empirique → espérance
- [[central_limit_theorem]] - Distribution de la somme → Normale
- [[maximum_likelihood]] - Estimation paramètres distributions

---

## 📝 Résumé Rapide

### Formules Clés

| Concept | Discrète | Continue |
|---------|----------|----------|
| **PMF/PDF** | $$p_X(x) = P(X=x)$$ | $$f_X(x)$$ (densité) |
| **CDF** | $$F_X(x) = \sum_{k \leq x} p_X(k)$$ | $$F_X(x) = \int_{-\infty}^x f_X(t)dt$$ |
| **Espérance** | $$E[X] = \sum_x x \cdot p_X(x)$$ | $$E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x)dx$$ |
| **Variance** | $$\text{Var}(X) = E[X^2] - (E[X])^2$$ | $$\text{Var}(X) = E[X^2] - (E[X])^2$$ |

### Propriétés Essentielles

- **Linéarité espérance** : $$E[aX + b] = aE[X] + b$$
- **Variance transformation** : $$\text{Var}(aX + b) = a^2\text{Var}(X)$$
- **Somme indépendantes** : $$\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$$ (si indép.)

### Code Minimal

```python
import numpy as np
from scipy import stats

# VA discrète
valeurs = [1, 2, 3, 4]
probs = [0.1, 0.2, 0.4, 0.3]

esperance = np.sum(np.array(valeurs) * np.array(probs))
variance = np.sum(np.array(valeurs)**2 * np.array(probs)) - esperance**2

# VA continue (Normale)
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)
```
