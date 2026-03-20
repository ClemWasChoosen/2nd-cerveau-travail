# 🎲 Fondements de la Théorie des Probabilités

> **Résumé en une phrase** : La théorie des probabilités fournit le langage mathématique rigoureux pour quantifier l'incertitude, base de toute inférence statistique et de l'apprentissage automatique.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-19 |
| **Dernière mise à jour** | 2026-03-19 |
| **Domaine** | Théorie des Probabilités |
| **Niveau** | Intermédiaire |
| **Durée de lecture** | ~50 minutes |
| **Fichier** | `probability_foundations.md` |
| **Emplacement** | `/00_statistics_foundations/02_probability_theory/` |
| **Tags** | `#probability` `#bayes` `#conditional-probability` `#independence` `#axioms` `#foundations` |

### Prérequis

- [x] [[measures_central_tendency]] - Moyenne, médiane (utile pour exemples)
- [ ] Mathématiques de base (ensembles, fractions, pourcentages)
- [ ] Logique de base (ET, OU, NON)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[measures_central_tendency]] - Statistiques descriptives
- **Complémentaires** : 
  - [[data_visualization_principles]] - Visualiser probabilités
- **Suite recommandée** : 
  - [[random_variables]] - Variables aléatoires et espérance
  - [[common_distributions]] - Distributions de probabilité classiques
  - [[bayesian_foundations]] - Statistiques bayésiennes
  - [[conditional_probability_advanced]] - Probabilités conditionnelles avancées

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Les probabilités sont **le langage de l'incertitude**. Dans le monde réel, nous ne pouvons presque jamais prédire avec certitude : *"Ce client va-t-il acheter ?" "Mon modèle ML va-t-il se tromper ?" "Cette feature est-elle vraiment importante ?"* Ce cours vous enseigne les **fondations mathématiques rigoureuses** de la théorie des probabilités : axiomes, probabilités conditionnelles, indépendance, et le célèbre **théorème de Bayes** qui révolutionne machine learning et IA.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Définir événements, espaces probabilisés, axiomes de Kolmogorov
2. **Appliquer** : Calculer probabilités simples, conditionnelles, utiliser règles addition/multiplication
3. **Analyser** : Distinguer événements indépendants vs mutuellement exclusifs
4. **Évaluer** : Identifier quand utiliser théorème de Bayes vs probabilités directes
5. **Créer** : Modéliser problèmes réels en termes probabilistes (arbres, tableaux)
6. **Synthétiser** : Résoudre problèmes complexes combinant probabilités conditionnelles et Bayes

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

**Sans probabilités, pas de Data Science moderne !**

Toutes ces applications reposent sur la théorie des probabilités :
- **Machine Learning** : Modèles probabilistes (naive Bayes, régression logistique, réseaux bayésiens)
- **Statistiques Inférentielles** : Tests d'hypothèses, intervalles de confiance, p-values
- **Deep Learning** : Fonctions de perte (cross-entropy), dropout, optimisation stochastique
- **IA Générative** : Modèles de langage (GPT), diffusion (Stable Diffusion, DALL-E)
- **Prise de décision** : Analyse de risque, A/B testing, théorie de la décision

**Exemple concret** : Spam filter

```python
import numpy as np
import matplotlib.pyplot as plt

# Données simulées d'un filtre spam simple
# P(mot | spam) et P(mot | ham)
mots = ['viagra', 'gratuit', 'réunion', 'rapport', 'offre']

# Probabilités qu'un mot apparaisse sachant que c'est un spam
prob_mot_sachant_spam = np.array([0.8, 0.7, 0.1, 0.05, 0.6])

# Probabilités qu'un mot apparaisse sachant que c'est ham (non-spam)
prob_mot_sachant_ham = np.array([0.01, 0.05, 0.7, 0.8, 0.1])

# Visualisation
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(mots))
width = 0.35

bars1 = ax.bar(x - width/2, prob_mot_sachant_spam, width, label='P(mot|Spam)', 
               color='red', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, prob_mot_sachant_ham, width, label='P(mot|Ham)', 
               color='green', alpha=0.7, edgecolor='black')

ax.set_xlabel('Mots', fontsize=12)
ax.set_ylabel('Probabilité', fontsize=12)
ax.set_title('Probabilités Conditionnelles : P(mot|classe)\nBase du Filtre Spam Bayésien', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mots)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("💡 Théorème de Bayes permet d'inverser :")
print("   P(mot|Spam) → P(Spam|mot)")
print("   'viagra' dans email → 80% de chance que ce soit spam")
```

**Sans comprendre probabilités conditionnelles et Bayes, impossible de comprendre comment fonctionne ce filtre !**

### Quel problème résout-il ?

**Problème** : Vous développez un système de diagnostic médical. Un test de dépistage du cancer donne un résultat **positif**. Le test est fiable à 95% (taux de vrais positifs). **Quelle est la probabilité que le patient ait vraiment le cancer ?**

**Réponse intuitive (FAUSSE)** : 95% !

**Réponse correcte** : Ça dépend de la **prévalence de la maladie** dans la population !

```python
import numpy as np

def diagnostic_bayesien(prevalence, sensibilite, specificite):
    """
    Calcule P(Maladie|Test+) avec théorème de Bayes.
    
    Args:
        prevalence: P(Maladie) - proportion malades dans population
        sensibilite: P(Test+|Maladie) - taux vrais positifs
        specificite: P(Test-|Pas Maladie) - taux vrais négatifs
    
    Returns:
        P(Maladie|Test+) - probabilité d'être malade sachant test positif
    """
    # P(Maladie)
    p_maladie = prevalence
    p_pas_maladie = 1 - prevalence
    
    # P(Test+|Maladie) = sensibilité
    p_test_pos_si_malade = sensibilite
    
    # P(Test+|Pas Maladie) = 1 - spécificité = taux faux positifs
    p_test_pos_si_pas_malade = 1 - specificite
    
    # P(Test+) = P(Test+|Maladie)×P(Maladie) + P(Test+|Pas Maladie)×P(Pas Maladie)
    p_test_pos = (p_test_pos_si_malade * p_maladie + 
                  p_test_pos_si_pas_malade * p_pas_maladie)
    
    # Théorème de Bayes :
    # P(Maladie|Test+) = P(Test+|Maladie) × P(Maladie) / P(Test+)
    p_malade_sachant_test_pos = (p_test_pos_si_malade * p_maladie) / p_test_pos
    
    return p_malade_sachant_test_pos

# Cas 1 : Maladie rare (1% de la population)
prob_1 = diagnostic_bayesien(prevalence=0.01, sensibilite=0.95, specificite=0.95)
print("Cas 1 : Maladie RARE (prévalence 1%)")
print(f"  Test positif → Probabilité réelle d'être malade : {prob_1:.1%}")
print(f"  ⚠️ Seulement {prob_1:.1%} alors que test à 95% de fiabilité !")

# Cas 2 : Maladie fréquente (20% de la population)
prob_2 = diagnostic_bayesien(prevalence=0.20, sensibilite=0.95, specificite=0.95)
print(f"\nCas 2 : Maladie FRÉQUENTE (prévalence 20%)")
print(f"  Test positif → Probabilité réelle d'être malade : {prob_2:.1%}")

print("\n💡 LEÇON : La probabilité a posteriori dépend CRUCIALE du taux de base (prévalence)")
print("   C'est le théorème de Bayes en action !")
```

**Résultat** :
- Maladie rare (1%) : Test positif → seulement **16% de chance** d'être vraiment malade
- Maladie fréquente (20%) : Test positif → **83% de chance** d'être malade

**Cette différence contre-intuitive est au cœur de nombreuses erreurs médicales, juridiques et statistiques !**

### Applications dans le monde réel

1. **Machine Learning** :
   - Naive Bayes classifier (spam, sentiment analysis)
   - Bayesian optimization (hyperparameter tuning)
   - Probabilistic graphical models

2. **Médecine & Diagnostic** :
   - Interprétation tests médicaux
   - Systèmes d'aide au diagnostic
   - Analyse risque génétique

3. **Finance** :
   - Pricing d'options (Black-Scholes)
   - Gestion de risque (VaR)
   - Prédiction marchés

4. **Sécurité & Détection** :
   - Filtres anti-spam
   - Détection de fraude
   - Systèmes d'intrusion

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous construisons la théorie depuis les axiomes fondamentaux (Kolmogorov), puis les règles dérivées, jusqu'aux applications (Bayes).

### 1. Expériences Aléatoires et Événements

#### 1.1 Définitions Fondamentales

**Expérience aléatoire** : Processus dont le résultat ne peut être prédit avec certitude avant son exécution.

**Exemples** :
- Lancer un dé
- Tirer une carte d'un jeu
- Mesurer le temps de réponse d'une API
- Observer si un email est un spam

**Espace des possibles** ($$\Omega$$) : Ensemble de **tous** les résultats possibles d'une expérience aléatoire.

**Exemples** :
- Dé à 6 faces : $$\Omega = \{1, 2, 3, 4, 5, 6\}$$
- Lancer 2 pièces : $$\Omega = \{(P,P), (P,F), (F,P), (F,F)\}$$ où P=Pile, F=Face
- Temps de réponse API : $$\Omega = [0, +\infty)$$ (continue)

**Événement** : Sous-ensemble de l'espace des possibles ($$A \subseteq \Omega$$).

**Exemples** :
- $$A$$ = "Obtenir un nombre pair" = $$\{2, 4, 6\}$$
- $$B$$ = "Obtenir au moins un Pile" = $$\{(P,P), (P,F), (F,P)\}$$
- $$C$$ = "Temps de réponse < 100ms" = $$[0, 100)$$

**Visualisation** :

```
Espace des possibles Ω (Dé à 6 faces)
┌─────────────────────────────────────┐
│  ●1   ●2   ●3   ●4   ●5   ●6        │
│                                     │
│  Événement A = "Pair" = {2,4,6}    │
│        ╔═══╗     ╔═══╗     ╔═══╗   │
│        ║ ●2║     ║ ●4║     ║ ●6║   │
│        ╚═══╝     ╚═══╝     ╚═══╝   │
└─────────────────────────────────────┘
```

#### 1.2 Opérations sur les Événements

**Union** ($$A \cup B$$) : Au moins l'un des événements se produit (OU logique).

**Intersection** ($$A \cap B$$) : Les deux événements se produisent simultanément (ET logique).

**Complémentaire** ($$A^c$$ ou $$\bar{A}$$) : L'événement ne se produit pas (NON logique).

**Différence** ($$A \setminus B$$) : $$A$$ se produit mais pas $$B$$.

**Événements mutuellement exclusifs** : $$A \cap B = \emptyset$$ (ne peuvent se produire en même temps).

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Fonction pour dessiner diagramme de Venn
def draw_venn(ax, title, highlight_A=False, highlight_B=False, 
              highlight_intersection=False, highlight_union=False):
    # Cercle A
    circle_A = patches.Circle((0.3, 0.5), 0.25, 
                               facecolor='blue' if highlight_A else 'lightblue',
                               edgecolor='black', linewidth=2, alpha=0.5, label='A')
    # Cercle B
    circle_B = patches.Circle((0.7, 0.5), 0.25, 
                               facecolor='red' if highlight_B else 'lightcoral',
                               edgecolor='black', linewidth=2, alpha=0.5, label='B')
    
    # Rectangle Ω
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                              edgecolor='black', facecolor='lightyellow', alpha=0.3)
    
    ax.add_patch(rect)
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Labels
    ax.text(0.3, 0.5, 'A', fontsize=16, ha='center', va='center', fontweight='bold')
    ax.text(0.7, 0.5, 'B', fontsize=16, ha='center', va='center', fontweight='bold')
    ax.text(0.05, 0.95, 'Ω', fontsize=16, fontweight='bold')

# 1. Union A ∪ B
draw_venn(axes[0, 0], 'Union : A ∪ B\n(Au moins un des deux)')

# 2. Intersection A ∩ B
draw_venn(axes[0, 1], 'Intersection : A ∩ B\n(Les deux simultanément)')

# 3. Complémentaire A^c
ax = axes[1, 0]
rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                          edgecolor='black', facecolor='lightgreen', alpha=0.5)
circle_A = patches.Circle((0.5, 0.5), 0.25, 
                           facecolor='white', edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.add_patch(circle_A)
ax.text(0.5, 0.5, 'A', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(0.05, 0.95, 'Ω', fontsize=16, fontweight='bold')
ax.text(0.15, 0.15, 'Aᶜ', fontsize=16, fontweight='bold', color='green')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Complémentaire : Aᶜ\n(Tout sauf A)', fontsize=14, fontweight='bold')

# 4. Événements mutuellement exclusifs
ax = axes[1, 1]
rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                          edgecolor='black', facecolor='lightyellow', alpha=0.3)
circle_A = patches.Circle((0.3, 0.5), 0.15, 
                           facecolor='lightblue', edgecolor='black', linewidth=2, alpha=0.7)
circle_B = patches.Circle((0.7, 0.5), 0.15, 
                           facecolor='lightcoral', edgecolor='black', linewidth=2, alpha=0.7)
ax.add_patch(rect)
ax.add_patch(circle_A)
ax.add_patch(circle_B)
ax.text(0.3, 0.5, 'A', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(0.7, 0.5, 'B', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(0.05, 0.95, 'Ω', fontsize=16, fontweight='bold')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Mutuellement Exclusifs\nA ∩ B = ∅', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

**Propriétés importantes** :

**Lois de De Morgan** :

$$(A \cup B)^c = A^c \cap B^c$$

$$(A \cap B)^c = A^c \cup B^c$$

**Intuition** : "Le contraire de (A OU B) = (contraire de A) ET (contraire de B)"

```python
# Vérification numérique lois De Morgan
import numpy as np

# Simulation : Ω = 100 résultats
Omega = np.arange(100)

# Événements
A = Omega[Omega % 2 == 0]  # Nombres pairs
B = Omega[Omega % 3 == 0]  # Multiples de 3

# Union et intersection
A_union_B = np.union1d(A, B)
A_inter_B = np.intersect1d(A, B)

# Complémentaires
A_comp = np.setdiff1d(Omega, A)
B_comp = np.setdiff1d(Omega, B)

# Loi 1 : (A ∪ B)^c = A^c ∩ B^c
gauche_1 = np.setdiff1d(Omega, A_union_B)
droite_1 = np.intersect1d(A_comp, B_comp)

# Loi 2 : (A ∩ B)^c = A^c ∪ B^c
gauche_2 = np.setdiff1d(Omega, A_inter_B)
droite_2 = np.union1d(A_comp, B_comp)

print("VÉRIFICATION LOIS DE DE MORGAN")
print("="*50)
print(f"Loi 1 : (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ")
print(f"  Égalité vérifiée : {np.array_equal(gauche_1, droite_1)}")

print(f"\nLoi 2 : (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ")
print(f"  Égalité vérifiée : {np.array_equal(gauche_2, droite_2)}")
```

---

### 2. Axiomes de Kolmogorov (Fondation Mathématique)

#### 2.1 Les Trois Axiomes Fondamentaux

En 1933, Andrey Kolmogorov formalisa la théorie des probabilités avec **3 axiomes** qui définissent rigoureusement ce qu'est une probabilité.

**Probabilité** : Fonction $$P: \mathcal{F} \to [0,1]$$ où $$\mathcal{F}$$ est l'ensemble des événements.

**Axiome 1 (Non-négativité)** :

$$P(A) \geq 0 \quad \forall A \in \mathcal{F}$$

*Probabilité toujours positive ou nulle.*

**Axiome 2 (Normalisation)** :

$$P(\Omega) = 1$$

*La probabilité de l'événement certain (tout l'espace) vaut 1.*

**Axiome 3 (Additivité dénombrable)** :

Pour événements **mutuellement exclusifs** $$A_1, A_2, \ldots$$ :

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

*La probabilité de l'union d'événements disjoints = somme de leurs probabilités.*

**Pourquoi ces axiomes ?**

Ces 3 axiomes simples suffisent à dériver **TOUTES** les règles de probabilité ! C'est la beauté des mathématiques : partir de fondations minimales et construire une théorie complète.

#### 2.2 Propriétés Dérivées des Axiomes

**Propriété 1 : Complémentaire**

$$P(A^c) = 1 - P(A)$$

**Démonstration** :
- $$A$$ et $$A^c$$ sont mutuellement exclusifs et $$A \cup A^c = \Omega$$
- Par axiome 3 : $$P(A \cup A^c) = P(A) + P(A^c)$$
- Par axiome 2 : $$P(\Omega) = 1$$
- Donc : $$P(A) + P(A^c) = 1$$ $$\Rightarrow$$ $$P(A^c) = 1 - P(A)$$

**Propriété 2 : Probabilité de l'ensemble vide**

$$P(\emptyset) = 0$$

**Démonstration** :
- $$\emptyset = \Omega^c$$
- Donc $$P(\emptyset) = P(\Omega^c) = 1 - P(\Omega) = 1 - 1 = 0$$

**Propriété 3 : Monotonie**

Si $$A \subseteq B$$ alors $$P(A) \leq P(B)$$

**Propriété 4 : Règle d'addition (générale)**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Pourquoi soustraire** $$P(A \cap B)$$ ?

Car si on additionne simplement $$P(A) + P(B)$$, on **compte deux fois** l'intersection !

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Visualisation du problème de double comptage
for idx, (ax, title) in enumerate([(axes[0], 'Sans correction\nP(A) + P(B) compte 2× intersection'),
                                     (axes[1], 'Avec correction\nP(A∪B) = P(A) + P(B) - P(A∩B)')]):
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                              edgecolor='black', facecolor='white', alpha=0.3)
    
    # Cercles avec transparence pour voir intersection
    circle_A = patches.Circle((0.35, 0.5), 0.2, 
                               facecolor='blue', edgecolor='black', 
                               linewidth=2, alpha=0.3)
    circle_B = patches.Circle((0.65, 0.5), 0.2, 
                               facecolor='red', edgecolor='black', 
                               linewidth=2, alpha=0.3)
    
    ax.add_patch(rect)
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)
    
    # Annotation intersection
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    ax.text(0.5, 0.9, 'A ∩ B\n(compté 2× à gauche)', 
            ha='center', fontsize=10, fontweight='bold', color='purple')
    
    ax.text(0.25, 0.5, 'A', fontsize=16, ha='center', fontweight='bold')
    ax.text(0.75, 0.5, 'B', fontsize=16, ha='center', fontweight='bold')
    ax.text(0.5, 0.5, 'A∩B', fontsize=10, ha='center', fontweight='bold', color='purple')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Exemple numérique
print("EXEMPLE : Lancer un dé")
print("="*50)
print("A = {2, 4, 6} (pairs)")
print("B = {3, 6} (multiples de 3)")
print("A ∩ B = {6}")
print()
print("P(A) = 3/6 = 0.5")
print("P(B) = 2/6 ≈ 0.333")
print("P(A ∩ B) = 1/6 ≈ 0.167")
print()
print("SANS CORRECTION :")
print("  P(A) + P(B) = 0.5 + 0.333 = 0.833 ❌ (FAUX, compte 6 deux fois)")
print()
print("AVEC CORRECTION :")
print("  P(A ∪ B) = P(A) + P(B) - P(A ∩ B)")
print("  P(A ∪ B) = 0.5 + 0.333 - 0.167 = 0.667 ✅")
print()
print("Vérification directe : A ∪ B = {2, 3, 4, 6} → 4/6 = 0.667 ✅")
```

**Sources académiques** :
- Kolmogorov, A. (1933). *Grundbegriffe der Wahrscheinlichkeitsrechnung* (Foundations of Probability Theory).
- [Axioms of Probability - MIT OpenCourseWare](https://ocw.mit.edu/courses/mathematics/)

---

### 3. Probabilités Conditionnelles

#### 3.1 Définition et Motivation

**Probabilité conditionnelle** : Probabilité qu'un événement $$A$$ se produise **sachant** qu'un événement $$B$$ s'est produit.

**Notation** : $$P(A|B)$$ (lu "probabilité de A sachant B")

**Formule fondamentale** :

$$P(A|B) = \frac{P(A \cap B)}{P(B)} \quad \text{si } P(B) > 0$$

**Intuition** :

Quand on sait que $$B$$ s'est produit, notre **espace des possibles se restreint** à $$B$$. La probabilité de $$A$$ devient la proportion de $$A \cap B$$ dans ce nouvel espace $$B$$.

**Visualisation** :

```
Espace original Ω               Après observation de B
┌──────────────────┐            ┌──────────────────┐
│      ┌─────────┐ │            │   B devient Ω    │
│   A  │    B    │ │    →       │   ┌──────────┐   │
│  ┌───┼──┐      │ │            │   │  A ∩ B   │   │
│  │ A∩B  │      │ │            │   └──────────┘   │
│  └───┼──┘      │ │            │                  │
│      └─────────┘ │            └──────────────────┘
└──────────────────┘            
                               P(A|B) = P(A∩B) / P(B)
```

#### 3.2 Exemples Concrets

**Exemple 1 : Tirage de cartes**

```python
import numpy as np

# Jeu de 52 cartes
total_cartes = 52
n_coeurs = 13  # 13 cartes de coeur
n_rois = 4     # 4 rois
n_roi_coeur = 1  # 1 roi de coeur

# Question 1 : P(Roi) ?
p_roi = n_rois / total_cartes
print("Question 1 : Probabilité de tirer un roi")
print(f"  P(Roi) = {n_rois}/{total_cartes} = {p_roi:.4f}")

# Question 2 : P(Roi | Coeur) ?
# On sait que c'est un coeur → espace réduit à 13 cartes
p_roi_sachant_coeur = n_roi_coeur / n_coeurs
print(f"\nQuestion 2 : Probabilité d'un roi SACHANT que c'est un coeur")
print(f"  P(Roi | Coeur) = {n_roi_coeur}/{n_coeurs} = {p_roi_sachant_coeur:.4f}")

# Vérification avec formule
p_roi_et_coeur = n_roi_coeur / total_cartes
p_coeur = n_coeurs / total_cartes
p_roi_sachant_coeur_formule = p_roi_et_coeur / p_coeur

print(f"\nVérification avec formule P(A|B) = P(A∩B) / P(B) :")
print(f"  P(Roi ∩ Coeur) = {n_roi_coeur}/{total_cartes} = {p_roi_et_coeur:.4f}")
print(f"  P(Coeur) = {n_coeurs}/{total_cartes} = {p_coeur:.4f}")
print(f"  P(Roi | Coeur) = {p_roi_et_coeur:.4f} / {p_coeur:.4f} = {p_roi_sachant_coeur_formule:.4f} ✅")

print(f"\n💡 L'information 'c'est un coeur' change la probabilité :")
print(f"   {p_roi:.4f} (sans info) → {p_roi_sachant_coeur:.4f} (avec info)")
```

**Exemple 2 : Test médical (reprise)**

```python
# Test de dépistage maladie
p_maladie = 0.01  # Prévalence 1%
p_test_pos_sachant_malade = 0.95  # Sensibilité
p_test_pos_sachant_sain = 0.05    # Taux faux positifs (1 - spécificité)

# Question : P(Maladie | Test+) ?

# Étape 1 : P(Test+)
p_test_pos = (p_test_pos_sachant_malade * p_maladie + 
              p_test_pos_sachant_sain * (1 - p_maladie))

# Étape 2 : P(Maladie | Test+) avec formule conditionnelle
p_maladie_sachant_test_pos = (p_test_pos_sachant_malade * p_maladie) / p_test_pos

print("TEST MÉDICAL")
print("="*50)
print(f"Prévalence maladie : {p_maladie:.1%}")
print(f"Sensibilité test : {p_test_pos_sachant_malade:.1%}")
print(f"Spécificité test : {1-p_test_pos_sachant_sain:.1%}")
print()
print(f"P(Test+ | Maladie) = {p_test_pos_sachant_malade:.1%}")
print(f"P(Test+ | Sain) = {p_test_pos_sachant_sain:.1%}")
print()
print(f"P(Test+) = {p_test_pos:.4f}")
print(f"P(Maladie | Test+) = {p_maladie_sachant_test_pos:.1%}")
print()
print(f"⚠️ Test positif ≠ 95% de chance d'être malade !")
print(f"   Vraie probabilité = {p_maladie_sachant_test_pos:.1%} (à cause de la faible prévalence)")
```

#### 3.3 Règle de Multiplication

De la définition de probabilité conditionnelle :

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

**Généralisation** (règle de la chaîne) :

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})$$

**Exemple : Tirage sans remise**

```python
# Urne avec 5 boules rouges et 3 boules blanches
# Tirer 2 boules SANS remise

n_rouges = 5
n_blanches = 3
total = n_rouges + n_blanches

# Probabilité de tirer 2 rouges consécutives

# P(R1) = probabilité 1ère rouge
p_r1 = n_rouges / total

# P(R2|R1) = probabilité 2ème rouge SACHANT 1ère rouge
# (il reste 4 rouges sur 7 boules)
p_r2_sachant_r1 = (n_rouges - 1) / (total - 1)

# P(R1 ∩ R2) = P(R1) × P(R2|R1)
p_deux_rouges = p_r1 * p_r2_sachant_r1

print("TIRAGE SANS REMISE")
print("="*50)
print(f"Urne : {n_rouges} rouges, {n_blanches} blanches")
print()
print(f"P(R₁) = {n_rouges}/{total} = {p_r1:.4f}")
print(f"P(R₂|R₁) = {n_rouges-1}/{total-1} = {p_r2_sachant_r1:.4f}")
print()
print(f"P(R₁ ∩ R₂) = P(R₁) × P(R₂|R₁)")
print(f"           = {p_r1:.4f} × {p_r2_sachant_r1:.4f}")
print(f"           = {p_deux_rouges:.4f}")
print()
print(f"💡 La probabilité du 2ème tirage DÉPEND du 1er (sans remise)")
```

---

### 4. Théorème de Bayes

#### 4.1 Formulation et Dérivation

**Théorème de Bayes** (1763) :

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Dérivation** (très simple depuis les probabilités conditionnelles) :

1. Par définition : $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
2. Par symétrie : $$P(B|A) = \frac{P(A \cap B)}{P(A)}$$ donc $$P(A \cap B) = P(B|A) \cdot P(A)$$
3. Substitution : $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Terminologie bayésienne** :

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

- $$P(A)$$ : **Prior** (probabilité a priori, avant observation)
- $$P(B|A)$$ : **Likelihood** (vraisemblance, probabilité des données sachant hypothèse)
- $$P(B)$$ : **Evidence** (probabilité des données, constante de normalisation)
- $$P(A|B)$$ : **Posterior** (probabilité a posteriori, après observation)

**En mots** :

*"Posterior = (Likelihood × Prior) / Evidence"*

**Pourquoi est-ce révolutionnaire ?**

Bayes permet **d'inverser** le conditionnement :
- On observe $$B$$ (les données)
- On veut connaître $$P(A|B)$$ (hypothèse sachant données)
- Mais on connaît plus facilement $$P(B|A)$$ (données sachant hypothèse)

**Exemple** : Test médical
- Facile de mesurer : $$P(\text{Test+} | \text{Maladie})$$ (en laboratoire, sur patients malades)
- Difficile de mesurer directement : $$P(\text{Maladie} | \text{Test+})$$
- **Bayes permet de calculer le 2ème à partir du 1er !**

#### 4.2 Forme Étendue avec Partition

Souvent, on ne connaît pas directement $$P(B)$$. On utilise alors **la loi des probabilités totales** :

$$P(B) = \sum_{i=1}^{n} P(B|A_i) \cdot P(A_i)$$

où $$A_1, \ldots, A_n$$ est une **partition** de $$\Omega$$ (mutuellement exclusifs et couvrent tout).

**Théorème de Bayes étendu** :

$$P(A_j|B) = \frac{P(B|A_j) \cdot P(A_j)}{\sum_{i=1}^{n} P(B|A_i) \cdot P(A_i)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Exemple : 3 usines produisent des pièces
usines = ['Usine A', 'Usine B', 'Usine C']

# Prior : Répartition de production
p_usines = np.array([0.50, 0.30, 0.20])  # Usine A produit 50%, B 30%, C 20%

# Likelihood : Taux de défaut par usine
p_defaut_sachant_usine = np.array([0.02, 0.04, 0.10])  # A: 2%, B: 4%, C: 10%

# Question : Une pièce est défectueuse. D'où vient-elle probablement ?

# Evidence : P(Défaut)
p_defaut = np.sum(p_defaut_sachant_usine * p_usines)

# Posterior : P(Usine | Défaut)
p_usine_sachant_defaut = (p_defaut_sachant_usine * p_usines) / p_defaut

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prior
axes[0].bar(usines, p_usines, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Probabilité')
axes[0].set_title('PRIOR : P(Usine)\nAvant observation', fontweight='bold', fontsize=12)
axes[0].set_ylim(0, 0.6)
axes[0].grid(alpha=0.3, axis='y')
for i, val in enumerate(p_usines):
    axes[0].text(i, val + 0.02, f'{val:.1%}', ha='center', fontweight='bold')

# Posterior
axes[1].bar(usines, p_usine_sachant_defaut, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Probabilité')
axes[1].set_title('POSTERIOR : P(Usine | Défaut)\nAprès observation pièce défectueuse', fontweight='bold', fontsize=12)
axes[1].set_ylim(0, 0.6)
axes[1].grid(alpha=0.3, axis='y')
for i, val in enumerate(p_usine_sachant_defaut):
    axes[1].text(i, val + 0.02, f'{val:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Résultats détaillés
print("THÉORÈME DE BAYES : Origine de pièce défectueuse")
print("="*70)
print(f"\nPRIOR (avant observation) :")
for i, usine in enumerate(usines):
    print(f"  P({usine}) = {p_usines[i]:.1%}")

print(f"\nLIKELIHOOD (taux de défaut par usine) :")
for i, usine in enumerate(usines):
    print(f"  P(Défaut | {usine}) = {p_defaut_sachant_usine[i]:.1%}")

print(f"\nEVIDENCE :")
print(f"  P(Défaut) = {p_defaut:.4f}")

print(f"\nPOSTERIOR (après observation pièce défectueuse) :")
for i, usine in enumerate(usines):
    print(f"  P({usine} | Défaut) = {p_usine_sachant_defaut[i]:.1%}")

print(f"\n💡 INTERPRÉTATION :")
print(f"   Avant observation : Usine A la plus probable (50%)")
print(f"   Après observation défaut : Usine A toujours la plus probable (33%)")
print(f"   MAIS Usine C (seulement 20% de production) représente maintenant 26% des défauts !")
print(f"   → Son fort taux de défaut (10%) augmente sa probabilité a posteriori")
```

#### 4.3 Applications Classiques du Théorème de Bayes

**Application 1 : Naive Bayes Classifier**

```python
import numpy as np
from collections import Counter

# Données d'entraînement : emails (mots) et labels (spam/ham)
emails_spam = [
    "viagra gratuit offre",
    "gratuit viagra maintenant",
    "offre spéciale gratuit"
]

emails_ham = [
    "réunion projet lundi",
    "rapport projet à envoyer",
    "lundi réunion équipe"
]

# Compter les mots
mots_spam = ' '.join(emails_spam).split()
mots_ham = ' '.join(emails_ham).split()

vocab_spam = Counter(mots_spam)
vocab_ham = Counter(mots_ham)

# Prior
n_spam = len(emails_spam)
n_ham = len(emails_ham)
p_spam = n_spam / (n_spam + n_ham)
p_ham = n_ham / (n_spam + n_ham)

print("NAIVE BAYES CLASSIFIER - Spam Filter")
print("="*60)
print(f"\nPRIOR :")
print(f"  P(Spam) = {p_spam:.2f}")
print(f"  P(Ham)  = {p_ham:.2f}")

# Nouvel email à classifier
nouveau = "gratuit offre spéciale"
mots_nouveau = nouveau.split()

print(f"\nNouvel email : '{nouveau}'")
print(f"\nCalcul Naive Bayes (hypothèse d'indépendance des mots) :")

# Likelihood avec lissage de Laplace (éviter P=0)
def likelihood(mots, vocab, total_mots, vocab_size):
    prob = 1.0
    for mot in mots:
        # P(mot | classe) = (count + 1) / (total + vocab_size)
        count = vocab.get(mot, 0)
        p_mot = (count + 1) / (total_mots + vocab_size)
        prob *= p_mot
    return prob

total_mots_spam = len(mots_spam)
total_mots_ham = len(mots_ham)
vocab_size = len(set(mots_spam + mots_ham))

likelihood_spam = likelihood(mots_nouveau, vocab_spam, total_mots_spam, vocab_size)
likelihood_ham = likelihood(mots_nouveau, vocab_ham, total_mots_ham, vocab_size)

# Posterior (non normalisé)
posterior_spam = likelihood_spam * p_spam
posterior_ham = likelihood_ham * p_ham

# Normalisation
evidence = posterior_spam + posterior_ham
posterior_spam_norm = posterior_spam / evidence
posterior_ham_norm = posterior_ham / evidence

print(f"\n  Likelihood(mots | Spam) = {likelihood_spam:.6f}")
print(f"  Likelihood(mots | Ham)  = {likelihood_ham:.6f}")

print(f"\n  Posterior(Spam | mots) ∝ {posterior_spam:.6f}")
print(f"  Posterior(Ham | mots)  ∝ {posterior_ham:.6f}")

print(f"\n  P(Spam | mots) = {posterior_spam_norm:.1%}")
print(f"  P(Ham | mots)  = {posterior_ham_norm:.1%}")

print(f"\n✅ Classification : {'SPAM' if posterior_spam_norm > posterior_ham_norm else 'HAM'}")
print(f"   (Probabilité : {max(posterior_spam_norm, posterior_ham_norm):.1%})")
```

**Sources académiques** :
- Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances". *Philosophical Transactions of the Royal Society*.
- [Bayes' Theorem - Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/bayes-theorem/)

---

### 5. Indépendance

#### 5.1 Définition

**Événements indépendants** : $$A$$ et $$B$$ sont indépendants si :

$$P(A \cap B) = P(A) \cdot P(B)$$

**Équivalent** : $$P(A|B) = P(A)$$ (si $$P(B) > 0$$)

**Intuition** : L'information que $$B$$ s'est produit **ne change pas** la probabilité de $$A$$.

**⚠️ ATTENTION** : Indépendance ≠ Mutuellement exclusifs !

| Relation | Définition | Implication |
|----------|------------|-------------|
| **Mutuellement exclusifs** | $$A \cap B = \emptyset$$ | Si $$A$$ se produit, $$B$$ **ne peut pas** se produire |
| **Indépendants** | $$P(A \cap B) = P(A) \cdot P(B)$$ | Si $$A$$ se produit, probabilité de $$B$$ **inchangée** |

**Exemple** :

```python
# Lancer 2 dés équilibrés

# Événements
# A = "Premier dé = 6"
# B = "Deuxième dé = 6"

p_A = 1/6
p_B = 1/6
p_A_et_B = 1/36  # (6,6) est 1 cas sur 36

# Vérification indépendance
print("INDÉPENDANCE : 2 dés")
print("="*50)
print(f"P(A) = {p_A:.4f}")
print(f"P(B) = {p_B:.4f}")
print(f"P(A ∩ B) = {p_A_et_B:.4f}")
print(f"P(A) × P(B) = {p_A * p_B:.4f}")
print()
print(f"Indépendants ? {np.isclose(p_A_et_B, p_A * p_B)}")
print()
print("💡 Le résultat du 2ème dé ne dépend pas du 1er → Indépendants")

print("\n" + "="*50)
print("CONTRE-EXEMPLE : Mutuellement exclusifs ≠ Indépendants")
print("="*50)

# Événements sur UN SEUL dé
# C = "Dé = 2"
# D = "Dé = 6"

p_C = 1/6
p_D = 1/6
p_C_et_D = 0  # Impossible d'avoir 2 ET 6 sur un seul dé

print(f"P(C) = {p_C:.4f}")
print(f"P(D) = {p_D:.4f}")
print(f"P(C ∩ D) = {p_C_et_D:.4f}")
print(f"P(C) × P(D) = {p_C * p_D:.4f}")
print()
print(f"Mutuellement exclusifs ? {p_C_et_D == 0} (C ∩ D = ∅)")
print(f"Indépendants ? {np.isclose(p_C_et_D, p_C * p_D)} (0 ≠ 0.028)")
print()
print("⚠️ Si C et D mutuellement exclusifs ET P(C)>0, P(D)>0")
print("   → Alors FORCÉMENT dépendants !")
print("   (Savoir que C s'est produit implique P(D|C) = 0 ≠ P(D))")
```

#### 5.2 Indépendance Mutuelle vs Indépendance Par Paires

**Indépendance par paires** : $$P(A_i \cap A_j) = P(A_i) \cdot P(A_j)$$ pour tout $$i \neq j$$

**Indépendance mutuelle** (plus forte) :

$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdot P(A_{i_2}) \cdots P(A_{i_k})$$

pour toute sous-collection.

**⚠️ Indépendance par paires n'implique PAS indépendance mutuelle !**

**Contre-exemple classique** :

```python
# 2 pièces équilibrées, on observe 3 événements
# A = "Pièce 1 = Pile"
# B = "Pièce 2 = Pile"
# C = "Nombre total de Piles est pair"

# Espace : {(P,P), (P,F), (F,P), (F,F)}
Omega = [('P','P'), ('P','F'), ('F','P'), ('F','F')]

# Événements
A = [('P','P'), ('P','F')]  # Pièce 1 = Pile
B = [('P','P'), ('F','P')]  # Pièce 2 = Pile
C = [('P','P'), ('F','F')]  # Nombre Piles pair (0 ou 2)

p_A = len(A) / len(Omega)
p_B = len(B) / len(Omega)
p_C = len(C) / len(Omega)

# Intersections 2 à 2
A_inter_B = [x for x in Omega if x in A and x in B]
A_inter_C = [x for x in Omega if x in A and x in C]
B_inter_C = [x for x in Omega if x in B and x in C]

p_A_inter_B = len(A_inter_B) / len(Omega)
p_A_inter_C = len(A_inter_C) / len(Omega)
p_B_inter_C = len(B_inter_C) / len(Omega)

# Intersection des 3
A_inter_B_inter_C = [x for x in Omega if x in A and x in B and x in C]
p_A_inter_B_inter_C = len(A_inter_B_inter_C) / len(Omega)

print("INDÉPENDANCE : Par paires vs Mutuelle")
print("="*60)
print(f"P(A) = {p_A:.2f}, P(B) = {p_B:.2f}, P(C) = {p_C:.2f}")
print()
print("INDÉPENDANCE PAR PAIRES :")
print(f"  P(A ∩ B) = {p_A_inter_B:.2f}, P(A)×P(B) = {p_A * p_B:.2f} → {'✅' if np.isclose(p_A_inter_B, p_A*p_B) else '❌'}")
print(f"  P(A ∩ C) = {p_A_inter_C:.2f}, P(A)×P(C) = {p_A * p_C:.2f} → {'✅' if np.isclose(p_A_inter_C, p_A*p_C) else '❌'}")
print(f"  P(B ∩ C) = {p_B_inter_C:.2f}, P(B)×P(C) = {p_B * p_C:.2f} → {'✅' if np.isclose(p_B_inter_C, p_B*p_C) else '❌'}")
print()
print("INDÉPENDANCE MUTUELLE :")
print(f"  P(A ∩ B ∩ C) = {p_A_inter_B_inter_C:.2f}")
print(f"  P(A)×P(B)×P(C) = {p_A * p_B * p_C:.2f}")
print(f"  → {'✅ Indépendants mutuellement' if np.isclose(p_A_inter_B_inter_C, p_A*p_B*p_C) else '❌ PAS indépendants mutuellement'}")
print()
print("💡 A, B, C sont indépendants PAR PAIRES mais PAS mutuellement !")
print("   Connaître A et B détermine complètement C")
```

---

## 💡 Compréhension Intuitive

### L'Analogie du Détective

**Probabilités = Enquête policière**

- **Prior** : Suspect le plus probable AVANT les preuves (historique, statistiques)
- **Evidence** : Nouvelles preuves découvertes (empreintes, témoin)
- **Likelihood** : Probabilité de ces preuves SI ce suspect est coupable
- **Posterior** : Suspect le plus probable APRÈS les preuves

**Théorème de Bayes** = Comment les preuves modifient nos croyances de manière rationnelle.

**Exemple** :
- **Prior** : Suspect A (riche, respecté) : 10% coupable
- **Prior** : Suspect B (casier, proche victime) : 60% coupable
- **Evidence** : Empreintes trouvées
- **Likelihood** : P(Empreintes | A coupable) = 90%
- **Likelihood** : P(Empreintes | B coupable) = 20%

**Bayes calcule** : Malgré prior faible, les empreintes augmentent fortement P(A coupable | Empreintes).

### Questions Rapides

1. **Q1** : $$P(A) = 0.3$$, $$P(B) = 0.5$$, $$A \cap B = \emptyset$$. Indépendants ?
   - *Réponse* : NON (mutuellement exclusifs ≠ indépendants)

2. **Q2** : $$P(A|B) = P(A)$$. Que peut-on conclure ?
   - *Réponse* : $$A$$ et $$B$$ sont indépendants

3. **Q3** : Test positif, sensibilité 99%. Ai-je 99% de chance d'être malade ?
   - *Réponse* : NON ! Dépend de la prévalence (Bayes)

4. **Q4** : $$P(A \cup B) = P(A) + P(B)$$. Vrai si...?
   - *Réponse* : Vrai si $$A$$ et $$B$$ mutuellement exclusifs

---

## 💻 Implémentation Pratique

### Simulateur de Problèmes de Probabilités

```python
"""
Titre : Simulateur Monte Carlo pour vérifier calculs probabilistes
Objectif : Valider formules théoriques par simulation
"""

import numpy as np
import matplotlib.pyplot as plt

def simuler_evenements(n_simulations=100000):
    """
    Simule lancer de 2 dés et vérifie diverses probabilités.
    """
    print("SIMULATION MONTE CARLO : 2 dés")
    print("="*60)
    print(f"Nombre de simulations : {n_simulations:,}")
    print()
    
    # Simulation
    de1 = np.random.randint(1, 7, n_simulations)
    de2 = np.random.randint(1, 7, n_simulations)
    somme = de1 + de2
    
    # Événements
    A = (de1 == 6)  # Premier dé = 6
    B = (somme == 7)  # Somme = 7
    C = (de1 == de2)  # Doubles
    
    # Probabilités empiriques
    p_A_emp = np.mean(A)
    p_B_emp = np.mean(B)
    p_C_emp = np.mean(C)
    p_A_et_B_emp = np.mean(A & B)
    p_A_ou_B_emp = np.mean(A | B)
    
    # Probabilités théoriques
    p_A_theo = 1/6
    p_B_theo = 6/36  # (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
    p_C_theo = 6/36  # (1,1), ..., (6,6)
    p_A_et_B_theo = 1/36  # (6,1)
    p_A_ou_B_theo = p_A_theo + p_B_theo - p_A_et_B_theo
    
    # Comparaison
    resultats = [
        ('P(Dé1=6)', p_A_theo, p_A_emp),
        ('P(Somme=7)', p_B_theo, p_B_emp),
        ('P(Doubles)', p_C_theo, p_C_emp),
        ('P(Dé1=6 ET Somme=7)', p_A_et_B_theo, p_A_et_B_emp),
        ('P(Dé1=6 OU Somme=7)', p_A_ou_B_theo, p_A_ou_B_emp),
    ]
    
    print(f"{'Événement':<30} | {'Théorique':>12} | {'Empirique':>12} | {'Erreur':>10}")
    print("-"*80)
    
    for nom, theo, emp in resultats:
        erreur = abs(theo - emp)
        print(f"{nom:<30} | {theo:>12.6f} | {emp:>12.6f} | {erreur:>10.6f}")
    
    print()
    print("💡 Les valeurs empiriques convergent vers théoriques (Loi des Grands Nombres)")
    
    # Visualisation convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convergence de P(Somme=7)
    n_vals = np.logspace(2, np.log10(n_simulations), 50, dtype=int)
    p_empiriques = []
    
    for n in n_vals:
        de1_sub = de1[:n]
        de2_sub = de2[:n]
        somme_sub = de1_sub + de2_sub
        p_empiriques.append(np.mean(somme_sub == 7))
    
    axes[0, 0].plot(n_vals, p_empiriques, linewidth=2, label='P empirique')
    axes[0, 0].axhline(p_B_theo, color='red', linestyle='--', linewidth=2, label=f'P théorique = {p_B_theo:.4f}')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Nombre de simulations')
    axes[0, 0].set_ylabel('Probabilité')
    axes[0, 0].set_title('Convergence : P(Somme=7)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Distribution de la somme
    axes[0, 1].hist(somme, bins=np.arange(2, 14)-0.5, density=True, 
                     edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Somme des dés')
    axes[0, 1].set_ylabel('Probabilité')
    axes[0, 1].set_title('Distribution de la Somme')
    axes[0, 1].set_xticks(range(2, 13))
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Test indépendance : P(Dé2=k | Dé1=6)
    de1_est_6 = de1 == 6
    p_de2_sachant_de1_6 = []
    
    for k in range(1, 7):
        p_cond = np.mean(de2[de1_est_6] == k)
        p_de2_sachant_de1_6.append(p_cond)
    
    axes[1, 0].bar(range(1, 7), p_de2_sachant_de1_6, alpha=0.7, 
                    edgecolor='black', color='lightgreen', label='P(Dé2=k | Dé1=6)')
    axes[1, 0].axhline(1/6, color='red', linestyle='--', linewidth=2, label='P(Dé2=k) = 1/6')
    axes[1, 0].set_xlabel('Valeur Dé 2')
    axes[1, 0].set_ylabel('Probabilité')
    axes[1, 0].set_title('Test Indépendance : P(Dé2 | Dé1=6)')
    axes[1, 0].set_xticks(range(1, 7))
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Tableau probabilités conditionnelles
    axes[1, 1].axis('off')
    
    # Calcul P(Somme=k | Dé1=6)
    prob_table = []
    for k in range(7, 13):
        p_somme_sachant_de1_6 = np.mean(somme[de1_est_6] == k)
        p_somme_theo = (1 if k >= 7 and k <= 12 else 0) / 6
        prob_table.append([f'{k}', f'{p_somme_theo:.4f}', f'{p_somme_sachant_de1_6:.4f}'])
    
    table_data = [['Somme', 'P(Somme|Dé1=6) Théo', 'Empirique']] + prob_table
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                              colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colorer header
    for i in range(3):
        table[(0, i)].set_facecolor('lightgray')
        table[(0, i)].set_text_props(weight='bold')
    
    axes[1, 1].set_title('Probabilités Conditionnelles', fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()

# Exécution
simuler_evenements(n_simulations=100000)
```

---

### Calculateur Bayésien Interactif

```python
"""
Titre : Calculateur théorème de Bayes pour diagnostic médical
Objectif : Outil pour calculer probabilités a posteriori
"""

def calculateur_bayes_medical(prevalence, sensibilite, specificite, verbose=True):
    """
    Calcule P(Maladie|Test+) avec Bayes.
    
    Args:
        prevalence (float): P(Maladie)
        sensibilite (float): P(Test+|Maladie)
        specificite (float): P(Test-|Pas Maladie)
        verbose (bool): Afficher détails
    
    Returns:
        dict: Toutes les probabilités calculées
    """
    # Calculs
    p_maladie = prevalence
    p_pas_maladie = 1 - prevalence
    
    p_test_pos_si_malade = sensibilite
    p_test_neg_si_malade = 1 - sensibilite
    
    p_test_neg_si_sain = specificite
    p_test_pos_si_sain = 1 - specificite
    
    # Loi probabilités totales
    p_test_pos = (p_test_pos_si_malade * p_maladie + 
                  p_test_pos_si_sain * p_pas_maladie)
    p_test_neg = (p_test_neg_si_malade * p_maladie + 
                  p_test_neg_si_sain * p_pas_maladie)
    
    # Bayes
    p_malade_si_test_pos = (p_test_pos_si_malade * p_maladie) / p_test_pos
    p_malade_si_test_neg = (p_test_neg_si_malade * p_maladie) / p_test_neg
    
    p_sain_si_test_pos = (p_test_pos_si_sain * p_pas_maladie) / p_test_pos
    p_sain_si_test_neg = (p_test_neg_si_sain * p_pas_maladie) / p_test_neg
    
    resultats = {
        'prevalence': prevalence,
        'sensibilite': sensibilite,
        'specificite': specificite,
        'p_test_pos': p_test_pos,
        'p_test_neg': p_test_neg,
        'p_malade_si_test_pos': p_malade_si_test_pos,
        'p_malade_si_test_neg': p_malade_si_test_neg,
        'p_sain_si_test_pos': p_sain_si_test_pos,
        'p_sain_si_test_neg': p_sain_si_test_neg
    }
    
    if verbose:
        print("CALCULATEUR BAYÉSIEN - Diagnostic Médical")
        print("="*60)
        print(f"\nDONNÉES D'ENTRÉE :")
        print(f"  Prévalence (taux de base)  : {prevalence:.1%}")
        print(f"  Sensibilité (VP rate)      : {sensibilite:.1%}")
        print(f"  Spécificité (VN rate)      : {specificite:.1%}")
        print(f"  Taux faux positifs         : {1-specificite:.1%}")
        print(f"  Taux faux négatifs         : {1-sensibilite:.1%}")
        
        print(f"\nPROBABILITÉS CALCULÉES :")
        print(f"  P(Test +) = {p_test_pos:.4f} ({p_test_pos:.1%})")
        print(f"  P(Test -) = {p_test_neg:.4f} ({p_test_neg:.1%})")
        
        print(f"\nRÉSULTATS BAYES (Probabilités a posteriori) :")
        print(f"  P(Malade | Test +) = {p_malade_si_test_pos:.4f} ({p_malade_si_test_pos:.1%}) ⭐")
        print(f"  P(Sain | Test +)   = {p_sain_si_test_pos:.4f} ({p_sain_si_test_pos:.1%})")
        print(f"  P(Malade | Test -) = {p_malade_si_test_neg:.4f} ({p_malade_si_test_neg:.1%})")
        print(f"  P(Sain | Test -)   = {p_sain_si_test_neg:.4f} ({p_sain_si_test_neg:.1%}) ⭐")
        
        print(f"\n💡 INTERPRÉTATION :")
        if p_malade_si_test_pos < 0.5:
            print(f"   ⚠️ Test + : PLUS probable d'être sain ({p_sain_si_test_pos:.1%}) que malade ({p_malade_si_test_pos:.1%}) !")
            print(f"   → Taux de base (prévalence) très faible")
        else:
            print(f"   ✅ Test + : Probable d'être malade ({p_malade_si_test_pos:.1%})")
        
        if p_sain_si_test_neg > 0.99:
            print(f"   ✅ Test - : Très fiable pour écarter maladie ({p_sain_si_test_neg:.1%})")
        
        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Avant test (Prior)
        axes[0].bar(['Malade', 'Sain'], [p_maladie, p_pas_maladie], 
                     color=['red', 'green'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Probabilité')
        axes[0].set_title('AVANT Test (Prior)', fontweight='bold', fontsize=12)
        axes[0].set_ylim(0, 1)
        axes[0].grid(alpha=0.3, axis='y')
        for i, (label, val) in enumerate([('Malade', p_maladie), ('Sain', p_pas_maladie)]):
            axes[0].text(i, val + 0.03, f'{val:.1%}', ha='center', fontweight='bold', fontsize=11)
        
        # Après test positif (Posterior)
        axes[1].bar(['Malade', 'Sain'], [p_malade_si_test_pos, p_sain_si_test_pos], 
                     color=['red', 'green'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Probabilité')
        axes[1].set_title('APRÈS Test + (Posterior)', fontweight='bold', fontsize=12)
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.3, axis='y')
        for i, (label, val) in enumerate([('Malade', p_malade_si_test_pos), ('Sain', p_sain_si_test_pos)]):
            axes[1].text(i, val + 0.03, f'{val:.1%}', ha='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    return resultats

# Exemples
print("EXEMPLE 1 : Maladie rare")
print("-"*60)
calculateur_bayes_medical(prevalence=0.01, sensibilite=0.95, specificite=0.95)

print("\n\n")
print("EXEMPLE 2 : Maladie fréquente")
print("-"*60)
calculateur_bayes_medical(prevalence=0.20, sensibilite=0.95, specificite=0.95)
```

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Confondre P(A|B) et P(B|A)

**Erreur du procureur** (Prosecutor's Fallacy)

```python
# Cas réel : Sally Clark (UK, 1999)
# 2 de ses bébés morts subitement (SIDS)

# Expert affirme : P(2 SIDS | Innocent) = 1/73 millions
# Conclusion tribunal : Donc P(Innocent | 2 SIDS) = 1/73 millions → Coupable !

# ❌ ERREUR : Confondre P(Données | Hypothèse) avec P(Hypothèse | Données)

print("ERREUR DU PROCUREUR")
print("="*60)
print("\n❌ FAUX raisonnement :")
print("  P(2 SIDS | Innocent) = 1 / 73,000,000 (très rare)")
print("  → Donc P(Innocent | 2 SIDS) ≈ 0")
print("  → Conclusion : Coupable")

print("\n✅ BON raisonnement (Bayes) :")

# On doit comparer 2 hypothèses
p_innocent = 0.999  # Prior : Très rare qu'une mère tue ses bébés
p_coupable = 0.001

p_2_sids_si_innocent = 1 / 73_000_000
p_2_sids_si_coupable = 0.10  # Si meurtre, probable de tuer les 2

# Evidence
p_2_sids = (p_2_sids_si_innocent * p_innocent + 
            p_2_sids_si_coupable * p_coupable)

# Posterior
p_innocent_sachant_2_sids = (p_2_sids_si_innocent * p_innocent) / p_2_sids
p_coupable_sachant_2_sids = (p_2_sids_si_coupable * p_coupable) / p_2_sids

print(f"  P(2 SIDS | Innocent) = {p_2_sids_si_innocent:.2e}")
print(f"  P(2 SIDS | Coupable) = {p_2_sids_si_coupable:.2f}")
print(f"  P(Innocent) prior = {p_innocent:.3f}")
print()
print(f"  P(Innocent | 2 SIDS) = {p_innocent_sachant_2_sids:.4f} ({p_innocent_sachant_2_sids:.1%})")
print(f"  P(Coupable | 2 SIDS) = {p_coupable_sachant_2_sids:.4f} ({p_coupable_sachant_2_sids:.1%})")
print()
print("💡 Malgré rareté de 2 SIDS, plus probable que innocente !")
print("   Sally Clark a été libérée après 3 ans de prison (erreur judiciaire)")
```

---

### ✅ Bonne Pratique 1 : Toujours Dessiner un Arbre/Tableau

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Problème : Maladie, Test médical
prevalence = 0.01
sensibilite = 0.95
specificite = 0.95

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Arbre de probabilités
# Niveau 1 : Maladie
ax.text(1, 9, 'Population', fontsize=14, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

ax.arrow(1, 8.7, 1, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(1, 8.7, -1, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')

ax.text(2, 7, f'Malade\n{prevalence:.1%}', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(0, 7, f'Sain\n{1-prevalence:.1%}', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Niveau 2 : Test (branche Malade)
ax.arrow(2, 6.7, 1, -1.5, head_width=0.15, head_length=0.15, fc='red', ec='red')
ax.arrow(2, 6.7, -0.5, -1.5, head_width=0.15, head_length=0.15, fc='red', ec='red')

ax.text(3, 4.8, f'Test+\n{sensibilite:.1%}', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
ax.text(1.5, 4.8, f'Test-\n{1-sensibilite:.1%}', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Niveau 2 : Test (branche Sain)
ax.arrow(0, 6.7, -1, -1.5, head_width=0.15, head_length=0.15, fc='green', ec='green')
ax.arrow(0, 6.7, 0.5, -1.5, head_width=0.15, head_length=0.15, fc='green', ec='green')

ax.text(-1, 4.8, f'Test-\n{specificite:.1%}', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(0.5, 4.8, f'Test+\n{1-specificite:.1%}', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

# Probabilités jointes (bas de l'arbre)
p_malade_et_test_pos = prevalence * sensibilite
p_malade_et_test_neg = prevalence * (1 - sensibilite)
p_sain_et_test_neg = (1 - prevalence) * specificite
p_sain_et_test_pos = (1 - prevalence) * (1 - specificite)

ax.text(3, 3.5, f'P(M ∩ T+)\n{p_malade_et_test_pos:.4f}', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.text(1.5, 3.5, f'P(M ∩ T-)\n{p_malade_et_test_neg:.4f}', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.text(-1, 3.5, f'P(S ∩ T-)\n{p_sain_et_test_neg:.4f}', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.text(0.5, 3.5, f'P(S ∩ T+)\n{p_sain_et_test_pos:.4f}', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Calculs Bayes (encadré)
p_test_pos = p_malade_et_test_pos + p_sain_et_test_pos
p_malade_si_test_pos = p_malade_et_test_pos / p_test_pos

ax.text(5, 7, 'THÉORÈME DE BAYES', fontsize=14, fontweight='bold', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

formule_text = f"""
P(Malade | Test+) = P(Test+ | Malade) × P(Malade) / P(Test+)

P(Test+) = P(Test+ ∩ Malade) + P(Test+ ∩ Sain)
         = {p_malade_et_test_pos:.4f} + {p_sain_et_test_pos:.4f}
         = {p_test_pos:.4f}

P(Malade | Test+) = {p_malade_et_test_pos:.4f} / {p_test_pos:.4f}
                  = {p_malade_si_test_pos:.4f}
                  = {p_malade_si_test_pos:.1%}
"""

ax.text(5, 5, formule_text, fontsize=10, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.title('Arbre de Probabilités + Théorème de Bayes', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("💡 Dessiner un arbre clarifie :")
print("   1. Toutes les branches possibles")
print("   2. Probabilités conditionnelles (sur les flèches)")
print("   3. Probabilités jointes (au bout des branches)")
print("   4. Application de Bayes devient mécanique")
```

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Fondamentaux

1. **"An Essay towards solving a Problem in the Doctrine of Chances"**
   - **Auteur** : Thomas Bayes (1763)
   - **URL** : [Royal Society Archives](https://royalsocietypublishing.org/)
   - **Contribution** : Théorème de Bayes original
   - **Niveau** : Historique, accessible

2. **"Grundbegriffe der Wahrscheinlichkeitsrechnung"**
   - **Auteur** : Andrey Kolmogorov (1933)
   - **Contribution** : Axiomatisation moderne des probabilités
   - **Niveau** : Mathématique avancé

3. **"The Unreasonable Effectiveness of Bayes' Theorem"**
   - Articles modernes sur applications ML/IA

---

### 📚 Ressources

**Livres** :
- **"Probability Theory: The Logic of Science"** - E. T. Jaynes (interprétation bayésienne)
- **"Introduction to Probability"** - Bertsekas & Tsitsiklis (MIT, gratuit en ligne)

**Cours en ligne** :
- [MIT 6.041 - Probabilistic Systems](https://ocw.mit.edu/)
- [Khan Academy - Probability](https://www.khanacademy.org/math/statistics-probability)

---

### 📖 Cours Connexes

**Suite naturelle** :
- [[random_variables]] - Variables aléatoires, espérance, variance
- [[common_distributions]] - Lois classiques (Bernoulli, Binomiale, Normale)
- [[conditional_probability_advanced]] - Espérance conditionnelle, indépendance conditionnelle

**Applications** :
- [[bayesian_foundations]] - Statistiques bayésiennes approfondies
- [[maximum_likelihood]] - Estimation par maximum de vraisemblance
- [[hypothesis_testing]] - Tests d'hypothèses (fréquentistes)

---

## 📝 Résumé Rapide

### Formules Clés

| Concept | Formule | Usage |
|---------|---------|-------|
| **Prob. conditionnelle** | $$P(A\|B) = \frac{P(A \cap B)}{P(B)}$$ | Restreindre espace |
| **Règle multiplication** | $$P(A \cap B) = P(A\|B) \cdot P(B)$$ | Prob. jointe |
| **Théorème de Bayes** | $$P(A\|B) = \frac{P(B\|A) \cdot P(A)}{P(B)}$$ | Inverser conditionnement |
| **Loi prob. totales** | $$P(B) = \sum_i P(B\|A_i) \cdot P(A_i)$$ | Calculer evidence |
| **Indépendance** | $$P(A \cap B) = P(A) \cdot P(B)$$ | Simplifier calculs |
| **Règle addition** | $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$ | Prob. union |

### Code Minimal

```python
import numpy as np

# Probabilité conditionnelle
def prob_cond(p_A_et_B, p_B):
    return p_A_et_B / p_B if p_B > 0 else 0

# Théorème de Bayes
def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Test indépendance
def sont_independants(p_A, p_B, p_A_et_B, tol=1e-6):
    return abs(p_A_et_B - p_A * p_B) < tol
```
