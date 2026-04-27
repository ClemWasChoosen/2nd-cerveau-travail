# UMAP (Uniform Manifold Approximation and Projection)

> **Résumé en une phrase** : UMAP est un algorithme de réduction de dimensionnalité non-linéaire fondé sur la topologie algébrique et la géométrie Riemannienne, qui préserve à la fois la structure locale et globale des données tout en étant significativement plus rapide que t-SNE.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-04-25 |
| **Dernière mise à jour** | 2026-04-25 |
| **Domaine** | Machine Learning - Apprentissage Non Supervisé |
| **Niveau** | Intermédiaire à Avancé |
| **Durée de lecture** | ~45 minutes |
| **Fichier** | `umap.md` |
| **Emplacement** | `/01_machine_learning/03_unsupervised/dimension_reduction/` |
| **Tags** | `#machine-learning` `#unsupervised` `#dimensionality-reduction` `#manifold-learning` `#topologie` `#visualisation` `#umap` |

### Prérequis

- [x] **Algèbre linéaire** - Vecteurs, matrices, distances
- [x] **Théorie des graphes** - Graphes k-NN, connectivité, composantes connexes
- [x] **Topologie algébrique** - Complexes simpliciaux, notions de base
- [x] **Probabilités** - Distributions, divergences
- [ ] **Géométrie différentielle** - Variétés Riemanniennes (sera expliqué dans ce cours)
- [ ] **Optimisation** - Descente de gradient stochastique (notions de base suffisantes)

### Cours connexes (Liens Zettelkasten)

- **Complémentaires** : [[tsne]] - Algorithme de réduction de dimensionnalité non-linéaire (structure locale)
- **Fondamentaux** : PCA - Réduction de dimensionnalité linéaire (si disponible)
- **Applications** : Visualisation d'embeddings (BERT, Word2Vec)

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

UMAP (Uniform Manifold Approximation and Projection) est une technique moderne de réduction de dimensionnalité développée par Leland McInnes, John Healy et James Melville en 2018. Contrairement à t-SNE qui se concentre sur la préservation de la structure locale, UMAP s'appuie sur des **fondements mathématiques rigoureux** en topologie algébrique et géométrie Riemannienne pour préserver à la fois les structures locales ET globales. Ce cours vous permettra de comprendre les bases théoriques solides qui sous-tendent UMAP, ses avantages par rapport aux alternatives, et quand l'utiliser dans vos projets de recherche.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Les fondements mathématiques de UMAP (topologie, manifolds, géométrie Riemannienne)
2. **Appliquer** : Choisir entre UMAP, t-SNE et PCA selon le contexte de votre projet
3. **Analyser** : Interpréter correctement les visualisations UMAP et identifier leurs limites
4. **Évaluer** : Comparer les résultats de différentes méthodes de réduction de dimensionnalité et justifier votre choix

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

La réduction de dimensionnalité est un problème central en machine learning moderne. Avec l'explosion des données haute dimension (embeddings de 768 dimensions pour BERT, features de milliers de dimensions en vision, données génomiques avec des dizaines de milliers de gènes), nous avons besoin de méthodes pour :

1. **Visualiser** ces données de manière compréhensible (2D/3D)
2. **Accélérer** les algorithmes downstream (clustering, classification)
3. **Comprendre** la structure intrinsèque des données
4. **Détecter** des patterns, anomalies, ou groupes

**Le problème fondamental** : Comment projeter des données de $$\mathbb{R}^d$$ (haute dimension) vers $$\mathbb{R}^2$$ ou $$\mathbb{R}^3$$ (basse dimension) en perdant le moins d'information structurelle possible ?

### Quel problème UMAP résout-il ?

**Limitations des approches existantes** :

- **PCA** : Rapide mais linéaire uniquement, ne capture pas les manifolds complexes
- **t-SNE** : Excellente structure locale mais :
  - Lent ($$O(n^2)$$ ou $$O(n \log n)$$ avec Barnes-Hut)
  - Ne préserve pas la structure globale
  - Paramètres difficiles à ajuster
  - Résultats non reproductibles (forte dépendance à l'initialisation)
  - Pas de transformation de nouveaux points (pas de embedding out-of-sample)

**Les contributions clés d'UMAP** :

1. ✅ **Fondements théoriques rigoureux** : basé sur la topologie et la géométrie, pas seulement heuristique
2. ✅ **Préserve structure locale ET globale** : équilibre ajustable
3. ✅ **Rapidité** : $$O(n)$$ après construction du graphe k-NN
4. ✅ **Scalabilité** : applicable à des millions de points
5. ✅ **Flexibilité** : supporte de nouvelles projections (out-of-sample embedding)
6. ✅ **Robustesse** : moins sensible aux hyperparamètres que t-SNE

### Applications dans le monde réel

1. **Visualisation d'embeddings** : 
   - Embeddings BERT/GPT (votre cas d'usage) : visualiser les relations sémantiques entre documents ou tokens
   - Word embeddings : comprendre l'espace sémantique
   - Image embeddings (ResNet, CLIP) : explorer la similarité visuelle

2. **Bioinformatique et génomique** :
   - Single-cell RNA sequencing : identifier des types cellulaires
   - Analyse de séquences protéiques : détecter des familles
   - Métagénomique : classifier des espèces microbiennes

3. **Anomaly detection** :
   - Détection de fraudes : points isolés dans l'espace UMAP
   - Cybersécurité : identifier des comportements anormaux
   - Quality control : détecter des défauts de production

4. **Preprocessing pour ML** :
   - Réduction de dimensionnalité avant clustering (K-means, DBSCAN)
   - Feature engineering : extraire des features bas-dimensionnelles informatives
   - Data exploration : comprendre la distribution avant modélisation

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Cette section construit les bases mathématiques rigoureuses d'UMAP. Nous commençons par la topologie, puis introduisons la géométrie Riemannienne, pour finalement comprendre comment UMAP les combine.

### 1. Concepts clés

#### 1.1 Manifolds (Variétés différentiables)

**Définition** :

Un **manifold** (variété) de dimension $$m$$ est un espace topologique qui, localement, ressemble à $$\mathbb{R}^m$$. Intuitivement, c'est une surface courbe qui peut être "aplatie" localement.

**Pourquoi cette définition ?**

L'hypothèse fondamentale de UMAP (et de nombreux algorithmes de manifold learning) est que **les données haute dimension résident sur ou près d'un manifold de dimension intrinsèque plus faible**. Par exemple :
- Images de visages : bien que représentées en millions de pixels, elles vivent sur un manifold de dimension bien plus faible (paramètres : angle, éclairage, identité, expression)
- Embeddings BERT : 768 dimensions, mais la structure sémantique pourrait vivre sur un manifold de dimension ~10-50

**Exemples concrets** :

1. **La sphère $$S^2$$** : surface de dimension 2 plongée dans $$\mathbb{R}^3$$
   - Localement (si vous êtes sur Terre), elle ressemble à un plan ($$\mathbb{R}^2$$)
   - Globalement, c'est courbe et fermée

2. **Le tore** : surface d'un donut, dimension 2 dans $$\mathbb{R}^3$$

3. **Swiss roll** : manifold 2D enroulé dans $$\mathbb{R}^3$$ (exemple classique)

**Propriétés importantes** :

- **Dimension intrinsèque** vs **dimension d'embedding** : le Swiss roll a dimension intrinsèque 2 mais vit dans $$\mathbb{R}^3$$
- **Courbure locale** : mesure à quel point le manifold dévie d'un espace plat
- **Topologie globale** : connexité, trous, composantes (invariants topologiques)

#### 1.2 Géométrie Riemannienne (concepts essentiels)

**Contexte** : 

Vous connaissez la topologie algébrique, qui étudie les propriétés qualitatives des espaces (connexité, trous, composantes). La **géométrie Riemannienne** enrichit cette étude en ajoutant une notion de **distance** et de **courbure** sur les manifolds.

**Définition simplifiée** :

Une **variété Riemannienne** est un manifold équipé d'une **métrique Riemannienne** $$g$$, qui permet de mesurer :
- Les **longueurs** des courbes
- Les **angles** entre vecteurs tangents
- Les **distances géodésiques** (plus courts chemins sur la surface)

**Analogie concrète** :

Imaginez la surface de la Terre :
- **Métrique Euclidienne** (naïve) : distance en ligne droite à travers la Terre (impossible)
- **Métrique Riemannienne** : distance le long de la surface (arc de grand cercle = géodésique)

La distance Paris-New York :
- Euclidienne : ~5 850 km (ligne droite impossible)
- Riemannienne (sur la sphère) : ~5 837 km (arc de cercle sur la surface)

**Composants mathématiques** :

1. **Tenseur métrique** $$g_{ij}$$ : généralise le produit scalaire
   - En $$\mathbb{R}^n$$ plat : $$g_{ij} = \delta_{ij}$$ (matrice identité)
   - Sur un manifold courbe : $$g_{ij}(x)$$ dépend du point $$x$$

2. **Distance géodésique** : 
   $$d_M(x, y) = \inf_{\gamma} \int_0^1 \sqrt{g(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt$$
   où $$\gamma$$ est une courbe reliant $$x$$ à $$y$$ sur le manifold

**Pourquoi c'est important pour UMAP ?**

UMAP suppose que vos données haute dimension vivent sur un **manifold Riemannien**. L'algorithme :
1. Estime la structure métrique locale de ce manifold
2. Construit une représentation topologique
3. Trouve une projection basse dimension qui préserve cette structure

#### 1.3 Complexes simpliciaux et topologie algébrique

**Définition** :

Un **complexe simplicial** est une généralisation d'un graphe qui inclut :
- **0-simplexes** : points (vertices)
- **1-simplexes** : arêtes reliant 2 points
- **2-simplexes** : triangles reliant 3 points
- **k-simplexes** : généralisation à k+1 points

**Pourquoi UMAP utilise cette structure ?**

Un graphe k-NN simple (juste des arêtes) ne capture que les relations pairwise. Les complexes simpliciaux permettent de capturer :
- Les **relations d'ordre supérieur** (triplets, quadruplets)
- La **topologie globale** (trous, composantes, structure homologique)

**Construction dans UMAP** :

1. Pour chaque point $$x_i$$, identifier ses $$k$$ plus proches voisins
2. Créer une arête entre $$x_i$$ et chaque voisin
3. Si 3 points sont mutuellement voisins, créer un 2-simplexe (triangle)
4. Généraliser à des dimensions supérieures

**Fuzzy simplicial sets** :

UMAP généralise en utilisant des **ensembles simpliciaux flous** où :
- Chaque simplexe a une **force d'appartenance** $$\in [0,1]$$ (au lieu de binaire)
- Cela permet de capturer l'**incertitude** sur la structure topologique
- Formellement : c'est une catégorie mathématique avec fuzzy membership functions

#### 1.4 Graphe k-NN pondéré

**Construction** :

Pour chaque point $$x_i$$ :

1. Trouver ses $$k$$ plus proches voisins : $$\{x_{i,1}, x_{i,2}, \ldots, x_{i,k}\}$$

2. Calculer les distances : $$d(x_i, x_{i,j})$$

3. Définir un **rayon local adaptatif** $$\rho_i$$ :
   $$\rho_i = \min_{j \in \{1, \ldots, k\}} d(x_i, x_{i,j})$$
   (distance au plus proche voisin)

4. Calculer les **poids des arêtes** avec une fonction exponentielle :
   $$w(x_i, x_{i,j}) = \exp\left(-\frac{\max(0, d(x_i, x_{i,j}) - \rho_i)}{\sigma_i}\right)$$

**Pourquoi cette formulation ?**

- $$\rho_i$$ assure que chaque point a au moins un voisin avec poids = 1 (connectivité locale garantie)
- $$\sigma_i$$ contrôle la décroissance des poids (analogue à la bande passante)
- Le $$\max(0, \cdot)$$ évite les poids > 1 pour le plus proche voisin

**Adaptation locale** :

Contrairement à un seuil global, $$\sigma_i$$ est **adapté localement** :
- Dans les régions denses : $$\sigma_i$$ petit (décroissance rapide)
- Dans les régions éparses : $$\sigma_i$$ grand (décroissance lente)

Cela normalise la densité et évite le problème où les régions denses dominent.

---

### 2. Les fondations mathématiques d'UMAP

#### 2.1 L'hypothèse du manifold Riemannien

**Hypothèse fondamentale** :

UMAP suppose que les données sont **échantillonnées uniformément** depuis un manifold Riemannien de dimension $$d$$ plongé dans $$\mathbb{R}^D$$ (avec $$d \ll D$$).

**Conséquence mathématique** :

La distance Euclidienne dans $$\mathbb{R}^D$$ est une **approximation** de la distance géodésique sur le manifold, mais seulement **localement** (dans un voisinage infinitésimal).

**Pourquoi "uniformément" ?**

Si l'échantillonnage n'est pas uniforme, les régions denses seront sur-représentées. UMAP compense en :
- Estimant la **densité locale** via $$\rho_i$$ et $$\sigma_i$$
- Normalisant les distances pour que chaque point "voie" environ le même nombre de voisins

#### 2.2 Estimation de la métrique locale

**Objectif** : estimer le tenseur métrique $$g$$ localement autour de chaque point.

**Approche de UMAP** :

Pour chaque point $$x_i$$, UMAP estime la métrique locale en trouvant $$\sigma_i$$ tel que :

$$\sum_{j \in N(i)} \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$

où $$N(i)$$ sont les $$k$$ voisins de $$x_i$$.

**Interprétation** :

- $$\log_2(k)$$ est la **perplexité cible** (comme dans t-SNE)
- Cette équation dit : "ajuste $$\sigma_i$$ pour que le nombre effectif de voisins soit $$k$$"
- C'est résolu par **recherche binaire** (dichotomie)

**Pourquoi cette approche ?**

Elle garantit que :
1. Chaque point contribue équitablement à la structure globale
2. La métrique locale est cohérente avec le voisinage observé
3. Les régions de densités variables sont normalisées

#### 2.3 Construction du complexe simplicial flou

**Étape 1 : Graphe dirigé pondéré**

À partir des poids calculés, on crée un graphe **dirigé** où :
$$w_{ij} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$

**Note** : $$w_{ij} \neq w_{ji}$$ en général (asymétrique)

**Étape 2 : Symétrisation**

On combine les arêtes dirigées en arêtes **non-dirigées** via une **t-conorm** (union probabiliste) :

$$w_{ij}^{\text{sym}} = w_{ij} + w_{ji} - w_{ij} \cdot w_{ji}$$

**Interprétation probabiliste** :

Si on interprète $$w_{ij}$$ comme la probabilité que $$x_i$$ et $$x_j$$ soient connectés :
- $$w_{ij} \cdot w_{ji}$$ = probabilité que les deux directions existent
- $$w_{ij} + w_{ji} - w_{ij} \cdot w_{ji}$$ = probabilité qu'au moins une direction existe (union)

**Étape 3 : Extension aux simplexes supérieurs**

Le graphe pondéré définit les 1-simplexes (arêtes). UMAP étend à :
- **2-simplexes** : si $$x_i, x_j, x_k$$ forment un triangle dans le graphe, avec poids = $$\min(w_{ij}, w_{jk}, w_{ik})$$
- **k-simplexes** : généralisation pour capturer la topologie globale

Cela forme un **fuzzy topological representation** de haute dimension.

#### 2.4 Projection en basse dimension

**Objectif** : trouver une représentation basse dimension $$Y = \{y_1, \ldots, y_n\} \subset \mathbb{R}^d$$ (typiquement $$d=2$$) qui a une structure topologique similaire.

**Construction du graphe basse dimension** :

On construit un graphe similaire dans $$\mathbb{R}^d$$ avec des poids :

$$v_{ij} = \frac{1}{1 + a \cdot \|y_i - y_j\|_2^{2b}}$$

où $$a$$ et $$b$$ sont des paramètres déterminés par `min_dist` (hyperparamètre utilisateur).

**Pourquoi cette forme fonctionnelle ?**

C'est une **distribution t-Student généralisée** (comme dans t-SNE mais paramétrisée différemment) :
- Queue lourde (heavy-tailed) : permet de séparer les clusters distants
- Flexible via $$a, b$$ : contrôle la transition local/global

**Relation avec min_dist** :

`min_dist` est la distance minimale entre points en basse dimension. UMAP résout :

$$\frac{1}{1 + a \cdot (\text{min\_dist})^{2b}} = 0.5$$

pour déterminer $$a$$ et $$b$$ (par optimisation numérique).

---

### 3. Fonction objectif et optimisation

#### 3.1 Cross-Entropy comme fonction de coût

**Objectif** : rendre la représentation basse dimension $$V$$ (poids $$v_{ij}$$) aussi proche que possible de la représentation haute dimension $$W$$ (poids $$w_{ij}^{\text{sym}}$$).

**Fonction de coût** :

UMAP minimise la **cross-entropy** entre les deux distributions :

$$C = \sum_{i,j} w_{ij}^{\text{sym}} \log\left(\frac{w_{ij}^{\text{sym```{v_{ij}}\right) + (1 - w_{ij}^{\text{sym}}) \log\left(\frac{1 - w_{ij}^{\text{sym```{1 - v_{ij}}\right)$$

**Pourquoi cross-entropy plutôt que KL-divergence ?**

- **KL-divergence** (utilisée par t-SNE) : $$\sum_{ij} w_{ij} \log(w_{ij}/v_{ij})$$
  - Pénalise surtout quand $$w_{ij}$$ est grand mais $$v_{ij}$$ est petit
  - Focus sur la structure locale (points proches)

- **Cross-entropy** (UMAP) : inclut aussi le terme $$(1-w_{ij}) \log((1-w_{ij})/(1-v_{ij}))$$
  - Pénalise également quand $$w_{ij}$$ est petit mais $$v_{ij}$$ est grand
  - **Préserve la structure globale** : force les points éloignés à rester éloignés

**C'est la clé de la préservation globale d'UMAP !**

#### 3.2 Optimisation par SGD

**Algorithme** : Descente de gradient stochastique (Stochastic Gradient Descent)

1. **Initialisation** : 
   - Aléatoire : $$y_i \sim \mathcal{N}(0, 0.0001)$$
   - Ou spectral : projection sur les vecteurs propres du Laplacien du graphe

2. **Échantillonnage des arêtes** :
   - Arêtes **positives** : échantillonnées selon $$w_{ij}^{\text{sym}}$$ (préférentiellement les fortes)
   - Arêtes **négatives** : paires aléatoires (force la répulsion des points éloignés)

3. **Mise à jour du gradient** :
   Pour une arête positive $$(i,j)$$ :
   $$\nabla_{y_i} C_{\text{attract}} = -\frac{2ab \cdot w_{ij}^{\text{sym}} \cdot \|y_i - y_j\|_2^{2b-2}}{(1 + a\|y_i - y_j\|_2^{2b})^2} (y_i - y_j)$$

   Pour une arête négative $$(i,k)$$ :
   $$\nabla_{y_i} C_{\text{repel}} = \frac{2b}{(0.001 + \|y_i - y_k\|_2^{2}) (1 + a\|y_i - y_k\|_2^{2b})} (y_i - y_k)$$

4. **Learning rate adaptatif** : décroît au fil des itérations

**Pourquoi SGD et pas gradient complet ?**

- **Scalabilité** : $$O(n)$$ par itération au lieu de $$O(n^2)$$
- **Rapidité** : converge plus vite en pratique
- **Évite les minima locaux** : le bruit stochastique aide à explorer

**Nombre d'époques** :

Typiquement 200-500 époques. UMAP converge plus vite que t-SNE (qui nécessite 1000-5000 itérations).

---

## 💡 Compréhension Intuitive

### Analogie du monde réel : La carte géographique

Imaginez que vous devez créer une **carte 2D de la Terre** (sphère 3D) :

**Approche PCA (projection linéaire)** :
- Aplatir la Terre comme si c'était une crêpe
- Les distances sont totalement déformées
- L'Antarctique et l'Arctique se retrouvent superposés
- Simple mais inexploitable

**Approche t-SNE (structure locale uniquement)** :
- Découper la Terre en petits morceaux
- Préserver les distances dans chaque morceau
- Disposer les morceaux en 2D sans contrainte globale
- **Problème** : Paris et Tokyo peuvent sembler proches si leurs "morceaux" sont placés côte à côte
- La distance entre continents n'a pas de sens

**Approche UMAP (local + global)** :
- Comprendre que la Terre est une sphère (manifold)
- Estimer la métrique locale (courbure)
- Construire une projection qui :
  - Préserve les distances locales (Paris-Lyon correct)
  - Préserve la structure globale (Paris loin de Tokyo)
  - Minimise la distorsion
- Résultat : une carte utilisable (comme les projections Mercator, Robinson, etc.)

**Message clé** : UMAP est comme un cartographe expert qui comprend la géométrie intrinsèque de votre espace de données.

### Questions pour vérifier la compréhension

Avant de continuer, assurez-vous de pouvoir répondre :

1. **Q1** : Quelle est la différence entre la dimension intrinsèque d'un manifold et sa dimension d'embedding ?
   - *Réponse attendue* : Le Swiss roll a dimension intrinsèque 2 (surface 2D) mais vit dans $$\mathbb{R}^3$$ (dimension d'embedding 3)

2. **Q2** : Pourquoi UMAP utilise des poids adaptatifs $$\sigma_i$$ plutôt qu'un seuil global ?
   - *Réponse attendue* : Pour normaliser la densité locale et éviter que les régions denses dominent ; chaque point contribue équitablement

3. **Q3** : Quelle est la différence clé entre la fonction de coût de t-SNE et celle d'UMAP ?
   - *Réponse attendue* : UMAP utilise cross-entropy (inclut le terme $$(1-w)$$) qui pénalise aussi les faux rapprochements, préservant ainsi la structure globale ; t-SNE utilise KL-divergence qui se concentre sur les attractions locales

4. **Q4** : Pourquoi la distribution t-Student en basse dimension permet de mieux séparer les clusters ?
   - *Réponse attendue* : Queues lourdes (heavy-tailed) : décroissance polynomiale lente, donc moins de "crowding" des points distants

---

## 💻 Implémentation Pratique (Minimale)

> **Note** : Cette section est volontairement réduite. L'accent est mis sur la compréhension conceptuelle plutôt que le code.

### 1. Utilisation basique

```python
"""
Utilisation minimale d'UMAP pour visualisation d'embeddings BERT
"""
import umap
import numpy as np
import matplotlib.pyplot as plt

# Données : embeddings BERT de documents
# Shape: (n_documents, 768)  - 768 dimensions pour BERT-base
embeddings = np.load('bert_embeddings.npy')  # Exemple

# Configuration UMAP
reducer = umap.UMAP(
    n_neighbors=15,      # Taille du voisinage local (k dans k-NN)
    min_dist=0.1,        # Distance minimale entre points en 2D
    n_components=2,      # Dimensions de sortie (2 pour visualisation)
    metric='cosine',     # Métrique adaptée aux embeddings
    random_state=42      # Reproductibilité
)

# Projection 2D
embedding_2d = reducer.fit_transform(embeddings)

# Visualisation
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
            s=5, alpha=0.5, cmap='Spectral')
plt.title('Projection UMAP des embeddings BERT')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Cluster (si labels disponibles)')
plt.show()
```

### 2. Explication des hyperparamètres clés

**`n_neighbors`** (défaut : 15)
- **Signification** : contrôle l'équilibre local/global
- **Petit (5-15)** : focus sur structure locale fine, plus de petits clusters
- **Grand (50-200)** : focus sur structure globale, vision "macroscopique"
- **Pour embeddings BERT** : 15-30 généralement approprié

**`min_dist`** (défaut : 0.1)
- **Signification** : distance minimale entre points en basse dimension
- **Petit (0.0-0.1)** : clusters très compacts, séparation forte
- **Grand (0.3-0.99)** : distribution plus uniforme, moins de clumping
- **Pour visualisation** : 0.1-0.3

**`metric`** 
- **Pour embeddings** : `'cosine'` (normalise la magnitude, focus sur direction)
- **Pour données euclidiennes** : `'euclidean'`
- **Autres** : `'manhattan'`, `'hamming'`, ou fonction personnalisée

### 3. Variations importantes

**UMAP supervisé** : incorpore les labels pour améliorer la séparation

```python
# Si vous avez des labels (ex: catégories de documents)
reducer_supervised = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    target_metric='categorical',  # Type des labels
    random_state=42
)

embedding_2d_sup = reducer_supervised.fit_transform(
    embeddings, 
    y=labels  # Labels des documents
)
# Résultat : meilleure séparation des classes
```

**Projection de nouveaux points** (out-of-sample embedding) :

```python
# Entraîner sur un dataset
reducer.fit(embeddings_train)

# Projeter de nouveaux documents
new_embeddings = np.load('new_bert_embeddings.npy')
new_projection = reducer.transform(new_embeddings)

# Avantage : pas besoin de ré-entraîner UMAP entièrement
```

---

## 🔬 Exemples Concrets

### Exemple : Visualisation d'embeddings BERT

**Contexte** :

Vous avez extrait des embeddings BERT pour 10 000 articles scientifiques (768 dimensions). Objectif : visualiser les thématiques et identifier des clusters de sujets similaires.

**Configuration recommandée** :

```python
reducer = umap.UMAP(
    n_neighbors=30,        # Structure globale (thématiques larges)
    min_dist=0.0,          # Clusters compacts
    n_components=2,
    metric='cosine',       # Standard pour embeddings
    n_epochs=500,          # Convergence complète
    random_state=42
)

projection = reducer.fit_transform(bert_embeddings)
```

**Interprétation des résultats** :

- **Clusters identifiables** : chaque groupe = thématique cohérente
  - Ex: machine learning, biologie, physique, etc.
  
- **Distances entre clusters** : contrairement à t-SNE, ces distances ont un sens relatif
  - Clusters proches = sujets connexes
  - Clusters éloignés = sujets distincts

- **Points isolés** : 
  - Articles interdisciplinaires OU
  - Outliers (vérifier la qualité)

- **Ponts entre clusters** : articles transitionnels ou multi-thématiques

**Validation** :

1. Vérifier la cohérence intra-cluster (lire quelques articles du même cluster)
2. Comparer avec labels si disponibles (pureté des clusters)
3. Tester la stabilité : ré-exécuter avec différents `random_state`

---

## ⚖️ Comparaisons et Choix de Design

> **Principe de contiguïté spatiale** : Comparaisons côte à côte pour faciliter la compréhension.

### Comparaison UMAP vs t-SNE vs PCA

| Critère | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Fondement mathématique** | Algèbre linéaire (vecteurs propres) | Divergence KL (probabilités) | Topologie algébrique + Géométrie Riemannienne |
| **Structure préservée** | Globale uniquement | Locale uniquement | Locale **ET** globale |
| **Linéarité** | Linéaire | Non-linéaire | Non-linéaire |
| **Complexité temporelle** | $$O(nd^2 + d^3)$$ | $$O(n^2)$$ ou $$O(n \log n)$$ (Barnes-Hut) | $$O(n^{1.14})$$ à $$O(n)$$ (après k-NN) |
| **Scalabilité** | Excellente (millions de points) | Limitée (~10k points) | Excellente (millions de points) |
| **Vitesse typique (10k points)** | Secondes | Minutes (standard) à ~1 min (Barnes-Hut) | Secondes (~20-60s) |
| **Reproductibilité** | Déterministe | Forte variance (dépend de l'init) | Relativement stable |
| **Out-of-sample embedding** | Oui (projection linéaire) | Non (nécessite ré-entraînement) | Oui (via transform) |
| **Hyperparamètres** | Quasi-automatique | Sensible (perplexity, learning rate) | Robuste (n_neighbors, min_dist) |
| **Préserve distances** | Globales (approximativement) | Non | Locales précisément, globales approximativement |
| **Sépare les clusters** | Faible (peut mélanger) | Excellente | Excellente |
| **Manifolds complexes** | Impossible | Bon (structure locale) | Excellent (topologie complète) |
| **Interprétabilité axes** | Oui (composantes principales) | Non | Non |
| **Cas d'usage principal** | Exploration rapide, preprocessing | Visualisation exploratoire | Visualisation, preprocessing, clustering |

### Quand utiliser chaque méthode ?

**Utiliser PCA quand** :
- ✅ Données approximativement linéaires
- ✅ Besoin de **rapidité maximale**
- ✅ Interprétabilité des axes nécessaire
- ✅ Très grand dataset (>1M points)
- ✅ Preprocessing avant autre algorithme (régression, etc.)
- ✅ Besoin de reproductibilité stricte

**Utiliser t-SNE quand** :
- ✅ **Visualisation exploratoire** pure (pas de réutilisation)
- ✅ Dataset petit/moyen (<10k points)
- ✅ Structure locale très complexe (sous-clusters fins)
- ✅ Pas besoin de projeter de nouveaux points
- ✅ Temps de calcul n'est pas critique

**Utiliser UMAP quand** :
- ✅ Besoin de **préserver structure locale ET globale**
- ✅ Dataset moyen à grand (1k - 10M points)
- ✅ Besoin de **rapidité** (vs t-SNE)
- ✅ Projection de nouveaux points nécessaire (out-of-sample)
- ✅ Preprocessing avant clustering (K-means, DBSCAN)
- ✅ Structures non-linéaires complexes (manifolds)
- ✅ **Visualisation d'embeddings** (BERT, Word2Vec, images)

**Workflow recommandé** :

1. **Exploration initiale** : PCA (rapide, baseline)
2. **Si structure non-linéaire évidente** : 
   - Dataset <10k → t-SNE OU UMAP (tester les deux)
   - Dataset >10k → UMAP uniquement
3. **Pour production/réutilisation** : UMAP (transform possible)

### UMAP vs t-SNE : Différences clés détaillées

#### Différences théoriques

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| **Fondement** | Heuristique basée sur SNE | Théorie topologique rigoureuse |
| **Hypothèse** | Similarités gaussiennes | Manifold Riemannien |
| **Fonction objectif** | KL-divergence | Cross-entropy |
| **Distribution basse dim** | t-Student (1 df) | t-Student généralisée (paramétrable) |
| **Préservation** | Attractions locales | Attractions ET répulsions |

#### Différences pratiques

**Rapidité** :
- t-SNE : ~5-10 min pour 10k points (Barnes-Hut)
- UMAP : ~20-60s pour 10k points
- **UMAP est 5-10× plus rapide**

**Stabilité** :
- t-SNE : forte variance entre exécutions (même random_state)
- UMAP : relativement stable (variance faible)

**Distances inter-clusters** :
- t-SNE : **non significatives** (peuvent être artificiellement compressées)
- UMAP : **approximativement préservées** (ont un sens relatif)

**Exemple concret** :

Imaginez 3 clusters A, B, C :
- En haute dimension : A très proche de B, les deux éloignés de C
- **t-SNE** : peut montrer A, B, C équidistants (compression globale)
- **UMAP** : montrera A proche de B, C éloigné des deux

**Densité des clusters** :
- t-SNE : taille des clusters = artefact (pas de lien avec densité réelle)
- UMAP : taille ≈ densité locale (plus représentatif)

### Recommandation pour vos embeddings BERT

**Pour la visualisation** : **UMAP** est recommandé

**Raisons** :
1. ✅ Rapidité (important pour itération)
2. ✅ Préserve les relations sémantiques globales entre documents
3. ✅ Permet de projeter de nouveaux documents sans ré-entraînement
4. ✅ Métrique cosine naturellement supportée
5. ✅ Structure globale utile pour comprendre les thématiques

**Configuration suggérée** :

```python
umap.UMAP(
    n_neighbors=30,      # Thématiques larges (pas trop local)
    min_dist=0.0,        # Clusters compacts
    metric='cosine',     # Standard pour embeddings
    n_components=2,
    random_state=42
)
```

**Si vous voulez aussi tester t-SNE pour comparaison** :

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,       # Équivalent à n_neighbors
    n_iter=1000,
    metric='cosine',
    random_state=42
)
```

Comparez visuellement : UMAP devrait montrer une meilleure séparation globale des thématiques.

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreurs fréquentes

#### Erreur 1 : Sur-interpréter les distances absolues

**Description** :

Bien qu'UMAP préserve mieux les distances que t-SNE, les distances **absolues** en 2D ne correspondent pas exactement aux distances en haute dimension.

**Exemple problématique** :

```python
# ❌ MAUVAIS : comparer des distances numériques
dist_2d = np.linalg.norm(embedding_2d[0] - embedding_2d[100])
# "Les documents 0 et 100 sont à distance 3.5 en 2D, 
#  donc ils sont modérément similaires" ← FAUX
```

**Pourquoi c'est problématique** :

La projection 2D compresse énormément l'information. Les distances absolues sont déformées.

**Solution** :

```python
# ✅ BON : utiliser les distances originales pour quantification
dist_original = np.linalg.norm(embeddings[0] - embeddings[100])
# Ou cosine similarity pour embeddings
similarity = np.dot(embeddings[0], embeddings[100]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[100])
)

# Utiliser UMAP uniquement pour visualisation qualitative
```

**Impact** : Conclusions erronées si on quantifie à partir de la projection 2D.

#### Erreur 2 : Négliger le choix de la métrique

**Description** :

La métrique de distance a un impact majeur. Par défaut, UMAP utilise `'euclidean'`, qui n'est **pas adaptée** aux embeddings.

**Exemple problématique** :

```python
# ❌ MAUVAIS pour embeddings BERT
reducer = umap.UMAP(metric='euclidean')  # ou par défaut
embedding_2d = reducer.fit_transform(bert_embeddings)
# Les embeddings sont normalisés par direction, pas magnitude
```

**Pourquoi c'est problématique** :

Les embeddings BERT encodent la sémantique dans la **direction**, pas la magnitude. La distance euclidienne est sensible aux deux, ce qui introduit du bruit.

**Solution** :

```python
# ✅ BON : métrique cosine pour embeddings
reducer = umap.UMAP(metric='cosine')
embedding_2d = reducer.fit_transform(bert_embeddings)
# Cosine distance = 1 - cosine similarity
# Focus sur l'angle, ignore la magnitude
```

**Impact** : Clusters mal formés, structure sémantique perdue.

**Source** : [UMAP Documentation - Metrics](https://umap-learn.readthedocs.io/en/latest/parameters.html#metric)

#### Erreur 3 : Comparer visuellement des projections avec paramètres différents

**Description** :

Changer `n_neighbors` ou `min_dist` change radicalement la projection. On ne peut pas comparer visuellement deux projections UMAP avec des paramètres différents.

**Exemple problématique** :

"J'ai testé UMAP avec `n_neighbors=5` et `n_neighbors=50`. La première montre 10 clusters, la seconde 3 clusters. Laquelle est correcte ?"

**Pourquoi c'est problématique** :

Ce ne sont pas des "vérités" différentes, mais des **niveaux de zoom** différents :
- `n_neighbors=5` : structure locale fine (sous-clusters)
- `n_neighbors=50` : structure globale (macro-clusters)

Les deux sont "correctes" à leur échelle.

**Solution** :

1. **Définir l'objectif** d'abord :
   - Identifier des sous-groupes fins → `n_neighbors` petit (5-15)
   - Comprendre la structure globale → `n_neighbors` grand (30-100)

2. **Fixer les paramètres** et interpréter à cette échelle

3. **Ne pas comparer** visuellement des projections avec paramètres différents

**Impact** : Confusion, conclusions contradictoires.

#### Erreur 4 : Ignorer la connectivité du graphe k-NN

**Description** :

Si le graphe k-NN est **déconnecté** (plusieurs composantes), UMAP peut produire des projections artificiellement séparées.

**Détection** :

```python
# Vérifier la connectivité
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

knn_graph = kneighbors_graph(
    embeddings, 
    n_neighbors=15, 
    mode='connectivity'
)
n_components, labels = connected_components(knn_graph)

if n_components > 1:
    print(f"⚠️ Graphe déconnecté : {n_components} composantes")
    # Augmenter n_neighbors ou vérifier les données
```

**Solution** :

- Augmenter `n_neighbors` jusqu'à connectivité
- Ou accepter que les données ont vraiment plusieurs modes disjoints

**Impact** : Fausse séparation de clusters en réalité connectés.

---

### ✅ Bonnes pratiques

#### Pratique 1 : Pipeline PCA → UMAP pour très haute dimension

**Principe** :

Pour des données en très haute dimension (>1000), appliquer d'abord PCA pour réduire à ~50-100 dimensions, puis UMAP.

**Justification scientifique** :

1. **Curse of dimensionality** : en très haute dimension, la notion de "voisinage" devient floue (tous les points sont équidistants)
2. **Bruit** : les dimensions de faible variance sont souvent du bruit
3. **Rapidité** : k-NN en 50D est beaucoup plus rapide qu'en 1000D

**Implémentation** :

```python
from sklearn.decomposition import PCA

# Étape 1 : PCA pour éliminer le bruit et réduire dim
pca = PCA(n_components=50, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)  # 768 → 50

# Étape 2 : UMAP sur l'espace réduit
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean')
embedding_2d = reducer.fit_transform(embeddings_pca)  # 50 → 2
```

**Note pour BERT** : 768 dimensions ne sont pas "très hautes", donc PCA optionnel. Mais pour des embeddings plus grands (GPT-3 : 12288 dim), c'est recommandé.

**Sources** :
- [UMAP FAQ - PCA preprocessing](https://umap-learn.readthedocs.io/en/latest/faq.html)
- Pratique courante en single-cell genomics (30k gènes → 50 PCs → UMAP)

#### Pratique 2 : Validation de la stabilité

**Principe** :

Toujours tester la **stabilité** de la projection en ré-exécutant avec plusieurs `random_state`.

**Justification** :

Même si UMAP est plus stable que t-SNE, l'optimisation SGD peut converger vers des minima locaux différents.

**Implémentation** :

```python
# Tester plusieurs random_state
projections = []
for seed in [42, 123, 456, 789, 1000]:
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed)
    proj = reducer.fit_transform(embeddings)
    projections.append(proj)

# Vérifier visuellement la cohérence
# Les structures principales (clusters) doivent être similaires
# Les détails (positions exactes) peuvent varier légèrement
```

**Critère de stabilité** :

- ✅ Clusters principaux identiques
- ✅ Nombre de clusters stable (±1)
- ⚠️ Positions relatives peuvent varier (rotation, miroir)

**Impact** : Confiance dans les conclusions tirées de la visualisation.

#### Pratique 3 : Initialisation spectrale pour stabilité

**Principe** :

Utiliser une initialisation **spectrale** (basée sur le Laplacien du graphe) plutôt qu'aléatoire.

**Justification** :

L'initialisation spectrale :
- Part d'une projection déjà structurée (vecteurs propres)
- Réduit la variance entre exécutions
- Accélère la convergence

**Implémentation** :

```python
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    init='spectral',     # Au lieu de 'random' (défaut)
    random_state=42
)
```

**Compromis** :

- ✅ Plus stable
- ✅ Converge plus vite
- ⚠️ Légèrement plus lent à l'initialisation (calcul des vecteurs propres)

**Source** : [UMAP Parameters - init](https://umap-learn.readthedocs.io/en/latest/parameters.html#init)

#### Pratique 4 : Utiliser UMAP supervisé pour améliorer la séparation

**Principe** :

Si vous avez des labels (catégories de documents, classes), utilisez **supervised UMAP** pour guider la projection.

**Justification** :

UMAP supervisé modifie la métrique pour :
- Rapprocher les points de même classe
- Éloigner les points de classes différentes
- Préserver la structure intrinsèque tout en optimisant la séparation

**Cas d'usage** :

- Visualisation de datasets labellisés
- Validation de modèles de classification (vérifier la séparabilité)
- Exploration de classes confuses

**Implémentation minimale** :

```python
# Avec labels disponibles
reducer_sup = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    target_metric='categorical',  # Pour labels discrets
    random_state=42
)

embedding_2d_sup = reducer_sup.fit_transform(embeddings, y=labels)
# Meilleure séparation des classes
```

**Important** : à utiliser avec précaution pour ne pas "forcer" une structure qui n'existe pas.

---

### 📋 Checklist de validation

Avant de considérer votre projection UMAP valide :

- [ ] **Métrique appropriée** : `'cosine'` pour embeddings, `'euclidean'` pour données spatiales
- [ ] **Connectivité vérifiée** : graphe k-NN connexe (ou justification si déconnecté)
- [ ] **Stabilité testée** : plusieurs `random_state` donnent des structures similaires
- [ ] **Hyperparamètres justifiés** : choix de `n_neighbors` et `min_dist` basé sur l'objectif
- [ ] **Nombre d'époques suffisant** : convergence atteinte (par défaut généralement OK)
- [ ] **Pas de sur-interprétation** : distances 2D utilisées qualitativement, pas quantitativement
- [ ] **Validation externe** : cohérence avec labels (si disponibles) ou connaissance du domaine

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques Fondamentaux

#### 1. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

- **Auteurs** : Leland McInnes, John Healy, James Melville (2018)
- **Publication** : arXiv preprint
- **URL** : [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
- **Contribution clé** : Introduction d'UMAP avec fondements mathématiques rigoureux en topologie algébrique et géométrie Riemannienne. Démontre la préservation de structure locale ET globale avec complexité $$O(n)$$.
- **Pertinence** : **Lecture essentielle** pour comprendre les fondements théoriques. Sections 2-3 expliquent la théorie, section 4 l'algorithme pratique.
- **Niveau** : Mathématique (topologie algébrique requise), mais section 4 accessible

#### 2. How UMAP Works

- **Auteurs** : Leland McInnes, John Healy, James Melville (2020)
- **Publication** : Blog post technique
- **URL** : [https://umap-learn.readthedocs.io/en/latest/how_umap_works.html](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)
- **Contribution clé** : Explication intuitive et visuelle de l'algorithme UMAP, sans les détails mathématiques lourds
- **Pertinence** : **Excellent point de départ** avant de lire le paper complet. Comprendre l'intuition avant la rigueur.
- **Niveau** : Accessible

#### 3. UMAP for supervised dimension reduction and metric learning

- **Auteurs** : Leland McInnes, John Healy (2020)
- **Publication** : arXiv preprint
- **URL** : [https://arxiv.org/abs/2007.05505](https://arxiv.org/abs/2007.05505)
- **Contribution clé** : Extension d'UMAP pour l'apprentissage supervisé et métrique learning. Utilise les labels pour améliorer la projection.
- **Pertinence** : Si vous travaillez avec des datasets labellisés et voulez optimiser la séparabilité
- **Niveau** : Technique

#### 4. Visualizing Data using t-SNE (pour comparaison)

- **Auteurs** : Laurens van der Maaten, Geoffrey Hinton (2008)
- **Publication** : Journal of Machine Learning Research
- **URL** : [http://jmlr.org/papers/v9/vandermaaten08a.html](http://jmlr.org/papers/v9/vandermaaten08a.html)
- **Contribution clé** : Paper fondateur de t-SNE, référence pour comprendre les différences avec UMAP
- **Pertinence** : **Complémentaire** - lire après UMAP pour apprécier les améliorations
- **Niveau** : Technique (plus accessible que le paper UMAP)

#### 5. Understanding UMAP (Distill.pub article)

- **Auteurs** : Andy Coenen, Adam Pearce (2019)
- **Publication** : Distill.pub (journal interactif)
- **URL** : [https://pair-code.github.io/understanding-umap/](https://pair-code.github.io/understanding-umap/)
- **Contribution clé** : Visualisations interactives montrant l'effet des hyperparamètres (`n_neighbors`, `min_dist`) en temps réel
- **Pertinence** : **Excellente ressource pédagogique** pour l'intuition pratique
- **Niveau** : Accessible, hautement interactif

---

### 📚 Ressources Complémentaires

#### Documentation et tutoriels

- **Documentation officielle UMAP**
  - [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
  - 📌 **Pourquoi** : guide complet des paramètres, exemples, FAQ
  - ⏱️ **Sections clés** : Parameters, Basic Usage, FAQ

- **UMAP vs t-SNE: Superior performance on benchmark datasets**
  - Blog post : [https://towardsdatascience.com/umap-vs-t-sne-c7c4e24f7c74](https://towardsdatascience.com/umap-vs-t-sne-c7c4e24f7c74)
  - 📌 **Pourquoi** : comparaisons empiriques sur datasets réels
  - ⏱️ **Durée** : ~15 min

#### Vidéos éducatives

- **UMAP Uniform Manifold Approximation and Projection for Dimension Reduction** par Leland McInnes (SciPy 2018)
  - [https://www.youtube.com/watch?v=nq6iPZVUxZU](https://www.youtube.com/watch?v=nq6iPZVUxZU)
  - 📌 **Pourquoi** : présentation par l'auteur principal, explication des fondements mathématiques
  - ⏱️ **Durée** : 30 min

- **StatQuest: t-SNE and UMAP, Clearly Explained**
  - [https://www.youtube.com/watch?v=NEaUSP4YerM](https://www.youtube.com/watch?v=NEaUSP4YerM)
  - 📌 **Pourquoi** : comparaison intuitive t-SNE vs UMAP avec visualisations
  - ⏱️ **Durée** : 22 min

---

### 🛠️ Outils et Extensions

#### 1. UMAP (bibliothèque Python officielle)

- **URL** : [https://github.com/lmcinnes/umap](https://github.com/lmcinnes/umap)
- **Description** : Implémentation officielle en Python avec NumPy/Scikit-learn
- **Installation** :

```bash
pip install umap-learn
```

- **Cas d'usage** : usage standard, CPU

#### 2. cuML UMAP (GPU-accelerated)

- **URL** : [https://docs.rapids.ai/api/cuml/stable/api.html#umap](https://docs.rapids.ai/api/cuml/stable/api.html#umap)
- **Description** : Implémentation GPU avec RAPIDS cuML, jusqu'à 50× plus rapide
- **Installation** :

```bash
# Nécessite CUDA
conda install -c rapidsai -c nvidia -c conda-forge cuml
```

- **Cas d'usage** : très grands datasets (>100k points), GPU disponible

#### 3. Parametric UMAP

- **URL** : [https://github.com/lmcinnes/umap/tree/master/umap](https://github.com/lmcinnes/umap/tree/master/umap)
- **Description** : Variante d'UMAP utilisant un réseau de neurones pour apprendre la transformation
- **Avantage** : projection de nouveaux points ultra-rapide (feedforward), intégration dans pipelines deep learning
- **Cas d'usage** : production avec flux continu de nouvelles données

#### 4. UMAP.js (JavaScript)

- **URL** : [https://github.com/PAIR-code/umap-js](https://github.com/PAIR-code/umap-js)
- **Description** : Port JavaScript pour visualisation interactive dans le navigateur
- **Cas d'usage** : dashboards web, exploration interactive

---

### 📖 Cours et Tutoriels Connexes

#### Dans votre repository (Liens Zettelkasten)

- **Complémentaires** :
  - [[tsne]] - t-SNE, algorithme précurseur se concentrant sur la structure locale. UMAP améliore en ajoutant la préservation globale et la rapidité.
  
- **Fondamentaux recommandés** (à créer si nécessaire) :
  - **PCA** - Réduction de dimensionnalité linéaire, baseline pour comparaison
  - **Graphes k-NN** - Construction et propriétés, fondamental pour UMAP
  - **Optimisation SGD** - Descente de gradient stochastique, utilisée dans UMAP

- **Sujets avancés connexes** :
  - **Topologie algébrique appliquée** - Complexes simpliciaux, homologie persistante
  - **Géométrie Riemannienne** - Variétés, métriques, géodésiques
  - **Manifold learning** - Famille d'algorithmes (Isomap, LLE, Laplacian Eigenmaps)

#### Cours externes recommandés

- **Computational Topology and Data Analysis** par Stanford (CS 468)
  - [http://graphics.stanford.edu/courses/cs468-09-fall/](http://graphics.stanford.edu/courses/cs468-09-fall/)
  - 📌 **Modules pertinents** : Simplicial Complexes, Persistent Homology
  - ⏱️ **Durée** : Cours complet ~40h, sections pertinentes ~10h

- **Dimensionality Reduction** par Coursera (Unsupervised Learning)
  - [https://www.coursera.org/learn/unsupervised-learning](https://www.coursera.org/learn/unsupervised-learning)
  - 📌 **Modules pertinents** : Semaines sur PCA, t-SNE, manifold learning
  - ⏱️ **Durée** : 4-6 heures

---

### 🔬 Extensions et variantes d'UMAP

#### Variantes algorithmiques

1. **Parametric UMAP** : utilise un réseau de neurones pour apprendre la transformation
   - Avantage : projection ultra-rapide de nouveaux points
   - Paper : [https://arxiv.org/abs/2009.12981](https://arxiv.org/abs/2009.12981)

2. **Supervised/Semi-supervised UMAP** : incorpore des labels pour améliorer la séparation
   - Cas d'usage : classification, validation de modèles

3. **UMAP for metric learning** : apprend une métrique optimale pour une tâche
   - Cas d'usage : recherche de similarité, ranking

4. **AlignedUMAP** : aligne plusieurs projections UMAP (différents datasets)
   - Cas d'usage : comparaison de conditions expérimentales (single-cell)

#### Applications spécialisées

- **Genomics** : single-cell RNA-seq, visualisation de 30k+ gènes
- **NLP** : visualisation d'embeddings (BERT, GPT, Word2Vec)
- **Computer Vision** : exploration de features CNN, image retrieval
- **Time series** : réduction de séquences temporelles avec métrique DTW

---

## 📝 Résumé Rapide (Quick Reference)

> **Carte de référence** : À consulter rapidement pour se remémorer l'essentiel.

### Concepts Clés

| Concept | Définition/Formule | Cas d'usage |
|---------|-------------------|-------------|
| **Manifold** | Espace courbe localement plat (dimension intrinsèque < embedding) | Hypothèse : données vivent sur manifold bas-dim |
| **Métrique Riemannienne** | Fonction de distance sur manifold courbe : $$d_M(x,y)$$ | Mesurer similarité sur la structure intrinsèque |
| **Graphe k-NN** | $$k$$ plus proches voisins de chaque point | Capture la topologie locale |
| **Poids adaptatifs** | $$w_{ij} = \exp(-(d_{ij}-\rho_i)/\sigma_i)$$ | Normalise densité locale |
| **Cross-entropy** | $$\sum w \log(w/v) + (1-w)\log((1-w)/(1-v))$$ | Préserve local ET global |

### Hyperparamètres Essentiels

| Paramètre | Valeurs typiques | Impact | Recommandation |
|-----------|------------------|--------|----------------|
| **n_neighbors** | 5-100 (défaut: 15) | Local (petit) ↔ Global (grand) | 15-30 pour embeddings |
| **min_dist** | 0.0-0.99 (défaut: 0.1) | Compacité clusters | 0.0-0.1 pour séparation, 0.3-0.5 pour uniformité |
| **metric** | 'euclidean', 'cosine', ... | Type de similarité | **'cosine' pour embeddings** |
| **n_components** | 2-3 pour viz, 10-50 pour ML | Dimensions de sortie | 2 pour visualisation |

### Configuration Minimale pour Embeddings BERT

```python
import umap

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    n_components=2,
    random_state=42
)

projection_2d = reducer.fit_transform(bert_embeddings)
```

### Décisions Clés : UMAP vs t-SNE vs PCA

**Quand utiliser UMAP** :
├─ Dataset > 10k points → UMAP (rapidité)
├─ Besoin de structure globale → UMAP
├─ Projection de nouveaux points nécessaire → UMAP (transform)
└─ Visualisation d'embeddings → **UMAP recommandé**

**Quand utiliser t-SNE** :
├─ Dataset < 5k points → t-SNE ou UMAP (équivalent)
├─ Structure locale très fine → t-SNE légèrement meilleur
└─ Exploration pure sans réutilisation → t-SNE acceptable

**Quand utiliser PCA** :
├─ Exploration rapide initiale → PCA (baseline)
├─ Données linéaires → PCA suffisant
├─ Dataset > 1M points → PCA seul viable
└─ Preprocessing avant régression → PCA

### Pièges à éviter

1. ⚠️ **Distance euclidienne pour embeddings** → Solution : metric='cosine'
2. ⚠️ **Sur-interpréter distances 2D** → Solution : quantifier en haute dimension
3. ⚠️ **Comparer projections avec paramètres différents** → Solution : fixer paramètres avant interprétation
4. ⚠️ **Ignorer la connectivité** → Solution : vérifier composantes connexes du graphe k-NN

### Checklist Rapide

- [ ] Métrique = 'cosine' (si embeddings)
- [ ] n_neighbors ajusté à l'échelle souhaitée (15-30 standard)
- [ ] Stabilité testée (plusieurs random_state)
- [ ] Graphe k-NN connexe
- [ ] Interprétation qualitative (pas quantitative des distances 2D)

---

## 🔗 Intégration Repository GitHub

### Fichiers à mettre à jour

Lors de l'ajout de ce cours, mettre à jour :

1. **`tsne.md`** (cours existant) - Ajouter lien bidirectionnel

Ajouter dans la section "Pour Aller Plus Loin" ou "Cours connexes" :

```markdown
### Alternatives modernes

- **[[umap]]** - UMAP (Uniform Manifold Approximation and Projection)
  - **Avantages vs t-SNE** : plus rapide, préserve structure globale, out-of-sample embedding
  - **Quand l'utiliser** : datasets >10k points, besoin de structure globale, projection de nouveaux points
  - **Fondements** : topologie algébrique et géométrie Riemannienne (vs heuristique probabiliste)
