# Cours : t-SNE (t-distributed Stochastic Neighbor Embedding)

## 📚 Table des matières

1. Introduction et contexte
2. La problématique de la réduction de dimensionnalité
3. PCA : l'approche linéaire classique
4. SNE : les fondations de t-SNE
5. t-SNE : les améliorations clés
6. Les mathématiques détaillées
7. L'algorithme en pratique
8. Hyperparamètres et tuning
9. Interprétation et pièges à éviter
10. Comparaison PCA vs t-SNE
11. Cas d'usage et applications
12. Ressources et références

---

## 1. Introduction et contexte

### Qu'est-ce que t-SNE ?

**t-SNE** (t-distributed Stochastic Neighbor Embedding) est une technique de **réduction de dimensionnalité non-linéaire** particulièrement efficace pour la **visualisation** de données haute dimension. Elle a été développée par Laurens van der Maaten et Geoffrey Hinton en 2008.

### Pourquoi t-SNE existe ?

Dans le machine learning et le deep learning, on travaille souvent avec des données en haute dimension :
- Images : 28×28 pixels = 784 dimensions (MNIST)
- Embeddings de mots : 300-768 dimensions (Word2Vec, BERT)
- Features CNN : milliers de dimensions

**Le problème** : l'humain ne peut visualiser au maximum que 3 dimensions. Comment représenter ces données complexes de manière compréhensible ?

**L'objectif de t-SNE** : projeter les données haute dimension en 2D ou 3D tout en **préservant la structure locale** des données (les points proches restent proches, les points éloignés restent éloignés).

---

## 2. La problématique de la réduction de dimensionnalité

### Les défis

Quand on réduit des dimensions, on perd forcément de l'information. Les questions clés :

1. **Quelle information préserver ?**
   - Distances globales (structure générale)
   - Distances locales (voisinages)
   - Variances maximales

2. **Linéarité vs Non-linéarité**
   - Approches linéaires (PCA) : rapides mais limitées
   - Approches non-linéaires (t-SNE) : plus expressives mais plus complexes

3. **Le "crowding problem"**
   - En haute dimension, il y a plus d'espace pour espacer les points
   - En 2D/3D, les points risquent de se chevaucher → perte d'information

### Pourquoi la structure locale ?

En visualisation, ce qui nous intéresse souvent :
- **Identifier des clusters** (groupes similaires)
- **Détecter des anomalies** (points isolés)
- **Comprendre les relations de similarité** entre échantillons

La structure locale est donc cruciale pour ces tâches.

---

## 3. PCA : l'approche linéaire classique

### Principe de PCA (Principal Component Analysis)

PCA trouve les directions de **variance maximale** dans les données et projette sur ces axes.

**Mathématiquement** :
1. Centrer les données : $$\mathbf{X}_c = \mathbf{X} - \mathbf{\mu}$$
2. Calculer la matrice de covariance : $$\mathbf{C} = \frac{1}{n}\mathbf{X}_c^T\mathbf{X}_c$$
3. Trouver les vecteurs propres : $$\mathbf{C}\mathbf{v}_i = \lambda_i\mathbf{v}_i$$
4. Projeter sur les k premiers vecteurs propres : $$\mathbf{Y} = \mathbf{X}_c\mathbf{W}_k$$

où $$\mathbf{W}_k$$ contient les k vecteurs propres avec les plus grandes valeurs propres $$\lambda_i$$.

### Avantages de PCA

- ✅ **Rapide** : calcul en $$O(d^3 + nd^2)$$ où d = dimensions, n = échantillons
- ✅ **Déterministe** : même résultat à chaque exécution
- ✅ **Interprétable** : les composantes principales ont un sens
- ✅ **Préserve les distances globales**

### Limitations de PCA

- ❌ **Linéaire uniquement** : ne capture pas les structures non-linéaires
- ❌ **Suppose que variance = information** : pas toujours vrai
- ❌ **Peut mélanger des clusters** bien séparés en haute dimension

### Exemple concret

Imagine des données en forme de "Swiss roll" (rouleau suisse) en 3D :
- PCA va "aplatir" le rouleau mais les points aux extrémités du rouleau seront proches en 2D alors qu'ils sont éloignés sur la surface
- t-SNE va "dérouler" le rouleau et préserver les distances le long de la surface

---

## 4. SNE : les fondations de t-SNE

### L'idée de base

SNE (Stochastic Neighbor Embedding, Hinton & Roweis 2002) utilise une approche probabiliste :

**Principe** : modéliser les similarités entre points comme des **probabilités conditionnelles**.

### Étape 1 : Similarités en haute dimension

Pour chaque paire de points $$\mathbf{x}_i$$ et $$\mathbf{x}_j$$, on définit la probabilité que $$\mathbf{x}_i$$ choisisse $$\mathbf{x}_j$$ comme voisin :

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

**Interprétation** :
- C'est une **distribution gaussienne** centrée sur $$\mathbf{x}_i$$
- Les points proches ont une haute probabilité
- $$\sigma_i$$ contrôle la "largeur" de la distribution (adaptatif par point)

### Étape 2 : Similarités en basse dimension

De même, pour les points projetés $$\mathbf{y}_i$$ et $$\mathbf{y}_j$$ en 2D :

$$q_{j|i} = \frac{\exp(-\|\mathbf{y}_i - \mathbf{y}_j\|^2)}{\sum_{k \neq i} \exp(-\|\mathbf{y}_i - \mathbf{y}_k\|^2)}$$

### Étape 3 : Minimiser la divergence

On cherche à rendre les distributions $$Q$$ et $$P$$ aussi similaires que possible en minimisant la **divergence de Kullback-Leibler** :

$$\text{Cost} = \sum_i KL(P_i \| Q_i) = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

**Pourquoi KL divergence ?**
- Mesure la différence entre deux distributions de probabilité
- Asymétrique : pénalise davantage quand $$p_{j|i}$$ est grande mais $$q_{j|i}$$ est petite
- → Force les points proches en haute dimension à rester proches en basse dimension

### Problèmes de SNE

1. **Asymétrie** : $$p_{j|i} \neq p_{i|j}$$ → calculs plus complexes
2. **Crowding problem** : difficile de représenter des distances modérées en 2D
3. **Optimisation difficile** : nombreux minima locaux

---

## 5. t-SNE : les améliorations clés

### Amélioration 1 : Symétrisation

Au lieu de $$p_{j|i}$$, t-SNE utilise des probabilités jointes **symétriques** :

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Pourquoi ?**
- Simplifie les calculs du gradient
- Garantit que $$p_{ij} = p_{ji}$$
- Plus stable numériquement

### Amélioration 2 : Distribution de Student (le "t" de t-SNE)

**C'est l'innovation majeure !**

En basse dimension, t-SNE utilise une **distribution de Student avec 1 degré de liberté** (distribution de Cauchy) au lieu d'une gaussienne :

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Pourquoi la distribution de Student ?**

C'est LA question clé ! La distribution de Student a des **queues lourdes** ("heavy tails") :

1. **Résout le crowding problem** :
   - En haute dimension : beaucoup d'espace pour espacer les points
   - En basse dimension : peu d'espace → les points se "tassent"
   - Les queues lourdes permettent aux points modérément éloignés d'être représentés plus loin en 2D
   
2. **Analogie visuelle** :
   - Gaussienne : décroît très vite → force tous les points "pas très proches" à être très compressés
   - Student : décroît plus lentement → permet plus d'espace entre clusters

3. **Mathématiquement** :
   - Gaussienne : $$\exp(-x^2)$$ → décroissance exponentielle
   - Student (1 df) : $$(1 + x^2)^{-1}$$ → décroissance polynomiale (plus lente)

### La fonction de coût finale

$$\text{Cost} = KL(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

## 6. Les mathématiques détaillées

### Calcul de $$\sigma_i$$ : la perplexité

Le paramètre $$\sigma_i$$ pour chaque point est déterminé par la **perplexité** souhaitée.

**Perplexité** : mesure du nombre effectif de voisins considérés. Définie comme :

$$\text{Perp}(P_i) = 2^{H(P_i)}$$

où $$H(P_i)$$ est l'entropie de Shannon :

$$H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$

**Algorithme** : Pour chaque point $$i$$, on fait une recherche binaire sur $$\sigma_i$$ pour atteindre la perplexité cible.

**Pourquoi perplexité plutôt que $$\sigma$$ directement ?**
- La perplexité est adaptative : dans les zones denses, $$\sigma_i$$ sera plus petit
- Donne un contrôle sémantique : "combien de voisins prendre en compte"
- Valeurs typiques : 5-50 (souvent 30 par défaut)

### Le gradient

Pour optimiser, on calcule le gradient de la fonction de coût par rapport aux positions $$\mathbf{y}_i$$ :

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4\sum_j (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

**Interprétation physique** :
- Chaque paire de points exerce une "force" sur $$\mathbf{y}_i$$
- Si $$p_{ij} > q_{ij}$$ : force attractive (rapprocher les points)
- Si $$p_{ij} < q_{ij}$$ : force répulsive (éloigner les points)
- Le terme $$(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$ pondère selon la distance actuelle

### Optimisation : gradient descent avec momentum

L'algorithme utilise une descente de gradient avec momentum :

$$\mathbf{Y}^{(t)} = \mathbf{Y}^{(t-1)} + \eta \frac{\partial C}{\partial \mathbf{Y}} + \alpha(t)(\mathbf{Y}^{(t-1)} - \mathbf{Y}^{(t-2)})$$

où :
- $$\eta$$ : learning rate
- $$\alpha(t)$$ : momentum (petit au début, augmente ensuite)

**Pourquoi le momentum ?**
- Accélère la convergence
- Aide à échapper aux minima locaux peu profonds
- Lisse les oscillations

### Astuce : early exaggeration

Au début de l'optimisation (premières 250 itérations), on multiplie les $$p_{ij}$$ par 4 :

$$p_{ij}^{\text{early}} = 4 \cdot p_{ij}$$

**Pourquoi ?**
- Force les clusters à se former rapidement et nettement
- Crée des "espaces vides" entre clusters
- Puis on revient aux vraies valeurs pour affiner

---

## 7. L'algorithme en pratique

### Pseudo-code

Algorithme t-SNE
Entrée:
```
Données X = {x₁, ..., xₙ} en dimension d
Perplexité, nombre d'itérations T

Calcul des similarités haute dimension
Pour chaque i:
    Recherche binaire pour trouver σᵢ donnant la perplexité cible
    Calculer p_{j|i} pour tous j Symétriser: p_{ij} = (p_{j|i} + p_{i|j}) / 2n
Initialisation
    Y ← échantillonnage aléatoire (normal(0, 10⁻⁴))
Pour t = 1 à T:
  a. Calculer q_{ij} pour tous i,j avec distribution Student
  b. Calculer le gradient ∂C/∂Y
  c. Mettre à jour Y avec gradient descent + momentum
  d. Si t < 250: utiliser early exaggeration (p_{ij} ← 4p_{ij})
Retourner Y
```

### Complexité

- **Temps** : $$O(n^2)$$ par itération (paires de points)
- **Mémoire** : $$O(n^2)$$ pour stocker les $$p_{ij}$$

**Pour de grandes données** : utiliser des approximations (Barnes-Hut t-SNE : $$O(n \log n)$$)

---

## 8. Hyperparamètres et tuning

### 1. Perplexité (perplexity)

**Valeur typique** : 5-50, défaut = 30

**Impact** :
- **Petite perplexité (5-15)** : focus sur la structure très locale
  - Clusters plus nombreux et fragmentés
  - Peut révéler des sous-structures fines
  
- **Grande perplexité (30-50)** : structure plus globale
  - Clusters plus larges et cohérents
  - Meilleure vue d'ensemble

**Conseil** : Essayer plusieurs valeurs (ex: 10, 30, 50) et comparer

### 2. Nombre d'itérations

**Valeur typique** : 1000-5000

**Pourquoi important ?**
- t-SNE converge lentement
- Moins de 1000 itérations → souvent incomplet
- Vérifier visuellement que la structure se stabilise

### 3. Learning rate

**Valeur typique** : 10-1000, défaut = 200

**Impact** :
- Trop petit : convergence très lente, risque de rester coincé
- Trop grand : instabilité, points qui "explosent"
- Si tous les points sont compressés en une boule → augmenter le learning rate

### 4. Initialisation

**Options** :
- **Aléatoire** (défaut) : normal(0, 10⁻⁴)
- **PCA** : utiliser les 2 premières composantes principales

**Pourquoi PCA ?**
- Donne un bon point de départ
- Peut accélérer la convergence
- Résultats souvent plus stables

---

## 9. Interprétation et pièges à éviter

### Ce que t-SNE préserve

✅ **Structure locale** : les voisinages sont bien préservés

✅ **Clusters** : les groupes similaires sont regroupés

✅ **Topologie** : la forme générale des manifolds

### Ce que t-SNE ne préserve PAS

❌ **Distances globales** : la distance entre clusters n'a pas de sens absolu

❌ **Densité** : la taille d'un cluster ne reflète pas sa densité en haute dimension

❌ **Axes** : les axes x et y n'ont aucune signification interprétable

### Pièges courants

1. **Sur-interpréter les distances entre clusters**
   - ❌ "Le cluster A est 2x plus éloigné de B que de C"
   - ✅ "Le cluster A est séparé de B et C"

2. **Comparer les tailles de clusters**
   - ❌ "Ce cluster est plus important car plus grand visuellement"
   - ✅ "Vérifier le nombre réel de points dans chaque cluster"

3. **Non-déterminisme**
   - t-SNE est stochastique : 2 exécutions donnent des résultats différents
   - Les positions absolues changent, mais la structure relative reste similaire
   - **Solution** : fixer le random seed pour la reproductibilité

4. **Ne pas exécuter assez d'itérations**
   - Arrêter trop tôt → structure incomplète ou artificielle

### Bonnes pratiques

1. **Toujours essayer plusieurs perplexités** (ex: 10, 30, 50)
2. **Vérifier la convergence** : regarder la courbe de coût
3. **Colorer par labels** (si disponibles) pour valider
4. **Ne pas faire de clustering directement sur t-SNE** : faire le clustering en haute dimension, puis visualiser avec t-SNE
5. **Utiliser en exploration** : t-SNE est un outil de visualisation, pas d'analyse quantitative

---

## 10. Comparaison PCA vs t-SNE

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| **Type** | Linéaire | Non-linéaire |
| **Objectif** | Maximiser la variance | Préserver les voisinages |
| **Distances** | Globales | Locales |
| **Vitesse** | Très rapide ($$O(nd^2)$$) | Lent ($$O(n^2)$$ ou $$O(n \log n)$$) |
| **Déterministe** | Oui | Non (stochastique) |
| **Hyperparamètres** | Peu (nombre de composantes) | Plusieurs (perplexité, itérations, lr) |
| **Interprétabilité** | Axes interprétables | Axes non interprétables |
| **Séparation clusters** | Peut mélanger | Excellente séparation |
| **Structures complexes** | Limitée | Excellente (manifolds, swiss roll) |

### Quand utiliser quoi ?

**Utiliser PCA quand** :
- Données approximativement linéaires
- Besoin de rapidité
- Besoin de reproductibilité stricte
- Analyse quantitative des composantes
- Grande quantité de données (>10k points)

**Utiliser t-SNE quand** :
- Structures non-linéaires complexes
- Visualisation exploratoire
- Détection/visualisation de clusters
- Données haute dimension (embeddings, images)
- Taille modérée (<10k points, sinon Barnes-Hut)

**Workflow combiné** :
1. PCA pour réduire à 50 dimensions (si d >> 50)
2. Puis t-SNE de 50D → 2D
→ Plus rapide et souvent meilleurs résultats

---

## 11. Cas d'usage et applications

### 1. Visualisation d'embeddings

**Word embeddings** (Word2Vec, GloVe, BERT) :
- Visualiser les relations sémantiques entre mots
- Identifier des groupes thématiques
- Exemple : "roi", "reine", "prince" regroupés

**Exemple** :
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# word_vectors: shape (vocab_size, embedding_dim)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(word_vectors)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, word in enumerate(vocabulary):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
