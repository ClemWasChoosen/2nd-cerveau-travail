# Fonctions d'Activation dans les Réseaux de Neurones

## Table des matières

1. [Introduction fondamentale](#introduction-fondamentale-pourquoi-les-fonctions-dactivation)
2. [Propriétés mathématiques essentielles](#2-propriétés-mathématiques-essentielles)
3. [Fonctions classiques](#3-fonctions-classiques)
4. [Variantes de ReLU](#4-variantes-de-relu)
5. [Fonctions lisses modernes](#5-fonctions-lisses-modernes)
6. [Fonctions spécialisées](#6-fonctions-spécialisées)
7. [Analyse comparative](#7-analyse-comparative)
8. [Fonctions récentes et expérimentales](#8-fonctions-récentes-et-expérimentales)
9. [Guide de décision pratique](#9-guide-de-décision-pratique)
10. [Références et ressources](#10-références-et-ressources)

---

## 1. Introduction Fondamentale : Pourquoi les Fonctions d'Activation ?

### 1.1 Le Problème de la Linéarité

Imaginons un réseau de neurones sans fonction d'activation. Chaque couche effectue simplement une transformation linéaire :

$$z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}$$

où :
- $$z^{(l)}$$ est la sortie de la couche $$l$$
- $\$W^{(l)}$$ est la matrice de poids
- $$a^{(l-1)}$$ est l'activation de la couche précédente
- $$b^{(l)}$$ est le biais

**Problème majeur** : La composition de plusieurs transformations linéaires reste une transformation linéaire.

**Démonstration mathématique** :

Pour un réseau à 2 couches sans activation :
$$z^{(2)} = W^{(2)} \cdot (W^{(1)} \cdot x + b^{(1)}) + b^{(2)}$$
$$z^{(2)} = W^{(2)} \cdot W^{(1)} \cdot x + W^{(2)} \cdot b^{(1)} + b^{(2)}$$
$$z^{(2)} = W_{eq} \cdot x + b_{eq}$$

Avec $\$W_{eq} = W^{(2)} \cdot W^{(1)}$$ et $$b_{eq} = W^{(2)} \cdot b^{(1)} + b^{(2)}$$

**Conséquence** : Un réseau profond de $$n$$ couches linéaires est équivalent à un simple perceptron (une seule couche). La profondeur ne sert à rien !

### 1.2 L'Apport de la Non-Linéarité

Les fonctions d'activation introduisent de la **non-linéarité** en appliquant une transformation non-linéaire après chaque transformation affine :

$$a^{(l)} = f(z^{(l)})$$

où $$f$$ est la fonction d'activation.

**Pourquoi est-ce crucial ?**

1. **Capacité d'approximation universelle** : Le théorème d'approximation universelle stipule qu'un réseau de neurones avec au moins une couche cachée et une fonction d'activation non-linéaire peut approximer n'importe quelle fonction continue sur un compact avec une précision arbitraire.

2. **Apprentissage de patterns complexes** : Les données réelles présentent des relations non-linéaires complexes :
   - En vision : Les contours, textures, formes ne sont pas des combinaisons linéaires de pixels
   - En NLP : Les relations sémantiques entre mots ne sont pas linéaires
   - En général : La plupart des phénomènes naturels sont non-linéaires

3. **Hiérarchie de représentations** : Chaque couche peut apprendre des représentations de plus en plus abstraites :
   - Couche 1 : Détection de contours simples
   - Couche 2 : Combinaison de contours en formes
   - Couche 3 : Objets complets
   - Cette hiérarchie n'est possible qu'avec la non-linéarité

### 1.3 Rôle dans la Rétropropagation

La fonction d'activation joue un rôle crucial dans l'apprentissage via la rétropropagation. Le gradient de la perte par rapport aux poids dépend directement de la dérivée de la fonction d'activation :

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

où $$\frac{\partial a^{(l)}}{\partial z^{(l)}} = f'(z^{(l)})$$ est la dérivée de la fonction d'activation.

**Impact direct** : Les propriétés de $$f'(x)$$ déterminent :
- La vitesse de convergence de l'apprentissage
- La stabilité numérique du gradient
- La capacité du réseau à apprendre des couches profondes

### 1.4 Exemple Concret : [Classification XOR](https://medium.com/@aryanrusia8/the-limitations-of-perceptron-why-it-struggles-with-xor-21905d31f924)

Le problème XOR est le cas classique qui illustre la nécessité de la non-linéarité.

![Image affichage XOR pb](https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ZFy9FXMhynKVJMIXL3Dog.png)

**Données** :
- Points : (0,0) → 0, (0,1) → 1, (1,0) → 1, (1,1) → 0
- Ces points ne sont **pas linéairement séparables**

**Sans fonction d'activation** : Impossible de résoudre le XOR avec un simple perceptron linéaire.

**Avec fonction d'activation non-linéaire** : Un réseau à 2 couches peut facilement résoudre ce problème en créant une frontière de décision non-linéaire.

### 1.5 Critères Essentiels pour une Bonne Fonction d'Activation

Une fonction d'activation efficace doit posséder plusieurs propriétés clés :

1. **Non-linéarité** : Condition sine qua non pour l'approximation de fonctions complexes

2. **Différentiabilité** : Nécessaire pour la rétropropagation (sauf exceptions comme ReLU en 0)

3. **Monotonie** : Facilite la convergence et évite les oscillations

4. **Range approprié** : 
   - Borné : Peut aider à la stabilité numérique
   - Non-borné : Peut éviter les problèmes de saturation

5. **Coût computationnel** : Important pour l'entraînement de réseaux profonds

6. **Comportement du gradient** : 
   - Éviter le gradient vanishing (gradient qui tend vers 0)
   - Éviter le gradient exploding (gradient qui explose)

7. **Propriété de centrage** : Sortie centrée autour de 0 peut accélérer la convergence

### 1.6 Vue d'Ensemble Historique

L'évolution des fonctions d'activation reflète notre compréhension croissante du deep learning :

- **Années 1980-1990** : Sigmoid et Tanh dominent (inspirées par les neurones biologiques)
- **Années 2000-2010** : ReLU révolutionne le deep learning (simplicité + efficacité)
- **Années 2010-2020** : Variantes de ReLU pour résoudre ses limitations
- **Années 2020+** : Fonctions lisses modernes (GELU, Swish) deviennent standard dans les Transformers

Cette évolution montre une tendance claire : du bio-inspiré vers le pragmatisme computationnel.

## 2. Propriétés Mathématiques Essentielles

Cette section analyse les propriétés mathématiques fondamentales qui déterminent le comportement et l'efficacité d'une fonction d'activation. Comprendre ces propriétés est crucial pour choisir la bonne fonction selon le contexte.

### 2.1 Continuité et Différentiabilité

#### 2.1.1 Continuité

Une fonction d'activation est **continue** en $$x_0$$ si :

$$\lim_{x \to x_0} f(x) = f(x_0)$$

**Pourquoi c'est important** : La continuité garantit que de petites variations en entrée produisent de petites variations en sortie, ce qui est essentiel pour la stabilité numérique et la convergence de l'optimisation.

#### 2.1.2 Différentiabilité

Une fonction est **différentiable** en $$x_0$$ si sa dérivée existe en ce point :

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$

**Cas particuliers** :
- ReLU n'est pas différentiable en $$x = 0$$ (mais on utilise une sous-dérivée)
- En pratique, on définit arbitrairement $$f'(0)$$ (généralement 0 ou 1)

**Pourquoi c'est crucial** : La rétropropagation nécessite le calcul de $$\frac{\partial \mathcal{L}}{\partial w} = \delta \cdot x$$ où $$\delta$$ dépend de $$f'(x)$$. Sans dérivée, impossible de calculer le gradient.

#### 2.1.3 Lissité (Smoothness)

Une fonction est **lisse** si elle est infiniment différentiable (classe $\$C^\infty$$). Plus formellement, toutes ses dérivées existent et sont continues.

**Exemples** :
- Sigmoid, Tanh, GELU : lisses ($\$C^\infty$$)
- ReLU : non lisse (discontinuité de la dérivée en 0)
- Leaky ReLU : continue mais pas $\$C^1$$ (dérivée discontinue en 0)

**Impact pratique** :
- Fonctions lisses : Optimisation plus stable, paysage de perte plus "doux"
- Fonctions non lisses : Peuvent converger plus vite mais avec plus d'oscillations

### 2.2 Saturation et Gradient Vanishing

#### 2.2.1 Définition de la Saturation

Une fonction d'activation **sature** dans une région si sa dérivée devient très proche de zéro :

$$|f'(x)| \approx 0 \text{ quand } |x| \to \infty$$

**Régions de saturation** :
- **Saturation gauche** : $$f'(x) \approx 0$$ pour $$x \to -\infty$$
- **Saturation droite** : $$f'(x) \approx 0$$ pour $$x \to +\infty$$

#### 2.2.2 Le Problème du Gradient Vanishing

Lorsque $$f'(x) \approx 0$$, le gradient se propage mal à travers le réseau :

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial z^{(n)}} \cdot \prod_{l=2}^{n} f'(z^{(l)}) \cdot W^{(l)}$$

Si plusieurs $$f'(z^{(l)}) < 1$$, le produit tend exponentiellement vers 0 !

**Exemple concret** : Avec Sigmoid où $$f'(x) \leq 0.25$$

Pour un réseau à 10 couches en saturation :
$$\text{Gradient} \propto (0.25)^{10} = 9.5 \times 10^{-7}$$

Le gradient devient infinitésimal, les premières couches n'apprennent plus.

**Pourquoi cela pose problème** :
- Les couches profondes ne peuvent pas influencer les couches initiales
- L'apprentissage devient extrêmement lent voire impossible
- Les représentations de bas niveau restent figées

#### 2.2.3 Saturation Unilatérale vs Bilatérale

**Saturation bilatérale** (Sigmoid, Tanh) :
- Sature des deux côtés : $$x \to -\infty$$ ET $$x \to +\infty$$
- Plus problématique car plus de chances de saturer

**Saturation unilatérale** (Softplus, ELU) :
- Ne sature que d'un côté
- Moins problématique mais peut toujours causer du gradient vanishing

**Pas de saturation** (ReLU, Leaky ReLU) :
- $$f'(x) = \text{constante} \neq 0$$ pour au moins une demi-droite
- Résout largement le problème du gradient vanishing

### 2.3 Range (Intervalle de Sortie)

#### 2.3.1 Fonctions Bornées

**Définition** : $$\exists M > 0, \forall x \in \mathbb{R}, |f(x)| \leq M$$

**Exemples** : 
- Sigmoid : $$f(x) \in (0, 1)$$
- Tanh : $$f(x) \in (-1, 1)$$

**Avantages** :
- Stabilité numérique : Les activations ne peuvent pas exploser
- Contrôle des valeurs intermédiaires
- Utile pour les probabilités (Sigmoid en couche de sortie)

**Inconvénients** :
- Saturation inévitable
- Gradient vanishing pour les réseaux profonds
- Ralentissement de la convergence

#### 2.3.2 Fonctions Non-Bornées

**Définition** : $$\forall M > 0, \exists x \in \mathbb{R}, |f(x)| > M$$

**Exemples** :
- ReLU : $$f(x) \in [0, +\infty)$$
- Leaky ReLU : $$f(x) \in (-\infty, +\infty)$$

**Avantages** :
- Pas de saturation d'un côté (ou des deux)
- Meilleur flux de gradient
- Convergence plus rapide en pratique

**Inconvénients** :
- Risque d'explosion des activations
- Nécessite souvent une normalisation (Batch Norm, Layer Norm)

### 2.4 Monotonie

#### 2.4.1 Définition

Une fonction est **monotone croissante** si :

$$\forall x_1, x_2 \in \mathbb{R}, x_1 < x_2 \implies f(x_1) \leq f(x_2)$$

**Exemples monotones** : Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish

**Exemples non-monotones** : Swish pour certaines valeurs (bien qu'elle soit quasi-monotone)

#### 2.4.2 Pourquoi la Monotonie est Souhaitée

1. **Préservation de l'ordre** : Si $$x_1 < x_2$$, alors $$f(x_1) \leq f(x_2)$$
   - Facilite l'interprétation
   - Préserve les relations d'ordre dans les données

2. **Simplification du paysage de perte** :
   - Les fonctions monotones créent des surfaces d'optimisation plus simples
   - Moins de minima locaux parasites
   - Convergence plus stable

3. **Gradient toujours de même signe** :
   - Si $$f$$ est croissante, $$f'(x) \geq 0$$ partout
   - Évite les oscillations du gradient

### 2.5 Symétrie et Centrage

#### 2.5.1 Symétrie (Zero-Centered)

Une fonction est **centrée sur zéro** (zero-centered) si :

$$f(0) = 0 \quad \text{et} \quad f(-x) = -f(x)$$ (symétrie impaire)

**Exemples** :
- Tanh : $$\tanh(-x) = -\tanh(x)$$ ✓
- ReLU : $$\text{ReLU}(0) = 0$$ mais pas impaire ✗
- Sigmoid : Non centrée (range [0,1]) ✗

#### 2.5.2 Pourquoi le Centrage est Important

**Problème avec les fonctions non-centrées** (ex: Sigmoid) :

Si $$f(x) > 0$$ toujours, alors les activations sont toujours positives.

Lors de la rétropropagation :
$$\frac{\partial \mathcal{L}}{\partial w_i} = \delta \cdot x_i$$

Si tous les $$x_i > 0$$ (sortie de Sigmoid), alors tous les gradients $$\frac{\partial \mathcal{L}}{\partial w_i}$$ ont le même signe que $$\delta$$.

**Conséquence** : Les poids d'un neurone doivent tous augmenter ou tous diminuer ensemble. L'optimisation devient inefficace et suit un chemin en zigzag.

**Solution avec fonctions centrées** (Tanh, ReLU avec Batch Norm) :
- Les activations peuvent être positives ou négatives
- Les gradients peuvent avoir des signes différents
- Optimisation plus efficace, convergence plus directe

### 2.6 Comportement Asymptotique

#### 2.6.1 Limites aux Infinis

Le comportement pour $$x \to \pm\infty$$ est crucial :

**Type 1 : Saturation complète**
$$\lim_{x \to +\infty} f(x) = c_1, \quad \lim_{x \to -\infty} f(x) = c_2$$
- Exemple : Sigmoid, Tanh
- Gradient vanishing garanti

**Type 2 : Croissance linéaire**
$$\lim_{x \to +\infty} \frac{f(x)}{x} = c \neq 0$$
- Exemple : ReLU ($$c=1$$), Leaky ReLU
- Pas de saturation, meilleur gradient flow

**Type 3 : Croissance sous-linéaire**
$$\lim_{x \to +\infty} f(x) = +\infty, \quad \lim_{x \to +\infty} f'(x) = 0$$
- Exemple : Softplus
- Compromis entre saturation et non-saturation

#### 2.6.2 Comportement près de Zéro

Le comportement autour de $$x = 0$$ détermine comment les petites activations sont traitées :

**Linéaire en 0** : $$f(x) \approx f'(0) \cdot x$$ pour $$x \approx 0$$
- Exemple : Tanh, Sigmoid (en 0), GELU
- Permet aux petits signaux de passer

**Coupure en 0** : $$f(x) = 0$$ pour $$x \leq 0$$
- Exemple : ReLU
- Introduit de la parcimonie (sparsity)

### 2.7 Propriétés Liées au Gradient Flow

#### 2.7.1 Norme du Gradient

Pour une bonne propagation du gradient, on souhaite :

$$\mathbb{E}[|f'(x)|] \approx 1$$

**Pourquoi** : Si $$\mathbb{E}[|f'(x)|] < 1$$ de manière répétée, gradient vanishing.
Si $$\mathbb{E}[|f'(x)|] > 1$$ de manière répétée, gradient exploding.

**Analyse pour différentes fonctions** :

- **Sigmoid** : $$\max(f'(x)) = 0.25$$ → Gradient vanishing
- **Tanh** : $$\max(f'(x)) = 1$$ → Mieux mais peut saturer
- **ReLU** : $$f'(x) \in \{0, 1\}$$ → Bon mais neurones morts possibles
- **SELU** : Conçue pour avoir $$\mathbb{E}[f'(x)] \approx 1$$ (auto-normalisation)

#### 2.7.2 Variance du Gradient

La variance de $$f'(x)$$ impacte la stabilité :

**Faible variance** : Optimisation stable mais possiblement lente
**Forte variance** : Convergence rapide mais instable

**Exemple** :
- ReLU : $$\text{Var}(f'(x))$$ élevée (0 ou 1)
- Tanh : $$\text{Var}(f'(x))$$ plus faible près de 0

### 2.8 Coût Computationnel

#### 2.8.1 Complexité de Calcul

**Forward pass** : Calcul de $$f(x)$$
**Backward pass** : Calcul de $$f'(x)$$

**Classement par coût croissant** :

1. **ReLU** : $$f(x) = \max(0, x)$$ → Comparaison simple
2. **Leaky ReLU** : $$f(x) = \max(\alpha x, x)$$ → Une multiplication en plus
3. **Tanh** : Fonctions exponentielles (plus coûteux)
4. **Sigmoid** : Idem, exponentielle
5. **GELU, Swish** : Nécessitent des approximations ou calculs complexes

**Pourquoi c'est important** :
- Pour des réseaux profonds (millions de neurones), le coût s'accumule
- En production, la latence est critique
- Le rapport performance/coût peut favoriser des fonctions plus simples

#### 2.8.2 Approximations Numériques

Certaines fonctions modernes utilisent des approximations pour réduire le coût :

**GELU approximé** :
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

Cela permet d'éviter le calcul de la fonction de répartition de la loi normale.

### 2.9 Résumé des Propriétés Clés

| Propriété | Idéal pour l'apprentissage | Problèmes si absent |
|-----------|---------------------------|---------------------|
| Non-linéarité | Obligatoire | Pas d'approximation universelle |
| Différentiabilité | Nécessaire (sauf exceptions) | Pas de rétropropagation |
| Pas de saturation | $$f'(x)$$ ne tend pas vers 0 | Gradient vanishing |
| Zero-centered | $$f(-x) = -f(x)$$ | Optimisation en zigzag |
| Monotonie | Croissante | Minima locaux parasites |
| $$\|f'(x)\| \approx 1$$ | Gradient stable | Vanishing ou exploding |
| Coût faible | Opérations simples | Lenteur en production |

Ces propriétés mathématiques sont les outils d'analyse qui permettront de comprendre **pourquoi** chaque fonction d'activation présentée dans les sections suivantes a été développée et **quand** l'utiliser.

## 3. Fonctions d'Activation Classiques

Cette section couvre les trois fonctions fondamentales qui ont marqué l'histoire du deep learning. Bien que certaines soient moins utilisées aujourd'hui pour les couches cachées, leur compréhension est essentielle pour saisir l'évolution du domaine.

### 3.1 Sigmoid (Logistique)

#### 3.1.1 Définition Mathématique

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Forme alternative** :
$$\sigma(x) = \frac{e^x}{e^x + 1}$$

**Range** : $$\sigma(x) \in (0, 1)$$

#### 3.1.2 Dérivée

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

**Démonstration** :
$$\sigma'(x) = \frac{d}{dx}\left(\frac{1}{1 + e^{-x}}\right) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

En remarquant que $$\sigma(x) = \frac{1}{1 + e^{-x}}$$, on a :
$\$1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}$$

Donc :
$$\sigma'(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot (1 - \sigma(x))$$

**Propriété remarquable** : La dérivée s'exprime uniquement en fonction de $$\sigma(x)$$, ce qui est computationnellement efficace en pratique (pas besoin de recalculer l'exponentielle).

#### 3.1.3 Valeurs Clés

- $$\sigma(0) = 0.5$$
- $$\sigma'(0) = 0.25$$ (maximum de la dérivée)
- $$\lim_{x \to +\infty} \sigma(x) = 1$$
- $$\lim_{x \to -\infty} \sigma(x) = 0$$
- $$\lim_{x \to \pm\infty} \sigma'(x) = 0$$

**Observation critique** : La dérivée maximale est seulement 0.25. Cela signifie que même dans le meilleur cas, le gradient est divisé par 4 à chaque couche.

#### 3.1.4 Caractéristiques Graphiques

**Forme** : Courbe en "S" (sigmoïde)
- Croissante
- Symétrique par rapport au point (0, 0.5)
- Saturation rapide pour $$|x| > 4$$

**Comportement** :
- $$x \in [-6, -4]$$ : $$\sigma(x) \approx 0$$ (saturation gauche)
- $$x \in [-2, 2]$$ : Zone linéaire approximative
- $$x \in [4, 6]$$ : $$\sigma(x) \approx 1$$ (saturation droite)

#### 3.1.5 Origine et Motivation Historique

**Pourquoi Sigmoid a été adoptée ?**

1. **Inspiration biologique** : Modélise le taux de décharge d'un neurone biologique
   - Un neurone est "inactif" (0) en dessous d'un seuil
   - Il s'active progressivement
   - Il sature à un maximum (1)

2. **Interprétation probabiliste** : 
   - Output dans $$(0, 1)$$ peut être interprété comme une probabilité
   - Naturel pour les problèmes de classification binaire
   - Connexion avec la régression logistique

3. **Différentiabilité partout** : Contrairement à la fonction échelon (step function) utilisée dans les perceptrons originaux

4. **Propriétés mathématiques élégantes** :
   - Dérivée simple : $$\sigma'(x) = \sigma(x)(1-\sigma(x))$$
   - Propriété de symétrie : $\$1 - \sigma(x) = \sigma(-x)$$

#### 3.1.6 Avantages

1. **Sortie bornée [0, 1]** :
   - Stabilité numérique garantie
   - Parfait pour les couches de sortie en classification binaire

2. **Interprétation probabiliste claire**

3. **Lisse et différentiable partout**

4. **Historiquement bien étudiée** : Beaucoup de théorie développée

#### 3.1.7 Inconvénients Majeurs

**1. Gradient Vanishing Sévère**

Pour un réseau à $\$L$$ couches :
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{l=1}^{L} \sigma'(z^{(l)})$$

Avec $$\sigma'(x) \leq 0.25$$, on a :
$$\prod_{l=1}^{L} \sigma'(z^{(l)}) \leq (0.25)^L$$

**Exemple concret** : Pour $\$L = 10$$ couches :
$$(0.25)^{10} \approx 9.5 \times 10^{-7}$$

Le gradient est pratiquement nul ! Les premières couches n'apprennent pas.

**2. Non Zero-Centered**

La sortie est toujours positive : $$\sigma(x) \in (0, 1)$$

**Conséquence sur l'optimisation** :

Si l'entrée d'un neurone est $$x = [x_1, ..., x_n]$$ avec tous les $$x_i > 0$$ (sortie de sigmoid), alors :

$$\frac{\partial \mathcal{L}}{\partial w_i} = \delta \cdot x_i$$

Tous les $$\frac{\partial \mathcal{L}}{\partial w_i}$$ ont le même signe (celui de $$\delta$$).

**Problème** : Les poids ne peuvent se mettre à jour que dans certaines directions (quadrants), l'optimisation fait des zigzags au lieu d'aller droit au minimum.

**Illustration** : Pour 2 poids $$(w_1, w_2)$$, on ne peut pas faire :
- $$w_1$$ augmente ET $$w_2$$ diminue simultanément

Il faut plusieurs étapes : d'abord les deux augmentent, puis les deux diminuent, etc.

**3. Coût Computationnel**

Calcul de l'exponentielle $$e^{-x}$$ est relativement coûteux comparé à des opérations simples (max, multiplication).

**4. Saturation Facile**

Pour $$|x| > 5$$, on est déjà en saturation forte. Les réseaux profonds tombent rapidement dans ce régime.

#### 3.1.8 Quand Utiliser Sigmoid ?

**Utilisation recommandée** :
- **Couche de sortie** pour classification binaire
- **Gates** dans les LSTM/GRU (pour des valeurs entre 0 et 1)
- Cas où l'interprétation probabiliste est nécessaire

**À éviter** :
- Couches cachées de réseaux profonds (gradient vanishing)
- Lorsque la vitesse d'apprentissage est critique
- Réseaux très profonds (> 5 couches sans normalisation)

### 3.2 Tangente Hyperbolique (Tanh)

#### 3.2.1 Définition Mathématique

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**Relation avec Sigmoid** :
$$\tanh(x) = 2\sigma(2x) - 1$$

où $$\sigma$$ est la fonction sigmoid.

**Range** : $$\tanh(x) \in (-1, 1)$$

#### 3.2.2 Dérivée

$$\tanh'(x) = 1 - \tanh^2(x)$$

**Démonstration** :
$$\tanh'(x) = \frac{d}{dx}\left(\frac{e^x - e^{-x}}{e^x + e^{-x}}\right)$$

En utilisant la règle du quotient :
$$= \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}$$

$$= \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$

$$= 1 - \left(\frac{e^x - e^{-x}}{e^x + e^{-x}}\right)^2 = 1 - \tanh^2(x)$$

**Forme alternative utile** :
$$\tanh'(x) = (1 - \tanh(x))(1 + \tanh(x))$$

#### 3.2.3 Valeurs Clés

- $$\tanh(0) = 0$$ (zero-centered !)
- $$\tanh'(0) = 1$$ (maximum de la dérivée)
- $$\lim_{x \to +\infty} \tanh(x) = 1$$
- $$\lim_{x \to -\infty} \tanh(x) = -1$$
- $$\lim_{x \to \pm\infty} \tanh'(x) = 0$$

**Point crucial** : $$\tanh'(0) = 1$$, ce qui est 4 fois mieux que Sigmoid (0.25).

#### 3.2.4 Propriété de Symétrie

$$\tanh(-x) = -\tanh(x)$$

**Démonstration** :
$$\tanh(-x) = \frac{e^{-x} - e^{x}}{e^{-x} + e^{x}} = -\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = -\tanh(x)$$

C'est une **fonction impaire**, centrée sur l'origine. Cette propriété est fondamentale.

#### 3.2.5 Caractéristiques Graphiques

**Forme** : Courbe en "S" similaire à Sigmoid mais centrée
- Croissante
- Symétrique par rapport à l'origine (0, 0)
- Saturation pour $$|x| > 3$$

**Comportement** :
- $$x \in [-4, -2]$$ : $$\tanh(x) \approx -1$$ (saturation gauche)
- $$x \in [-1, 1]$$ : Zone quasi-linéaire
- $$x \in [2, 4]$$ : $$\tanh(x) \approx 1$$ (saturation droite)

#### 3.2.6 Pourquoi Tanh est Meilleure que Sigmoid ?

**1. Zero-Centered (Centrage sur Zéro)**

La sortie peut être négative ou positive : $$\tanh(x) \in (-1, 1)$$

**Impact sur l'optimisation** :

Pour une couche avec entrée $$x = [x_1, ..., x_n]$$ où certains $$x_i$$ sont négatifs :

$$\frac{\partial \mathcal{L}}{\partial w_i} = \delta \cdot x_i$$

Les gradients peuvent avoir des signes différents → Optimisation plus efficace, convergence plus directe.

**Comparaison visuelle de la convergence** :
- Sigmoid : Trajectoire en zigzag
- Tanh : Trajectoire plus directe vers le minimum

**2. Gradient Maximal Plus Élevé**

$$\max(\tanh'(x)) = 1$$ vs $$\max(\sigma'(x)) = 0.25$$

Pour un réseau à 10 couches au point optimal :
- Sigmoid : gradient $$\propto (0.25)^{10} = 9.5 \times 10^{-7}$$
- Tanh : gradient $$\propto (1)^{10} = 1$$

**Différence massive** : 4 ordres de grandeur !

**3. Convergence Plus Rapide**

Empiriquement, les réseaux avec Tanh convergent 2-3x plus vite que ceux avec Sigmoid.

**Raison** : Combinaison du centrage et du meilleur gradient.

#### 3.2.7 Avantages

1. **Zero-centered** : Résout le problème d'optimisation de Sigmoid

2. **Gradient maximal = 1** : Meilleur flux de gradient que Sigmoid

3. **Sortie bornée [-1, 1]** : Stabilité numérique

4. **Symétrie impaire** : Propriétés mathématiques élégantes

5. **Range plus large** : 2x la plage de Sigmoid

#### 3.2.8 Inconvénients

**Les mêmes problèmes fondamentaux que Sigmoid** :

1. **Gradient Vanishing** : Toujours présent, juste moins sévère
   - Saturation bilatérale pour $$|x| > 3$$
   - Dans les zones saturées : $$\tanh'(x) \approx 0$$

2. **Coût computationnel** : Deux exponentielles à calculer

3. **Saturation** : Même si c'est mieux que Sigmoid, ça reste problématique pour les réseaux très profonds

#### 3.2.9 Analyse Comparative : Sigmoid vs Tanh

| Critère | Sigmoid | Tanh | Gagnant |
|---------|---------|------|---------|
| Range | (0, 1) | (-1, 1) | Tanh |
| Zero-centered | ✗ | ✓ | **Tanh** |
| $$\max(f'(x))$$ | 0.25 | 1 | **Tanh** |
| Gradient vanishing | Sévère | Modéré | **Tanh** |
| Vitesse convergence | Lente | Moyenne | **Tanh** |
| Interprétation probabilité | ✓ | ✗ | Sigmoid |
| Coût computationnel | 1 exp | 2 exp | Sigmoid |

**Conclusion** : Tanh est strictement meilleure que Sigmoid pour les couches cachées.

#### 3.2.10 Quand Utiliser Tanh ?

**Utilisation recommandée** :
- **Couches cachées** de réseaux peu profonds (< 5 couches)
- **RNN/LSTM** : Tanh est standard pour l'état caché
- Lorsque les données d'entrée sont déjà centrées
- Quand on a besoin de sorties positives ET négatives

**À éviter** :
- Réseaux très profonds sans normalisation
- Couche de sortie pour classification binaire (préférer Sigmoid)
- Lorsque la vitesse computationnelle est critique (préférer ReLU)

### 3.3 ReLU (Rectified Linear Unit)

#### 3.3.1 Définition Mathématique

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$$

**Forme alternative** :
$$\text{ReLU}(x) = x \cdot \mathbb{1}_{x > 0}$$

où $$\mathbb{1}_{x > 0}$$ est la fonction indicatrice.

**Range** : $$\text{ReLU}(x) \in [0, +\infty)$$

#### 3.3.2 Dérivée

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x < 0 \\ \text{non définie} & \text{si } x = 0 \end{cases}$$

**En pratique** : On définit arbitrairement $$\text{ReLU}'(0)$$ comme 0 ou 1 (généralement 0).

**Forme condensée** :
$$\text{ReLU}'(x) = \mathbb{1}_{x > 0}$$

**Propriété remarquable** : La dérivée est soit 0, soit 1. Pas de terme intermédiaire !

#### 3.3.3 Caractéristiques Graphiques

**Forme** : Deux demi-droites
- Segment nul pour $$x \leq 0$$
- Droite de pente 1 pour $$x > 0$$
- Point anguleux (non lisse) en $$x = 0$$

**Comportement** :
- $$x < 0$$ : Sortie = 0 (neurone "éteint")
- $$x > 0$$ : Sortie = $$x$$ (neurone "actif", identité)

#### 3.3.4 La Révolution ReLU (2010-2012)

**Contexte historique** : 

Avant 2010, les réseaux profonds étaient quasi impossibles à entraîner :
- Sigmoid/Tanh causaient un gradient vanishing sévère
- AlexNet (2012) a popularisé ReLU
- A permis d'entraîner des réseaux de 8+ couches

**Pourquoi ReLU a tout changé ?**

1. **Pas de saturation pour $$x > 0$$** :
   - $$\text{ReLU}'(x) = 1$$ pour tout $$x > 0$$
   - Le gradient se propage sans atténuation
   - Réseaux profonds deviennent possibles

2. **Simplicité computationnelle extrême** :
   - Juste une comparaison et un max
   - 5-6x plus rapide que Tanh/Sigmoid
   - Pas d'exponentielle coûteuse

3. **Sparsity (Parcimonie)** :
   - Environ 50% des neurones sont à 0
   - Représentations plus efficaces
   - Moins de co-adaptation entre neurones

4. **Similitude avec neurones biologiques** :
   - Les neurones réels ont un seuil
   - Ils ne peuvent pas avoir de taux de décharge négatif

#### 3.3.5 Analyse Mathématique Approfondie

**1. Gradient Flow**

Pour un réseau à $\$L$$ couches, considérons le gradient :

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{l=1}^{L} \text{ReLU}'(z^{(l)}) \cdot W^{(l)}$$

Si les neurones sont actifs ($$z^{(l)} > 0$$) :
$$\prod_{l=1}^{L} \text{ReLU}'(z^{(l)}) = \prod_{l=1}^{L} 1 = 1$$

**Pas d'atténuation du gradient** ! C'est révolutionnaire.

**Comparaison** :
- Sigmoid à 10 couches : gradient $$\times 9.5 \times 10^{-7}$$
- Tanh à 10 couches : gradient $$\times 1$$ (au mieux)
- ReLU à 10 couches : gradient $$\times 1$$

**2. Linéarité par Morceaux**

ReLU est une fonction **linéaire par morceaux** :
- Linéaire pour $$x < 0$$ (pente 0)
- Linéaire pour $$x > 0$$ (pente 1)

**Conséquence** : Un réseau avec ReLU est une fonction linéaire par morceaux du espace d'entrée. Le réseau "découpe" l'espace en régions linéaires.

**Capacité d'approximation** : Un réseau avec $$n$$ neurones ReLU peut créer jusqu'à $\$O(2^n)$$ régions linéaires.

**3. Non-Saturation Unilatérale**

$$\lim_{x \to +\infty} \text{ReLU}(x) = +\infty$$
$$\lim_{x \to +\infty} \text{ReLU}'(x) = 1$$

Pas de saturation à droite → Le gradient ne s'annule jamais pour les neurones actifs.

#### 3.3.6 Avantages Majeurs

1. **Résout le gradient vanishing** :
   - Pour les neurones actifs, gradient constant = 1
   - Permet l'entraînement de réseaux très profonds

2. **Efficacité computationnelle** :
   - Opération triviale : $$\max(0, x)$$
   - Pas d'exponentielle, juste une comparaison
   - Accélération typique : 5-6x vs Tanh

3. **Sparsity naturelle** :
   - Environ 50% des activations sont exactement 0
   - Représentations plus efficaces et interprétables
   - Moins de surapprentissage

4. **Convergence plus rapide** :
   - Empiriquement, convergence 3-5x plus rapide que Sigmoid/Tanh
   - Dû au meilleur gradient et à la simplicité

5. **Scalabilité** :
   - Permet de construire des réseaux de 100+ couches (avec ResNet, normalisation)

#### 3.3.7 Inconvénients et Problèmes

**1. Le Problème des "Dying ReLU" (Neurones Morts)**

**Définition** : Un neurone ReLU est "mort" si son activation est toujours $$\leq 0$$.

**Conséquence** :
$$\text{ReLU}(z) = 0 \quad \forall \text{ exemples}$$
$$\text{ReLU}'(z) = 0 \quad \forall \text{ exemples}$$

$$\frac{\partial \mathcal{L}}{\partial W} = 0$$ → Le neurone ne peut plus apprendre !

**Causes** :

a) **Learning rate trop élevé** :
   - Un gros update peut pousser tous les poids dans la région négative
   - Le neurone reste bloqué à 0

b) **Mauvaise initialisation** :
   - Si $\$W \cdot x + b < 0$$ pour tous les exemples initialement
   - Le neurone ne s'active jamais

c) **Distribution des données** :
   - Si les données sont principalement négatives après la transformation affine

**Exemple concret** :

Supposons un neurone avec poids $\$W$$ et biais $$b$$.
Si après un update : $\$W' \cdot x + b' < 0$$ pour tous les $$x$$ du dataset,
alors $$\text{ReLU}(W' \cdot x + b') = 0$$ toujours.

Le gradient est 0, donc $\$W$$ et $$b$$ ne changent plus. Le neurone est mort définitivement.

**Impact pratique** : On peut perdre 20-40% des neurones dans un réseau mal configuré !

**2. Non Zero-Centered**

$$\text{ReLU}(x) \in [0, +\infty)$$

Toutes les sorties sont positives, même problème que Sigmoid (mais moins grave car compensé par d'autres avantages).

**Atténuation** : Batch Normalization résout largement ce problème en pratique.

**3. Non-Bornée**

$$\text{ReLU}(x)$$ peut tendre vers l'infini.

**Risque** : Explosion des activations si mal régularisé.

**Solution** : Normalisation (Batch Norm, Layer Norm) est quasi-obligatoire.

**4. Non Lisse (Non Différentiable en 0)**

La dérivée a une discontinuité en $$x = 0$$.

**Impact** : Peut causer des problèmes théoriques, mais en pratique c'est rarement un souci.

#### 3.3.8 Solutions au Problème des Dying ReLU

1. **Bonne initialisation** : Xavier/He initialization
   - He init pour ReLU : $\$W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$$

2. **Learning rate adaptatif** : Adam, RMSprop plutôt que SGD pur

3. **Utiliser des variantes** : Leaky ReLU, PReLU, ELU (section suivante)

4. **Batch Normalization** : Maintient les activations dans une plage raisonnable

#### 3.3.9 Quand Utiliser ReLU ?

**Utilisation recommandée (Standard actuel)** :
- **Par défaut** pour les couches cachées de la plupart des architectures
- CNN pour la vision par ordinateur
- MLP classiques
- Réseaux profonds (avec normalisation)
- Lorsque la vitesse est importante

**Précautions** :
- Toujours utiliser avec une bonne initialisation (He init)
- Combiner avec Batch Normalization pour les réseaux profonds
- Surveiller le pourcentage de neurones morts pendant l'entraînement
- Utiliser un learning rate raisonnable (pas trop élevé)

**À éviter** :
- Sans normalisation dans les réseaux très profonds
- RNN/LSTM (Tanh reste meilleur)
- Si beaucoup de neurones meurent (passer à Leaky ReLU ou ELU)

#### 3.3.10 Impact Historique et Adoption

**Avant ReLU (pré-2012)** :
- Réseaux limités à 3-5 couches
- Entraînement lent et difficile
- Pré-entraînement non-supervisé nécessaire

**Après ReLU (post-2012)** :
- Réseaux de 20, 50, 100+ couches possibles
- Convergence rapide
- Entraînement end-to-end direct
- Base de presque toutes les architectures modernes (ResNet, VGG, etc.)

**Citation célèbre** (Krizhevsky et al., AlexNet paper) :
"Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units."

ReLU est probablement l'innovation la plus impactante en deep learning après la backpropagation elle-même.

### 3.4 Comparaison Récapitulative des Fonctions Classiques

| Propriété | Sigmoid | Tanh | ReLU |
|-----------|---------|------|------|
| **Range** | (0, 1) | (-1, 1) | [0, ∞) |
| **Zero-centered** | ✗ | ✓ | ✗ |
| **Gradient max** | 0.25 | 1 | 1 |
| **Gradient vanishing** | Sévère | Modéré | Non (si actif) |
| **Saturation** | Bilatérale | Bilatérale | Unilatérale (gauche) |
| **Lisse** | ✓ | ✓ | ✗ |
| **Coût calcul** | Élevé | Élevé | Très faible |
| **Sparsity** | ✗ | ✗ | ✓ |
| **Neurones morts** | N/A | N/A | ✓ (problème) |
| **Convergence** | Lente | Moyenne | Rapide |
| **Utilisation actuelle** | Sortie binaire | RNN/LSTM | **Standard** |

**Évolution du choix par défaut** :
- Années 1990-2000 : Sigmoid
- Années 2000-2010 : Tanh
- Années 2010+ : ReLU (et variantes)

**Leçon principale** : La simplicité de ReLU (fonction triviale) bat la complexité de Sigmoid/Tanh grâce à de meilleures propriétés de gradient, prouvant qu'en deep learning, les considérations pratiques (gradient flow, coût) dominent l'élégance mathématique.

