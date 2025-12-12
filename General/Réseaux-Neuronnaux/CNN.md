# Réseaux de Neurones Convolutifs (CNN) - Cours Complet

## 1. Introduction : Pourquoi les CNN ?

### 1.1 Le Problème des Images

**Contexte** : Une image est une matrice de pixels (ex: 224×224×3 = 150,528 valeurs)

**Problème avec les réseaux fully-connected** :
- Première couche dense avec 1000 neurones : **150M de paramètres** rien que pour cette couche !
- **Explosion paramétrique** : Intenable en mémoire, overfit garanti
- **Perte de structure spatiale** : L'image est aplatie en vecteur 1D → perd la notion de voisinage

**Innovation CNN** : Exploiter la **structure spatiale** et la **localité** des images
- **Convolutions** : Opérations locales qui préservent la topologie
- **Partage de poids** : Drastique réduction du nombre de paramètres
- **Hiérarchie de features** : Des patterns simples aux concepts complexes

**Résultat** : État de l'art en vision depuis 2012 (ImageNet)

---

## 2. Fondations : Du Perceptron aux Réseaux

### 2.1 Neurone Artificiel (Perceptron)

**Modèle mathématique** :
$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)$$

Où :
- $$x = (x_1, ..., x_n)$$ : Entrées (features)
- $$w = (w_1, ..., w_n)$$ : Poids (paramètres appris)
- $$b$$ : Biais (bias)
- $$f$$ : Fonction d'activation (non-linéarité)

**Interprétation géométrique** :
- $$w^T x + b = 0$$ définit un **hyperplan**
- Le neurone classifie de quel côté de l'hyperplan se trouve $$x$$

**Pourquoi le biais ?**
- Permet de **translater** l'hyperplan
- Sans biais : l'hyperplan passe obligatoirement par l'origine

### 2.2 Fonctions d'Activation

**Pourquoi nécessaires ?**
Sans activation : $$y = w^T x + b$$ (linéaire)
- Empilement de couches linéaires = fonction linéaire
- $$f(g(x)) = W_2(W_1 x + b_1) + b_2 = Wx + b$$ (toujours linéaire !)
- **Impossible** de modéliser des fonctions non-linéaires complexes

**Activations classiques** :

#### **Sigmoid** :
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Sortie : $$(0, 1)$$ → Interprétable comme probabilité
- **Problèmes** :
  - Saturation : Gradient ≈ 0 pour $$|x|$$ grand (vanishing gradient)
  - Non centrée en 0 : Ralentit convergence
- **Usage** : Couche de sortie (classification binaire)

#### **Tanh** :
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Sortie : $$(-1, 1)$$ → Centrée en 0
- **Avantage** : Gradient plus fort que sigmoid
- **Problème** : Saturation toujours présente

#### **ReLU (Rectified Linear Unit)** :
$$\text{ReLU}(x) = \max(0, x)$$

- **Avantages** :
  - Pas de saturation pour $$x > 0$$
  - Calcul très rapide
  - Induit **sparsité** (certains neurones "morts")
- **Problème** : "Dying ReLU" (neurone bloqué à 0)
- **Usage** : Standard pour couches cachées CNN

#### **Leaky ReLU** :
$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$ avec $$\alpha = 0.01$$ typiquement

- **Avantage** : Évite dying ReLU (gradient non-nul pour $$x < 0$$)

#### **GELU (Gaussian Error Linear Unit)** :
$$\text{GELU}(x) = x \cdot \Phi(x)$$

Où $$\Phi$$ est la fonction de répartition de la loi normale

- **Usage** : Transformers, modèles modernes
- Plus lisse que ReLU, performances empiriques supérieures

### 2.3 Multi-Layer Perceptron (MLP)

**Architecture** :

Input (x) → Hidden Layer 1 → Hidden Layer 2 → ... → Output (y)

**Mathématiques** (pour 1 couche cachée) :
$$h = f_1(W_1 x + b_1)$$
$$y = f_2(W_2 h + b_2)$$

**Théorème d'approximation universelle** :
Un MLP avec **une seule couche cachée** (avec suffisamment de neurones) peut approximer **n'importe quelle fonction continue**.

**Pourquoi plusieurs couches alors ?**
- Approximer avec **moins de neurones** (plus efficace)
- Représentations **hiérarchiques** : Couches profondes apprennent des concepts de plus en plus abstraits

---

## 3. Convolution : L'Opération Fondamentale

### 3.1 Définition Mathématique

**Convolution discrète 2D** :
$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)$$

Où :
- $\$I$$ : Image d'entrée (input)
- $\$K$$ : Noyau de convolution (kernel/filtre)
- $$(i, j)$$ : Position dans l'image de sortie

**En pratique** (cross-correlation, utilisée dans les CNN) :
$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$

**Note** : La différence (convolution vs cross-correlation) est négligeable car les poids du kernel sont **appris** → Le réseau apprend le kernel optimal quelle que soit la convention

### 3.2 Exemple Concret

**Image 5×5** :

```
1  2  3  4  5
6  7  8  9  10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

**Kernel 3×3** (détection de bords verticaux) :

```
-1  0  1
-1  0  1
-1  0  1
```

**Calcul pour position (1,1)** (centre du kernel sur pixel 7) :
$$(I * K)(1,1) = 1(-1) + 2(0) + 3(1) + 6(-1) + 7(0) + 8(1) + 11(-1) + 12(0) + 13(1)$$
$$= -1 + 3 - 6 + 8 - 11 + 13 = 6$$

**Interprétation** :
- Valeur positive → Transition clair→foncé (vers la droite)
- Valeur négative → Transition foncé→clair
- Proche de 0 → Pas de bord vertical

### 3.3 Pourquoi les Convolutions Fonctionnent ?

#### **Localité spatiale**
Les pixels voisins sont fortement corrélés
- Un bord, une texture se manifeste **localement**
- Pas besoin de connecter chaque pixel à tous les autres

#### **Partage de poids**
Le même kernel est appliqué à **toute l'image**
- **Invariance par translation** : Un chat détecté en haut-gauche = détecté partout
- **Réduction drastique** : Kernel 3×3 = 9 paramètres (vs millions pour fully-connected)

#### **Hiérarchie de features**
- **Couche 1** : Détecte bords, coins, textures simples
- **Couche 2** : Combine les bords → Formes (cercles, carrés)
- **Couche 3** : Combine formes → Parties d'objets (roue, œil)
- **Couche N** : Objets complets (voiture, visage)

### 3.4 Paramètres de Convolution

#### **Stride (s)**
Pas de déplacement du kernel

- $$s=1$$ : Slide de 1 pixel (standard)
- $$s=2$$ : Skip 1 pixel → Downsampling

**Impact sur dimension** :
$$\text{Output size} = \left\lfloor \frac{n - k}{s} \right\rfloor + 1$$

Où $$n$$ = taille input, $$k$$ = taille kernel

#### **Padding (p)**
Ajout de pixels sur les bords (généralement zéros)

**Pourquoi ?**
- **Préserver dimensions** : Sans padding, l'image rétrécit à chaque couche
- **Information de bord** : Les pixels de bord sont moins utilisés → Padding équilibre

**Types** :
- **Valid** : Pas de padding ($$p=0$$)
- **Same** : Padding tel que $$\text{output size} = \text{input size}$$ (si $$s=1$$)

**Formule avec padding** :
$$\text{Output size} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

**Exemple** : Input 32×32, kernel 5×5, stride 1
- Sans padding : $$\lfloor \frac{32-5}{1} \rfloor + 1 = 28$$
- Avec padding=2 : $$\lfloor \frac{32+4-5}{1} \rfloor + 1 = 32$$ ✓

#### **Dilation (d)**
Espacement entre les éléments du kernel

- $$d=1$$ : Kernel standard
- $$d=2$$ : Un pixel d'écart entre chaque élément

**Pourquoi ?**
- Augmente le **champ récepteur** sans augmenter le nombre de paramètres
- Capte des patterns à plus large échelle

**Effective kernel size** : $$k_{\text{eff}} = k + (k-1)(d-1)$$

---

## 4. Couche de Convolution (Conv Layer)

### 4.1 Architecture Complète

**Input** : Tensor de dimension $$(H, W, C_{\text{in}})$$
- $\$H$$ : Hauteur
- $\$W$$ : Largeur
- $\$C_{\text{in}}$$ : Nombre de canaux (3 pour RGB)

**Paramètres** :
- $\$F$$ : Nombre de filtres (feature maps en sortie)
- $\$K$$ : Taille du kernel (typiquement 3×3 ou 5×5)

**Poids** : Tensor $$(K, K, C_{\text{in}}, F)$$
- Chaque filtre est de dimension $$(K, K, C_{\text{in}})$$
- Convolve sur **tous les canaux** simultanément

**Output** : Tensor $$(H', W', F)$$

### 4.2 Nombre de Paramètres

**Par filtre** :
$$\text{Params} = K \times K \times C_{\text{in}} + 1$$

Le "+1" est le biais

**Total pour la couche** :
$$\text{Params}_{\text{total}} = F \times (K \times K \times C_{\text{in}} + 1)$$

**Exemple** : Input 224×224×3, 64 filtres 3×3
- Params = $\$64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1,792$$
- **Comparaison** : Fully-connected avec 64 neurones → $\$224 \times 224 \times 3 \times 64 = 9.6M$$ paramètres !

### 4.3 Champ Récepteur (Receptive Field)

**Définition** : Région de l'image d'entrée qui influence un neurone donné

**Calcul** (pour couches empilées) :
$$RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i$$

**Exemple** : 3 couches avec kernel 3×3, stride 1
- Couche 1 : $\$RF = 3$$
- Couche 2 : $\$RF = 3 + (3-1) \times 1 = 5$$
- Couche 3 : $\$RF = 5 + (3-1) \times 1 = 7$$

**Pourquoi important ?**
- Couches profondes capturent du **contexte global**
- Trade-off : Grands kernels (RF large) vs Profondeur (plus de non-linéarités)

---

## 5. Pooling : Réduction de Dimension

### 5.1 Principe

**Objectif** : Downsampling pour réduire dimension spatiale

**Avantages** :
- **Réduction calcul** : Moins de paramètres dans les couches suivantes
- **Invariance** : Petites translations de l'input ne changent pas l'output
- **Champ récepteur** : Augmente plus rapidement

### 5.2 Types de Pooling

#### **Max Pooling**
Prend la **valeur maximale** dans une fenêtre

**Exemple** : Max pooling 2×2, stride 2 sur :

```
Input 4×4:          Output 2×2:
1  3  2  4          3  4
2  1  4  3    →     5  6
5  4  1  2
3  2  6  1
```

**Intuition** : Garde la **feature la plus forte** (ex: présence d'un bord)

#### **Average Pooling**
Prend la **moyenne** dans une fenêtre

**Usage** : Moins courant dans les couches intermédiaires, utilisé en sortie (Global Average Pooling)

#### **Global Average Pooling (GAP)**
Moyenne sur **toute la feature map** ($\$H \times W → 1 \times 1$$)

**Avantages** :
- **Régularisation** : Pas de paramètres → Réduit overfit
- Remplace souvent les dernières couches fully-connected

### 5.3 Pooling vs Strided Convolutions

**Débat moderne** : Remplacer pooling par convolutions avec stride > 1 ?

**Arguments pour strided conv** :
- Pooling = opération fixe (pas de paramètres appris)
- Strided conv apprend le downsampling optimal

**Arguments pour pooling** :
- Moins de paramètres
- Invariance géométrique explicite

**Pratique actuelle** : Mix des deux (pooling toujours très utilisé)

---

## 6. Batch Normalization

### 6.1 Le Problème : Internal Covariate Shift

**Observation** : Les distributions d'activations changent au cours de l'entraînement
- Couche $$l$$ reçoit des inputs dont la distribution varie
- Ralentit convergence, nécessite learning rates faibles

### 6.2 Principe de Batch Normalization

**Normalisation par mini-batch** :

Pour un mini-batch $$\mathcal{B} = \{x_1, ..., x_m\}$$ :

1. **Moyenne** :
$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

2. **Variance** :
$$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

3. **Normalisation** :
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

4. **Scale & Shift** (paramètres appris) :
$$y_i = \gamma \hat{x}_i + \beta$$

### 6.3 Pourquoi Scale & Shift ?

**Problème** : La normalisation pourrait supprimer la capacité d'apprentissage
- Ex: Sigmoid fonctionne bien dans ses zones non-linéaires (pas autour de 0)

**Solution** : Paramètres $$\gamma, \beta$$ appris
- Le réseau peut **apprendre à annuler** la normalisation si nécessaire
- $$\gamma = \sigma_{\mathcal{B}}, \beta = \mu_{\mathcal{B}} \Rightarrow$$ identité

### 6.4 Avantages

✅ **Convergence plus rapide** : Learning rates plus élevés possibles  
✅ **Régularisation** : Léger effet de bruit (variance du mini-batch)  
✅ **Moins sensible** à l'initialisation des poids  

**Placement** : Généralement **après convolution, avant activation**

Conv → BatchNorm → ReLU

### 6.5 Inférence vs Entraînement

**Problème** : En inférence, pas de mini-batch (une seule image)

**Solution** : Utiliser les **statistiques globales** (moving average durant l'entraînement)
$$\mu_{\text{global}} = \mathbb{E}[\mu_{\mathcal{B}}]$$
$$\sigma_{\text{global}}^2 = \mathbb{E}[\sigma_{\mathcal{B}}^2]$$

---

## 7. Architectures Classiques

### 7.1 LeNet-5 (1998) - Yann LeCun

**Première CNN moderne** : Reconnaissance de chiffres manuscrits (MNIST)

**Architecture** :

```
Input (32×32×1)
→ Conv1 (6 filtres 5×5) → AvgPool (2×2)
→ Conv2 (16 filtres 5×5) → AvgPool (2×2)
→ FC1 (120 neurones)
→ FC2 (84 neurones)
→ Output (10 classes)
```

**Innovation** : Prouve l'efficacité des convolutions pour la vision

### 7.2 AlexNet (2012) - Krizhevsky, Sutskever, Hinton

**Révolution** : Gagne ImageNet 2012 avec 16.4% top-5 error (vs 26% précédent)

**Architecture** :

```
Input (224×224×3)
→ Conv1 (96 filtres 11×11, stride 4) → MaxPool
→ Conv2 (256 filtres 5×5) → MaxPool
→ Conv3 (384 filtres 3×3)
→ Conv4 (384 filtres 3×3)
→ Conv5 (256 filtres 3×3) → MaxPool
→ FC1 (4096) → Dropout
→ FC2 (4096) → Dropout
→ Output (1000 classes)
```

**Innovations** :
- **ReLU** : Première utilisation massive (6x plus rapide que tanh)
- **Dropout** : Régularisation dans les FC layers
- **Data augmentation** : Crops, flips, color jittering
- **GPU** : Entraînement sur 2 GTX 580 (parallelisation)

**Impact** : Déclenche la révolution deep learning en vision

### 7.3 VGGNet (2014) - Simonyan & Zisserman

**Philosophie** : Simplicité et profondeur
- **Uniquement** des convolutions 3×3
- Empilement profond (16 ou 19 couches)

**Architecture VGG-16** :

```
Input (224×224×3)
→ [Conv 3×3 × 2] (64 filtres) → MaxPool
→ [Conv 3×3 × 2] (128 filtres) → MaxPool
→ [Conv 3×3 × 3] (256 filtres) → MaxPool
→ [Conv 3×3 × 3] (512 filtres) → MaxPool
→ [Conv 3×3 × 3] (512 filtres) → MaxPool
→ FC (4096) → FC (4096) → Output (1000)
```

**Pourquoi 3×3 ?**
- **Champ récepteur** : Deux 3×3 = un 5×5 (mais moins de paramètres)
  - $\$2 \times (3 \times 3 \times C \times C) = 18C^2$$
  - vs $\$5 \times 5 \times C \times C = 25C^2$$
- **Plus de non-linéarités** : 2 ReLU au lieu d'1

**Limitations** : 138M paramètres (lourd en mémoire)

### 7.4 GoogLeNet / Inception (2014)

**Innovation** : **Inception Module** - convolutions multi-échelles parallèles

**Module Inception** :
```
Input
├→ Conv 1×1 → Output1
├→ Conv 1×1 → Conv 3×3 → Output2
├→ Conv 1×1 → Conv 5×5 → Output3
└→ MaxPool 3×3 → Conv 1×1 → Output4
→ Concatenate [Output1, Output2, Output3, Output4]
```

**Intuitions** :
- **Multi-échelle** : Capture patterns de différentes tailles simultanément
- **Conv 1×1** : "Bottleneck" pour réduire dimensionnalité (compression de canaux)

**Avantages** :
- Plus efficace que VGG (6.8M params vs 138M)
- Meilleures performances

### 7.5 ResNet (2015) - He et al. **[RÉVOLUTION]**

**Problème** : Réseaux très profonds (> 20 couches) dégradent les performances
- Pas de l'overfit (erreur train aussi augmente !)
- **Degradation problem** : Difficile d'apprendre la fonction identité

**Innovation** : **Residual Connections (Skip Connections)**

**Bloc Résiduel** :
$$\mathcal{F}(x) = H(x) - x$$

Le réseau apprend le **résidu** $$\mathcal{F}(x)$$ au lieu de $\$H(x)$$

**Mathématiquement** :
$$y = \mathcal{F}(x, \{W_i\}) + x$$

Où $$\mathcal{F}$$ représente 2-3 couches conv

**Architecture (bloc basique)** :
```
Input (x)
├→ Conv 3×3 → BN → ReLU
└→ Conv 3×3 → BN
↓
(+) x  [skip connection]
↓
ReLU
```

**Pourquoi ça marche ?**

1. **Gradient flow** : Le gradient peut passer directement par les skip connections
   $$\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + 1$$
   → Le "+1" assure un gradient non-nul

2. **Apprentissage facilité** :
   - Si la fonction optimale est proche de l'identité → $$\mathcal{F}(x) \approx 0$$ (facile à apprendre)
   - Le réseau peut ajouter des "raffinements" progressifs

**Variantes** :
- **ResNet-50/101/152** : 50, 101, 152 couches
- **ResNet-50** devient le standard (compromis performance/coût)

**Impact** : Permet d'entraîner des réseaux de **1000+ couches**

### 7.6 Architectures Modernes

#### **DenseNet (2017)**
Connecte **chaque couche à toutes les suivantes**
- Réutilisation maximale des features
- Moins de paramètres que ResNet

#### **EfficientNet (2019)**
Optimise simultanément **profondeur, largeur et résolution**
- Compound scaling method
- État de l'art en efficacité (performance/params)

#### **Vision Transformers (ViT, 2020)**
Applique l'architecture Transformer aux images
- Divise l'image en patches
- Plus de convolutions !
- Performances supérieures avec beaucoup de données

---

## 8. Entraînement : Backpropagation & Optimisation

### 8.1 Fonction de Coût (Loss)

#### **Classification Multi-Classes : Cross-Entropy**

Pour une image avec vraie classe $$y$$ :
$$\mathcal{L} = -\log(p_y)$$

Où $$p_y$$ est la probabilité prédite pour la classe $$y$$

**Avec softmax** :
$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(p_i)$$

Où $$y_i$$ est l'encodage one-hot de la vraie classe

**Pourquoi cross-entropy ?**
- Pénalise fortement les **mauvaises prédictions confiantes**
- $$p_y \to 0 \Rightarrow \mathcal{L} \to \infty$$
- Gradients bien conditionnés

#### **Classification Binaire : Binary Cross-Entropy**
$$\mathcal{L} = -[y \log(p) + (1-y) \log(1-p)]$$

### 8.2 Backpropagation

**Principe** : Calculer $$\frac{\partial \mathcal{L}}{\partial w}$$ pour chaque poids $$w$$

**Chain rule** :
$$\frac{\partial \mathcal{L}}{\partial w_l} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z_L} \cdot ... \cdot \frac{\partial z_l}{\partial w_l}$$

**Algorithme** :
1. **Forward pass** : Calculer toutes les activations
2. **Compute loss** : $$\mathcal{L}$$
3. **Backward pass** : Propager les gradients de la sortie vers l'entrée
4. **Update weights** : $$w := w - \eta \frac{\partial \mathcal{L}}{\partial w}$$

### 8.3 Gradient de la Convolution

**Forward** :
$$y_{i,j} = \sum_m \sum_n x_{i+m, j+n} \cdot w_{m,n}$$

**Backward** (gradient par rapport à $$w$$) :
$$\frac{\partial \mathcal{L}}{\partial w_{m,n}} = \sum_i \sum_j x_{i+m, j+n} \cdot \frac{\partial \mathcal{L}}{\partial y_{i,j}}$$

**Interprétation** : Le gradient sur un poids du kernel est une **convolution** entre l'input et le gradient de l'output

### 8.4 Optimisateurs

#### **SGD (Stochastic Gradient Descent)**
$$w_{t+1} = w_t - \eta \nabla \mathcal{L}(w_t)$$

- Simple, mais convergence lente
- Sensible au learning rate $$\eta$$

#### **SGD avec Momentum**
$$v_{t+1} = \beta v_t + \nabla \mathcal{L}(w_t)$$
$$w_{t+1} = w_t - \eta v_{t+1}$$

- **Intuition** : "Boule qui roule" accumule de la vitesse
- Franchit les plateaux, réduit oscillations

#### **Adam (Adaptive Moment Estimation)**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}$$  (moment d'ordre 1)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L})^2$$  (moment d'ordre 2)

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$, $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$ (bias correction)

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Avantages** :
- Learning rate adaptatif par paramètre
- Robuste, fonctionne bien "out of the box"
- **Standard actuel**

**Hyperparamètres typiques** : $$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$$

### 8.5 Learning Rate Scheduling

**Problème** : Learning rate fixe → Convergence sous-optimale

**Stratégies** :

#### **Step Decay**
Diviser $$\eta$$ par 10 tous les N epochs

#### **Cosine Annealing**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$$

- Décroissance douce
- Populaire pour training long

#### **Warmup**
Augmenter linéairement $$\eta$$ durant les premiers epochs
- **Pourquoi ?** BatchNorm n'est pas stable au début
- Évite divergence

---

## 9. Régularisation : Éviter l'Overfitting

### 9.1 Dropout

**Principe** : Désactiver aléatoirement des neurones durant l'entraînement

**Mathématiquement** :
$$y = \text{Dropout}(x, p) = \begin{cases} 
\frac{x}{1-p} & \text{avec probabilité } 1-p \\
0 & \text{avec probabilité } p
\end{cases}$$

**Intuition** :
- Force le réseau à ne pas **dépendre d'un neurone spécifique**
- Équivalent à entraîner un **ensemble de sous-réseaux**

**Paramètre** : $$p=0.5$$ typique (50% de neurones désactivés)

**En inférence** : Tous les neurones actifs (scaling automatique avec $$\frac{1}{1-p}$$)

### 9.2 Data Augmentation

**Transformations géométriques** :
- Random crops
- Horizontal flips
- Rotations (±15°)
- Scaling

**Transformations photométriques** :
- Brightness/contrast jittering
- Saturation
- Cutout (masquage aléatoire de zones)

**Mixup** :
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Où $$\lambda \sim \text{Beta}(\alpha, \alpha)$$

**CutMix** : Coller un patch d'une image sur une autre

### 9.3 Weight Decay (L2 Regularization)

**Ajout à la loss** :
$$\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2} \sum_i w_i^2$$

**Effet** : Pénalise les poids trop grands → Force le réseau à **distribuer** l'information

**Équivalent dans l'optimisation** :
$$w_{t+1} = w_t - \eta(\nabla \mathcal{L} + \lambda w_t) = (1 - \eta\lambda)w_t - \eta \nabla \mathcal{L}$$

→ Décroissance exponentielle des poids non utilisés

### 9.4 Early Stopping

**Principe** : Surveiller performance sur validation set
- Arrêter quand la validation error commence à augmenter

**Intuition** : Trouver le sweet spot avant overfit

---

## 10. Transfer Learning

### 10.1 Principe

**Idée** : Réutiliser un modèle pré-entraîné sur une large tâche (ImageNet)

**Justification** :
- Les **premières couches** apprennent des features génériques (bords, textures)
- Transférables à d'autres tâches visuelles

### 10.2 Stratégies

#### **Feature Extraction (Freeze)**
1. Charger modèle pré-entraîné
2. **Freeze** toutes les couches (pas de gradient)
3. Remplacer la dernière couche (classification head)
4. Entraîner uniquement la nouvelle couche

**Quand ?** Dataset petit (< 10k images), proche d'ImageNet

#### **Fine-Tuning**
1. Charger modèle pré-entraîné
2. **Unfreeze** certaines couches (généralement les dernières)
3. Entraîner avec learning rate **faible**

**Quand ?** Dataset moyen (10k-100k images)

#### **Full Training**
Entraîner toutes les couches

**Quand ?** Dataset large (> 100k images), tâche très différente d'ImageNet

### 10.3 Règles Empiriques

| Dataset size | Similarité à ImageNet | Stratégie |
|--------------|----------------------|-----------|
| Petit | Haute | Feature extraction |
| Petit | Faible | Fine-tuning (risque overfit) |
| Moyen | Haute | Fine-tuning |
| Moyen | Faible | Fine-tuning ou Full |
| Large | Quelconque | Full training |

---

## 11. Interprétabilité : Comprendre les CNN

### 11.1 Pourquoi Interpréter ?

CNN = **Boîte noire** → Comprendre ce qui est appris est crucial pour :
- Debugging (le modèle triche ?)
- Confiance (médical, sécurité)
- Amélioration (identifier faiblesses)

### 11.2 Visualisation des Filtres

**Première couche** : Visualiser directement les poids du kernel
- Interprétables (détecteurs de bords, couleurs)

**Couches profondes** : Non visualisables directement (trop abstraits)

### 11.3 Activation Maps

**Principe** : Visualiser quelle région active un neurone donné

**Méthode** :
1. Passer une image
2. Observer les activations d'une feature map
3. Upsampler à la résolution originale

**Interprétation** : Zones "importantes" pour un neurone

### 11.4 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Objectif** : Localiser les régions importantes pour la prédiction d'une classe

**Algorithme** :
1. Forward pass jusqu'à une couche conv $\$A$$
2. Calculer gradient de la classe $$c$$ par rapport à $\$A$$ : $$\frac{\partial y^c}{\partial A}$$
3. Global average pooling sur les gradients : $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$
4. Weighted sum : $\$L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

**Résultat** : Heatmap montrant les régions décisionnelles

---

## 12. Aspects Pratiques

### 12.1 Initialisation des Poids

**Pourquoi important ?**
- Mauvaise init → Gradients explosent ou disparaissent
- Symétrie : Si tous les poids identiques, les neurones apprennent la même chose

#### **Xavier / Glorot Initialization**
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

**Pour** : Tanh, Sigmoid

#### **He Initialization**
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

**Pour** : ReLU

**Pourquoi différent ?**
- ReLU tue 50% des activations → Compenser avec variance plus élevée

### 12.2 Hyperparamètres Clés

| Hyperparamètre | Valeurs typiques | Impact |
|----------------|------------------|--------|
| **Learning rate** | 0.001-0.1 | Vitesse convergence |
| **Batch size** | 32-256 | Stabilité gradient, mémoire |
| **Weight decay** | 1e-4 à 1e-5 | Régularisation |
| **Dropout** | 0.3-0.5 | Régularisation |
| **Epochs** | 50-200 | Temps entraînement |

### 12.3 Diagnostique : Overfit vs Underfit

**Signaux d'overfit** :
- Train accuracy ≫ Validation accuracy
- Validation loss augmente alors que train loss baisse

**Solutions** :
- Plus de données / augmentation
- Régularisation (dropout, weight decay)
- Réduire complexité modèle

**Signaux d'underfit** :
- Train et validation accuracy faibles

**Solutions** :
- Modèle plus complexe
- Entraîner plus longtemps
- Réduire régularisation

### 12.4 Métriques d'Évaluation

#### **Accuracy**
$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$

**Limitation** : Insensible aux classes déséquilibrées

#### **Précision & Rappel**
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### **F1-Score**
$\$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### **Top-5 Accuracy**
Prédit correctement si la vraie classe est dans les 5 prédictions les plus probables

---

## 13. Architectures Spécialisées

### 13.1 Fully Convolutional Networks (FCN)

**Idée** : Remplacer FC layers par convolutions
- **Avantage** : Accepte n'importe quelle taille d'image
- **Usage** : Segmentation sémantique

### 13.2 U-Net

**Architecture** : Encoder-Decoder avec skip connections
- **Downsampling** : Convolutions + pooling
- **Upsampling** : Transposed convolutions
- **Skip connections** : Concatène features haute résolution

**Usage** : Segmentation médicale (state-of-the-art)

### 13.3 Depthwise Separable Convolutions

**Principe** : Factoriser convolution standard en 2 étapes

1. **Depthwise** : Convolution par canal (pas de mixing)
2. **Pointwise** : Convolution 1×1 (mixing de canaux)

**Réduction de paramètres** :
- Standard : $\$K \times K \times C_{\text{in}} \times C_{\text{out}}$$
- Separable : $\$K \times K \times C_{\text{in}} + C_{\text{in}} \times C_{\text{out}}$$

**Ratio** : Environ $$\frac{1}{C_{\text{out}}} + \frac{1}{K^2}$$ (réduction de 8-9x pour $\$K=3$$)

**Usage** : MobileNet, EfficientNet (modèles légers pour mobile)

---

## 14. Challenges & Limitations

### 14.1 Besoins en Données

CNN nécessite **beaucoup de données annotées** (10k+ images minimum)
- Coûteux en labellisation
- Solutions : Transfer learning, semi-supervised, self-supervised

### 14.2 Adversarial Examples

**Problème** : Perturbations imperceptibles peuvent tromper le réseau

**Exemple** : $$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$$

→ Image quasi-identique, mais prédiction complètement différente

**Implications** : Sécurité (voitures autonomes, reconnaissance faciale)

### 14.3 Biais dans les Données

CNN apprend les **biais du dataset**
- Ex: "Médecin" → Toujours des hommes blancs
- Solutions : Audits, datasets diversifiés

### 14.4 Coût Computationnel

Training de gros modèles = $\$10^{18}$$ à $\$10^{20}$$ FLOPs
- Jours/semaines sur GPU/TPU
- Impact environnemental

---

## 15. Ressources & Sources

### Papers Fondamentaux

1. **LeNet-5** : LeCun et al. (1998)
   *"Gradient-Based Learning Applied to Document Recognition"*
   http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

2. **AlexNet** : Krizhevsky et al. (2012)
   *"ImageNet Classification with Deep Convolutional Neural Networks"*
   https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

3. **VGGNet** : Simonyan & Zisserman (2014)
   *"Very Deep Convolutional Networks for Large-Scale Image Recognition"*
   https://arxiv.org/abs/1409.1556

4. **GoogLeNet/Inception** : Szegedy et al. (2014)
   *"Going Deeper with Convolutions"*
   https://arxiv.org/abs/1409.4842

5. **ResNet** : He et al. (2015)
   *"Deep Residual Learning for Image Recognition"*
   https://arxiv.org/abs/1512.03385

6. **Batch Normalization** : Ioffe & Szegedy (2015)
   *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"*
   https://arxiv.org/abs/1502.03167

7. **Dropout** : Srivastava et al. (2014)
   *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"*
   http://jmlr.org/papers/v15/srivastava14a.html

### Architectures Modernes

8. **EfficientNet** : Tan & Le (2019)
   *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*
   https://arxiv.org/abs/1905.11946

9. **Vision Transformer** : Dosovitskiy et al. (2020)
   *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*
   https://arxiv.org/abs/2010.11929

### Interprétabilité

10. **Grad-CAM** : Selvaraju et al. (2017)
    *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*
    https://arxiv.org/abs/1610.02391

### Livres & Cours

- **Deep Learning Book** : Goodfellow, Bengio, Courville (2016)
  https://www.deeplearningbook.org/

- **CS231n: Convolutional Neural Networks for Visual Recognition** (Stanford)
  http://cs231n.stanford.edu/

---

## 16. Récapitulatif : Points Clés

### Concepts Fondamentaux
✅ **Convolution** : Opération locale, partage de poids → invariance translation  
✅ **Hiérarchie** : Features simples → complexes (bords → objets)  
✅ **Pooling** : Downsampling, invariance, réduction paramètres  
✅ **Activations** : ReLU = standard (pas de saturation)  

### Mathématiques
✅ **Output size** : $$\lfloor \frac{n + 2p - k}{s} \rfloor + 1$$  
✅ **Params convolution** : $\$F \times (K \times K \times C_{\text{in}} + 1)$$  
✅ **Receptive field** : Augmente avec profondeur (contexte global)  
✅ **Backprop** : Chain rule, gradient = convolution  

### Architectures
✅ **AlexNet** : Révolution (2012), ReLU + Dropout  
✅ **VGG** : Simplicité (stacks de 3×3)  
✅ **ResNet** : Skip connections → réseaux ultra-profonds  
✅ **EfficientNet** : État de l'art efficacité  

### Entraînement
✅ **Cross-entropy loss** : Standard classification  
✅ **Adam optimizer** : Adaptatif, robuste  
✅ **Batch Normalization** : Convergence rapide  
✅ **Learning rate scheduling** : Warmup + decay  

### Pratique
✅ **Transfer learning** : Toujours essayer en premier  
✅ **Data augmentation** : Crucial pour généralisation  
✅ **Régularisation** : Dropout + weight decay + augmentation  
✅ **Diagnostique** : Monitor train vs validation gap  

### Limitations
✅ **Données** : Nécessite beaucoup d'annotations  
✅ **Adversarial** : Vulnérable à perturbations  
✅ **Biais** : Reproduit biais des données  
✅ **Coût** : Computationnel et environnemental  

---

**Fin du cours** - Tu as maintenant une compréhension complète des CNN, de leurs fondations mathématiques aux applications pratiques ! 🚀
