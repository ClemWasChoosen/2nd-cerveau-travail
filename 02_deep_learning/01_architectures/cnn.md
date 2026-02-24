# Réseaux de Neurones Convolutifs (CNN)

> **Résumé** : Les CNN exploitent la structure spatiale des images via des convolutions locales et le partage de poids, permettant d'apprendre des hiérarchies de features (bords → formes → objets) avec drastiquement moins de paramètres que les réseaux fully-connected. Architecture fondamentale de la vision par ordinateur depuis 2012.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2024 (date à préciser) |
| **Dernière mise à jour** | 2026-02-24 |
| **Domaine** | Deep Learning |
| **Sous-domaine** | Architectures - Computer Vision |
| **Niveau** | ⭐⭐ Intermédiaire |
| **Durée de lecture** | ~60 minutes (cours très complet) |
| **Fichier** | `cnn.md` |
| **Emplacement** | `/02_deep_learning/01_architectures/` |
| **Tags** | `#cnn` `#computer-vision` `#convolution` `#image-classification` `#resnet` `#architecture` |

---

## 🔗 Navigation

### Prérequis

- [x] **[Fonctions d'Activation](../02_fundamentals/fonction_activation.md)** - Obligatoire : comprendre ReLU, Sigmoid, Tanh
- [x] **[Preprocessing Data](../../01_machine_learning/01_fundamentals/preprocessing_data.md)** - Normalisation des images cruciale
- [ ] Algèbre linéaire (matrices, produit matriciel)
- [ ] Calcul différentiel (dérivées, chain rule pour backpropagation)
- [ ] Python et NumPy (pour implémentation)

### Cours connexes

- **Alternatives modernes** : [Vision Transformers (ViT)](./vit.md) - Architecture sans convolution (2020+)
- **Application CNN** : [YOLO](./yolo.md) - Détection d'objets en temps réel basée sur CNN
- **Interprétabilité** : [Integrated Gradients](../04_interpretabilite/integrated_gradients.md) - Expliquer les décisions CNN

### Suite recommandée

- [Vision Transformers](./vit.md) - Comprendre l'alternative moderne
- [YOLO](./yolo.md) - Application pratique pour détection
- [Integrated Gradients](../04_interpretabilite/integrated_gradients.md) - Visualiser ce que le CNN apprend

---

## 🎯 Objectifs d'apprentissage

À la fin de ce cours, vous serez capable de :

1. **Comprendre** pourquoi les convolutions sont supérieures aux fully-connected pour les images (localité, partage de poids, hiérarchie)
2. **Calculer** les dimensions de sortie d'une couche convolutionnelle (avec stride, padding, dilation)
3. **Analyser** les architectures classiques (LeNet, AlexNet, VGG, ResNet) et leurs innovations respectives
4. **Implémenter** un CNN complet en PyTorch pour classification d'images
5. **Diagnostiquer** les problèmes d'entraînement (overfitting, gradient vanishing, dying ReLU)
6. **Appliquer** le transfer learning pour de nouvelles tâches
7. **Interpréter** les décisions via Grad-CAM et visualisation de filtres


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
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B```{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

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

De manière générale, le "backbone (colonne vertebrale)" extrait la feature map à plusieurs résolutions différentes. Le "neck (cou)" combine ces maps et enfin, la "head (tête)", réalise la prédiction finale (bounding box, classe associée, ...).  

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
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out```\right)$$

**Pour** : Tanh, Sigmoid

#### **He Initialization**
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in```\right)$$

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

**Ratio** : Environ $$\frac{1}{C_{\text{out``` + \frac{1}{K^2}$$ (réduction de 8-9x pour $\$K=3$$)

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

## 💻 Implémentation Pratique

### Exemple Complet : CNN pour CIFAR-10 (PyTorch)

```python
"""
CNN pour Classification CIFAR-10 (10 classes, images 32×32×3)
Architecture inspirée de VGG avec adaptations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# PARTIE 1 : DÉFINITION DE L'ARCHITECTURE
# ============================================

class SimpleCNN(nn.Module):
    """
    CNN simple pour CIFAR-10
    Architecture : Conv → Conv → Pool → Conv → Conv → Pool → FC → FC
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Bloc 1 : 32×32×3 → 32×32×64 → 16×16×64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # same padding
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Bloc 2 : 16×16×64 → 16×16×128 → 8×8×128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # Bloc 3 : 8×8×128 → 8×8×256 → 4×4×256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8×8 → 4×4
        
        # Couches fully-connected
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Bloc 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Bloc 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Bloc 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 256*4*4)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Bloc résiduel pour ResNet-like architecture
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (si dimension change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet adapté pour CIFAR-10 (images 32×32)
    """
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# ============================================
# PARTIE 2 : CHARGEMENT DES DONNÉES
# ============================================

def get_cifar10_loaders(batch_size=128):
    """
    Charge CIFAR-10 avec data augmentation pour train
    """
    
    # Normalisation : mean et std de CIFAR-10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Transformations pour train (avec augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Transformations pour test (seulement normalisation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Téléchargement et chargement
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# ============================================
# PARTIE 3 : ENTRAÎNEMENT
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entraîne le modèle sur une epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(loader)} - '
                  f'Loss: {loss.item():.3f} - '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(loader), 100. * correct / total


def test(model, loader, criterion, device):
    """
    Évalue le modèle sur le test set
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(loader), 100. * correct / total


def train_model(model, train_loader, test_loader, epochs=50, lr=0.001, device='cuda'):
    """
    Boucle d'entraînement complète
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Sauvegarder historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%')
        
        # Sauvegarder meilleur modèle
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ Nouveau meilleur modèle sauvegardé (acc: {best_acc:.2f}%)')
    
    return history


# ============================================
# PARTIE 4 : VISUALISATION
# ============================================

def plot_training_history(history):
    """
    Visualise les courbes d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss au cours de l\'entraînement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy au cours de l\'entraînement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def visualize_filters(model, layer_name='conv1'):
    """
    Visualise les filtres de la première couche convolutionnelle
    """
    # Extraire les poids
    if hasattr(model, layer_name):
        weights = getattr(model, layer_name).weight.data.cpu()
    else:
        print(f"Couche {layer_name} non trouvée")
        return
    
    # Normaliser pour affichage
    weights = weights - weights.min()
    weights = weights / weights.max()
    
    num_filters = min(64, weights.shape[0])
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Afficher le filtre (moyenner les canaux RGB si 3 canaux)
            if weights.shape[1] == 3:
                img = weights[i].permute(1, 2, 0).numpy()
            else:
                img = weights[i, 0].numpy()
            
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Filtres de la couche {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('conv_filters.png', dpi=150)
    plt.show()


# ============================================
# PARTIE 5 : UTILISATION
# ============================================

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Utilisation de : {device}')
    
    # Chargement données
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Choix du modèle
    # model = SimpleCNN(num_classes=10).to(device)
    model = ResNetCIFAR(num_classes=10).to(device)
    
    # Afficher architecture
    print(f'\nNombre de paramètres : {sum(p.numel() for p in model.parameters()):,}')
    
    # Entraînement
    history = train_model(
        model, train_loader, test_loader,
        epochs=50, lr=0.001, device=device
    )
    
    # Visualisations
    plot_training_history(history)
    visualize_filters(model, layer_name='conv1')
    
    print(f'\n✓ Entraînement terminé !')
    print(f'Meilleure test accuracy : {max(history["test_acc"]):.2f}%')
```

---

## ⚖️ Comparaisons : CNN vs Alternatives

### CNN vs Vision Transformers (ViT)

| Critère | CNN | Vision Transformers | Gagnant |
|---------|-----|---------------------|---------|
| **Performance (small data < 10k)** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Faible (besoin de prétraining) | **CNN** |
| **Performance (large data > 1M)** | ⭐⭐⭐⭐ Très bon | ⭐⭐⭐⭐⭐ État de l'art | **ViT** |
| **Inductive bias (localité)** | Fort (bénéfique small data) | Faible (apprend from scratch) | **CNN** (small data) |
| **Champ récepteur global** | Nécessite profondeur | Dès la 1ère couche (attention) | **ViT** |
| **Vitesse d'inférence** | ⭐⭐⭐⭐ Rapide | ⭐⭐⭐ Plus lent (attention coûteuse) | **CNN** |
| **Taille modèle** | Compact (ResNet-50: 25M params) | Large (ViT-B: 86M params) | **CNN** |
| **Interprétabilité** | ⭐⭐⭐ Filtres visualisables | ⭐⭐⭐⭐ Attention maps claires | **ViT** |

**Quand utiliser CNN** :
- ✅ Dataset < 100k images
- ✅ Contraintes latence/mémoire (edge deployment)
- ✅ Tâches classiques bien résolues (classification, détection)
- ✅ Besoin de compréhension des features (filtres)

**Quand utiliser ViT** :
- ✅ Dataset > 1M images ou access à modèle prétrainé
- ✅ Tâche complexe nécessitant contexte global
- ✅ Ressources GPU abondantes
- ✅ État de l'art recherche

**Tendance actuelle (2024-2026)** : **Hybrid models** (ConvNeXt, CoAtNet)
- Combinent meilleur des deux mondes
- Conv pour features locales + Attention pour contexte global

📖 **Voir aussi** : [Vision Transformers (ViT)](./vit.md)

---

### CNN vs Fully-Connected Networks

| Aspect | Fully-Connected | CNN | Avantage CNN |
|--------|----------------|-----|--------------|
| **Params (image 224×224×3, 1000 neurones)** | 150M | ~2k (conv 3×3) | **75,000×** moins |
| **Invariance translation** | ❌ Non | ✅ Oui (partage poids) | **CNN** |
| **Structure spatiale** | ❌ Perdue (flatten) | ✅ Préservée | **CNN** |
| **Hiérarchie features** | ❌ Manuelle | ✅ Automatique | **CNN** |
| **Overfitting** | ⚠️ Très sensible | ✅ Régularisé (partage) | **CNN** |

**Conclusion** : CNN systématiquement supérieur pour données spatiales (images, vidéos)

---

## 💡 Points Clés à Retenir

- 🔑 **Convolutions exploitent 3 principes** : Localité spatiale, partage de poids, hiérarchie de features
- 🔑 **Réduction paramétrique drastique** : Kernel 3×3 = 9 params vs millions pour FC
- 🔑 **Formule dimension output** : $$\text{output} = \lfloor \frac{n + 2p - k}{s} \rfloor + 1$$
- 🔑 **Batch Normalization = Standard** : Après Conv, avant ReLU → Stabilise et accélère
- 🔑 **ResNet révolution (2015)** : Skip connections permettent réseaux profonds (> 100 couches)
- 🔑 **Transfer Learning = Pratique standard** : Prétraining ImageNet puis fine-tuning
- 🔑 **Data Augmentation cruciale** : Random crops, flips → Évite overfitting
- 🔑 **Adam optimizer** : Fonctionne bien "out of the box" pour la plupart des cas
- 🔑 **Global Average Pooling** : Remplace FC en sortie → Moins de params, moins d'overfit
- 🔑 **Architectures modernes** : ResNet-50/EfficientNet = compromis performance/coût optimal

---

## ⚠️ Pièges à Éviter

| ❌ Erreur courante | ✅ Bonne pratique | 💡 Pourquoi |
|-------------------|-------------------|-------------|
| Oublier de normaliser les images | Normaliser avec mean/std du dataset | CNN très sensible à l'échelle des pixels |
| Utiliser uniquement convolutions 5×5 ou plus | Empiler des 3×3 | Même champ récepteur, moins de params, plus de non-linéarités |
| Padding=0 systématiquement | Padding='same' pour préserver dimension | Évite shrinkage, préserve info de bord |
| Batch Norm après activation | Conv → BatchNorm → ReLU | Normaliser avant non-linéarité (standard actuel) |
| Learning rate trop élevé au début | Warmup (augmentation progressive) | BatchNorm instable initialement |
| Fine-tuning avec même LR que training | LR 10-100× plus faible | Évite de "casser" les features pré-apprises |
| Ignorer data augmentation | Toujours augmenter (sauf test set) | Réduit drastiquement overfitting |
| Sauvegarder seulement le dernier modèle | Sauvegarder best validation accuracy | Évite overfit des derniers epochs |
| Dropout sur couches conv | Dropout seulement sur FC | Conv déjà régularisées (partage poids) |
| Oublier model.eval() en inférence | Toujours model.eval() avant test | BatchNorm/Dropout comportent différemment |

---

## 📚 Ressources Complémentaires

### Papers Fondamentaux

1. **"Gradient-Based Learning Applied to Document Recognition"** - LeCun et al. (1998)
   - [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
   - 📌 **LeNet-5** : Première CNN moderne, reconnaissance MNIST
   
2. **"ImageNet Classification with Deep Convolutional Neural Networks"** - Krizhevsky et al. (2012)
   - [PDF](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
   - 📌 **AlexNet** : Déclenche la révolution deep learning (ImageNet 2012)
   
3. **"Very Deep Convolutional Networks for Large-Scale Image Recognition"** - Simonyan & Zisserman (2014)
   - [arXiv](https://arxiv.org/abs/1409.1556)
   - 📌 **VGGNet** : Prouve l'efficacité des convolutions 3×3 empilées
   
4. **"Deep Residual Learning for Image Recognition"** - He et al. (2015)
   - [arXiv](https://arxiv.org/abs/1512.03385)
   - 📌 **ResNet** : Skip connections → Réseaux ultra-profonds possibles
   - 🏆 **Impact majeur** : Architecture la plus influente post-2015

5. **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** - Tan & Le (2019)
   - [arXiv](https://arxiv.org/abs/1905.11946)
   - 📌 **EfficientNet** : Optimisation simultanée profondeur/largeur/résolution

### Tutorials & Documentation

- **[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)** - Cours Stanford (référence)
- **[PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)** - Official tutorial
- **[Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/)** - Visualisations interactives

### Outils & Frameworks

- **[PyTorch](https://pytorch.org/)** - Framework de référence recherche
- **[TorchVision](https://pytorch.org/vision/stable/index.html)** - Modèles pré-entraînés (ResNet, VGG, etc.)
- **[Timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models)** - 700+ architectures pré-entraînées
- **[Netron](https://netron.app/)** - Visualiser architectures CNN

### Datasets

- **[ImageNet](https://image-net.org/)** - 1.4M images, 1000 classes (standard prétraining)
- **[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)** - 60k images 32×32 (benchmark rapide)
- **[COCO](https://cocodataset.org/)** - Détection d'objets, segmentation

---

## 🔙 Navigation

**Retour** : [INDEX Deep Learning](../INDEX.md) | [README Principal](../../README.md)

**Cours connexes dans ce domaine** :
- [Fonctions d'Activation](../02_fundamentals/fonction_activation.md) - Prérequis fondamental
- [Vision Transformers](./vit.md) - Alternative moderne sans convolutions
- [YOLO](./yolo.md) - Application CNN pour détection temps réel

**Progression recommandée** :
1. ✅ Fonctions d'Activation (prérequis)
2. ✅ CNN (ce cours)
3. → [YOLO](./yolo.md) - Application pratique
4. → [Vision Transformers](./vit.md) - Comprendre l'évolution post-CNN
5. → [Integrated Gradients](../04_interpretabilite/integrated_gradients.md) - Interpréter les CNN

---

*Dernière mise à jour : 2026-02-24*
*Cours complet sur les CNN : Des fondamentaux mathématiques aux architectures modernes*

**Fin du cours** - Tu as maintenant une compréhension complète des CNN, de leurs fondations mathématiques aux applications pratiques ! 🚀
