# Vision Transformer (ViT) : Guide Complet

## 1. Introduction et Contexte

### 1.1 Le problème avec les CNNs

Avant les ViT, les réseaux de neurones convolutifs (CNNs) dominaient la vision par ordinateur. Mais pourquoi chercher une alternative ?

**Limitations des CNNs :**
- **Biais inductif fort** : Les convolutions imposent une localité spatiale (chaque neurone ne voit qu'une petite fenêtre)
- **Champ réceptif limité** : Nécessite de nombreuses couches pour capturer des dépendances à longue distance
- **Difficile à paralléliser** : Les opérations sont séquentielles par nature
- **Scalabilité** : Les CNNs saturent en performance avec plus de données

**L'intuition des Transformers :**
Les Transformers, qui ont révolutionné le NLP (BERT, GPT), utilisent l'**attention** pour capturer des relations globales dès la première couche. Question : pourquoi ne pas appliquer ce mécanisme aux images ?

### 1.2 L'idée révolutionnaire de ViT

**Paper fondateur** : *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* (Dosovitskiy et al., Google Research, 2020)

**L'analogie avec le NLP :**
- En NLP : un texte = séquence de mots (tokens)
- En Vision : une image = séquence de patches (morceaux d'image)

**Principe :** Découper une image en patches, les traiter comme des tokens, et utiliser un Transformer encoder standard.

---

## 2. Architecture Détaillée

### 2.1 Vue d'ensemble

```
Image (H × W × C)
     ↓
Découpage en patches (N patches de P × P × C)
     ↓
Linear Projection (Embedding)
     ↓
Ajout Position Embeddings + [CLS] token
     ↓
Transformer Encoder (L couches)
     ↓
MLP Head (Classification)
     ↓
Classe prédite
```

### 2.2 Patch Embedding (Étape 1)

**Pourquoi découper en patches ?**
Les images sont trop grandes pour être traitées pixel par pixel (une image 224×224 = 50 176 pixels). Les patches réduisent la complexité.

**Mathématiquement :**

Image d'entrée : $$\mathbf{x} \in \mathbb{R}^{H \times W \times C}$$

Où :
- $$H, W$$ = hauteur, largeur
- $$C$$ = nombre de canaux (3 pour RGB)

On découpe en $$N$$ patches de taille $$P \times P$$ :

$$N = \frac{H \times W}{P^2}$$

**Exemple concret :**
- Image 224×224×3
- Patch size P=16
- Nombre de patches : $$N = \frac{224 \times 224}{16 \times 16} = 196$$ patches

Chaque patch devient un vecteur aplati :

$$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

Pour notre exemple : $$196 \times (16 \times 16 \times 3) = 196 \times 768$$

**Projection linéaire :**

On projette chaque patch dans un espace de dimension $$D$$ (dimension du modèle) :

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; ...; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$$

Où :
- $$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$ : matrice d'embedding (learnable)
- $$\mathbf{x}_{class}$$ : token [CLS] (voir section suivante)
- $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$ : position embeddings

**Implémentation pratique :**

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Projection linéaire via une convolution
        # Pourquoi conv2d ? C'est équivalent à découper + linear mais plus efficace
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

# Test
patch_embed = PatchEmbedding()
x = torch.randn(2, 3, 224, 224)  # Batch de 2 images
patches = patch_embed(x)
print(f"Shape: {patches.shape}")  # (2, 196, 768)
```

### 2.3 CLS Token et Position Embeddings

**Le token [CLS] :**

Inspiré de BERT, on ajoute un token spécial au début de la séquence. Son rôle :
- **Agrégateur global** : Ce token "observe" tous les patches via l'attention
- **Représentation de l'image** : Sa sortie finale sert pour la classification

$$\mathbf{x}_{class} \in \mathbb{R}^D$$ : vecteur learnable

**Position Embeddings :**

**Pourquoi nécessaire ?** Les Transformers n'ont pas de notion d'ordre naturel (contrairement aux RNNs). Sans position embeddings, permuter les patches ne changerait rien !

**Types de position encoding :**

1. **Learnable (utilisé dans ViT) :**
   $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$ : matrice apprise durant l'entraînement

2. **Sinusoïdal (Transformer original) :**
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

**Pourquoi learnable fonctionne mieux pour ViT ?**
- Les positions 2D d'une image ont une structure plus complexe qu'une séquence 1D
- Le modèle apprend implicitement les relations spatiales

```python
class ViTEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        
        self.patch_embeddings = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # CLS token : un paramètre learnable
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embeddings : N patches + 1 CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embeddings
        x = self.patch_embeddings(x)  # (B, n_patches, embed_dim)
        
        # Répéter le CLS token pour chaque élément du batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        
        # Concaténer CLS token au début
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # Ajouter position embeddings
        x = x + self.position_embeddings
        
        return x
```

### 2.4 Transformer Encoder

**Rappel : qu'est-ce qu'un Transformer Encoder ?**

Un bloc Transformer est composé de :
1. **Multi-Head Self-Attention (MSA)**
2. **MLP (Feed-Forward Network)**
3. **Layer Normalization** (avant chaque bloc)
4. **Connexions résiduelles**

**Architecture d'un bloc :**

```
Input x
  ↓
LayerNorm
  ↓
Multi-Head Attention
  ↓
+ (residual)
  ↓
LayerNorm
  ↓
MLP
  ↓
+ (residual)
  ↓
Output
```

**Mathématiquement :**

Pour la couche $$l$$ :

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

Où :
- $$\text{LN}$$ : Layer Normalization
- $$\text{MSA}$$ : Multi-Head Self-Attention
- $$\text{MLP}$$ : Multi-Layer Perceptron

---

### 2.5 Multi-Head Self-Attention (Le cœur du modèle)

**Self-Attention : l'intuition**

Chaque patch (token) "regarde" tous les autres patches pour comprendre le contexte global.

**Question : Comment un patch décide-t-il à quels autres patches prêter attention ?**

Réponse : via des scores d'attention calculés par similarité.

**Mathématiquement :**

Pour chaque token $$i$$, on crée 3 vecteurs :
- **Query (Q)** : "Qu'est-ce que je cherche ?"
- **Key (K)** : "Qu'est-ce que je contiens ?"
- **Value (V)** : "Quelle information je fournis ?"

$$\mathbf{Q} = \mathbf{z}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{z}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{z}\mathbf{W}_V$$

Où $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D_h}$$ sont des matrices learnable.

**Attention Scores :**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{D_h}}\right)\mathbf{V}$$

**Pourquoi diviser par $$\sqrt{D_h}$$ ?**

Sans normalisation, pour de grandes dimensions, le produit scalaire $$\mathbf{QK}^T$$ devient très grand, poussant le softmax vers des valeurs extrêmes (gradient vanishing). La division stabilise l'entraînement.

**Démonstration mathématique :**

Si $$Q_i, K_i \sim \mathcal{N}(0, 1)$$ (distribués normalement), alors :

$$\mathbb{E}[\mathbf{Q}\mathbf{K}^T] = 0$$

$$\text{Var}(\mathbf{Q}\mathbf{K}^T) = D_h$$

En divisant par $$\sqrt{D_h}$$, on ramène la variance à 1 :

$$\text{Var}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_h}}\right) = 1$$

**Multi-Head Attention :**

Au lieu d'une seule attention, on en calcule $$h$$ en parallèle (chacune avec ses propres $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$$).

**Pourquoi plusieurs têtes ?**

Chaque tête peut se spécialiser sur différents aspects :
- Tête 1 : relations de couleur
- Tête 2 : formes géométriques
- Tête 3 : textures
- etc.

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}_O$$

Où :

$$\text{head}_i = \text{Attention}(\mathbf{QW}_Q^i, \mathbf{KW}_K^i, \mathbf{VW}_V^i)$$

**Implémentation :**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        # Projections Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Projection de sortie
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape  # Batch, Num_tokens, Embed_dim
        
        # Calculer Q, K, V en une seule opération
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        
        # Appliquer attention sur V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        
        # Projection finale
        x = self.proj(x)
        
        return x

# Visualiser les attention maps
import matplotlib.pyplot as plt

def visualize_attention(attn_weights, img_size=224, patch_size=16):
    """
    attn_weights: (num_heads, N+1, N+1) où N = nombre de patches
    """
    n_patches = (img_size // patch_size)
    
    # Prendre l'attention du CLS token (première position)
    cls_attn = attn_weights[:, 0, 1:]  # (num_heads, N)
    
    # Reshaper en grille 2D
    cls_attn = cls_attn.reshape(-1, n_patches, n_patches)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for idx, ax in enumerate(axes.flat):
        if idx < cls_attn.shape[0]:
            ax.imshow(cls_attn[idx].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Head {idx+1}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
```

### 2.6 MLP (Feed-Forward Network)

Après l'attention, chaque token passe par un MLP à 2 couches :

$$\text{MLP}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

**Architecture typique :**
- Couche 1 : $$D \to 4D$$ (expansion)
- Activation GELU
- Couche 2 : $$4D \to D$$ (compression)

**Pourquoi cette expansion (hidden dim = 4×embed_dim) ?**

Le MLP augmente la capacité du modèle à capturer des transformations non-linéaires complexes. L'expansion permet au modèle d'explorer un espace de représentation plus riche.

**GELU vs ReLU :**

- **ReLU** : $$\max(0, x)$$ - hard cutoff
- **GELU** : $$x \cdot \Phi(x)$$ où $$\Phi$$ est la CDF de la gaussienne - smooth

GELU performe mieux en pratique car :
- Plus smooth (meilleurs gradients)
- Permet des valeurs légèrement négatives
- Utilisé dans BERT, GPT, etc.

```python
class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

### 2.7 Bloc Transformer Complet

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        # Attention block avec residual
        x = x + self.attn(self.norm1(x))
        
        # MLP block avec residual
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### 2.8 Architecture Complète du ViT

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Embeddings
        self.embeddings = ViTEmbeddings(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer Encoder (pile de blocs)
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer Norm finale
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head (MLP)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Embeddings
        x = self.embeddings(x)  # (B, N+1, embed_dim)
        
        # Encoder
        x = self.encoder(x)  # (B, N+1, embed_dim)
        
        # Layer Norm
        x = self.norm(x)
        
        # Extraire le CLS token (première position)
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # Classification
        logits = self.head(cls_token)  # (B, num_classes)
        
        return logits

# Exemple d'utilisation
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Nombre de paramètres
n_params = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres : {n_params / 1e6:.1f}M")  # ~86M pour ViT-Base

# Test forward pass
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # (2, 1000)
```

---

## 3. Variants du ViT

### 3.1 ViT-Base, ViT-Large, ViT-Huge

| Modèle | Layers (L) | Hidden Size (D) | MLP Size | Heads | Params |
|--------|-----------|----------------|----------|-------|--------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

### 3.2 Taille des Patches

- **ViT-B/16** : patches de 16×16 (196 patches pour 224×224)
- **ViT-B/32** : patches de 32×32 (49 patches)
- **ViT-B/8** : patches de 8×8 (784 patches) - plus précis mais plus coûteux

**Trade-off :**
- Petits patches = plus de détails mais plus coûteux en calcul (attention est O(N²))
- Grands patches = moins cher mais perd des détails fins

---

## 4. Entraînement du ViT

### 4.1 Le Paradoxe : Plus de Données = Meilleures Performances

**Observation clé du paper :**

| Dataset | ViT-B/16 | ResNet-152 |
|---------|----------|------------|
| ImageNet (1.3M) | 77.9% | **78.8%** |
| ImageNet-21k (14M) | **83.1%** | 81.8% |
| JFT-300M (300M) | **88.4%** | 85.7% |

**Pourquoi ce comportement ?**

1. **Biais inductif faible** : ViT n'impose pas de structure spatiale locale (contrairement aux CNNs avec leurs convolutions)
   - Avantage : plus flexible, apprend des patterns arbitraires
   - Inconvénient : nécessite beaucoup de données pour apprendre ces structures

2. **CNNs ont un biais inductif fort** :
   - Translation equivariance (déplacer l'objet = déplacer la détection)
   - Localité spatiale
   - → Meilleures performances avec peu de données

**En pratique : stratégie d'entraînement**

```
Étape 1 : Pre-training sur large dataset
    ↓ (ImageNet-21k ou JFT-300M)
Étape 2 : Fine-tuning sur dataset cible
    ↓ (ImageNet-1k, CIFAR, etc.)
Meilleure performance
```

### 4.2 Pré-entraînement

**Tâche : Classification supervisée**

Loss : Cross-Entropy

$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

**Hyperparamètres (du paper) :**
- Optimizer : Adam ($$\beta_1=0.9, \beta_2=0.999$$)
- Learning rate : warmup puis decay
- Batch size : 4096
- Weight decay : 0.1
- Gradient clipping : 1.0

**Data Augmentation :**
- RandAugment
- Mixup ($$\alpha=0.5$$)
- Cutmix

**Pourquoi ces augmentations ?**

Sans biais inductif, le modèle peut overfitter. Les augmentations agissent comme régularisation.

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1
)

# Learning rate schedule avec warmup
def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, num_warmup_steps=10000, num_training_steps=100000)
```

### 4.3 Fine-tuning

Lors du fine-tuning sur un dataset cible :

1. **Remplacer le head de classification** :
   - Supprimer l'ancien MLP head (num_classes du pre-training)
   - Ajouter un nouveau head avec le bon nombre de classes
   - Initialiser avec des petits poids aléatoires

2. **Learning rate plus faible** : typiquement 10× plus petit que le pre-training

3. **Résolution d'image différente** :
   - Si on fine-tune avec des images plus grandes (ex: 384×384 au lieu de 224×224)
   - Problème : plus de patches → position embeddings incompatibles
   - Solution : **interpolation 2D des position embeddings**

**Interpolation des position embeddings :**

```python
def interpolate_pos_embed(pos_embed, orig_size, new_size):
    """
    pos_embed: (1, N+1, D) où N = nombre original de patches
    orig_size: taille originale (ex: 14 pour 224/16)
    new_size: nouvelle taille (ex: 24 pour 384/16)
    """
    # Séparer CLS token et patch embeddings
    cls_embed = pos_embed[:, :1, :]  # (1, 1, D)
    patch_embed = pos_embed[:, 1:, :]  # (1, N, D)
    
    D = patch_embed.shape[-1]
    
    # Reshaper en grille 2D
    patch_embed = patch_embed.reshape(1, orig_size, orig_size, D)
    patch_embed = patch_embed.permute(0, 3, 1, 2)  # (1, D, orig_size, orig_size)
    
    # Interpolation bilinéaire
    patch_embed = nn.functional.interpolate(
        patch_embed,
        size=(new_size, new_size),
        mode='bilinear',
        align_corners=False
    )
    
    # Reshaper en séquence
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, -1, D)
    
    # Recombiner avec CLS token
    pos_embed = torch.cat([cls_embed, patch_embed], dim=1)
    
    return pos_embed
```

---

## 5. Comparaison ViT vs CNNs

### 5.1 Complexité Computationnelle

**Self-Attention :**

Pour $$N$$ patches de dimension $$D$$ :
- Calculer Q, K, V : $$O(N \cdot D^2)$$
- Attention scores $$\mathbf{QK}^T$$ : $$O(N^2 \cdot D)$$
- Appliquer sur V : $$O(N^2 \cdot D)$$
- **Total : $$O(N^2 \cdot D)$$** ← quadratique en nombre de patches !

**CNN :**

Pour une image $$H \times W$$ avec $$C$$ canaux et kernel $$K \times K$$ :
- Coût par couche : $$O(H \cdot W \cdot C \cdot K^2)$$
- **Linéaire en taille d'image**

**Pourquoi ViT est plus lent sur petites images ?**

Pour 224×224 avec patches 16×16 : $$N=196$$
- Attention : $$O(196^2 \cdot 768) \approx 30M$$ opérations
- Les CNNs sont optimisés avec decades de recherche

**Optimisations modernes :**
- Flash Attention (réduction mémoire)
- Linformer, Performer (approximations linéaires)
- DeiT (Data-efficient image Transformers)

### 5.2 Biais Inductif

| Aspect | CNN | ViT |
|--------|-----|-----|
| Localité spatiale | Forte (convolutions) | Faible (attention globale) |
| Translation equivariance | Oui (built-in) | Non (doit apprendre) |
| Data efficiency | Bonne (petit dataset) | Mauvaise (besoin beaucoup de données) |
| Scalabilité | Sature rapidement | Continue d'améliorer avec + de données |
| Interprétabilité | Difficile | Plus facile (attention maps) |

### 5.3 Performances State-of-the-Art (2024)

**ImageNet-1k Top-1 Accuracy :**

| Modèle | Params | Accuracy | Notes |
|--------|--------|----------|-------|
| ResNet-152 | 60M | 78.8% | Baseline CNN |
| EfficientNetV2-L | 120M | 85.7% | CNN optimisé |
| ViT-B/16 (ImageNet-21k) | 86M | 84.5% | ViT original |
| ViT-L/16 (ImageNet-21k) | 307M | 87.8% | ViT Large |
| DeiT-III-L | 304M | 87.2% | ViT data-efficient |
| BEiT-3-L | 675M | 89.6% | Self-supervised pre-training |

---

## 6. Variants et Extensions

### 6.1 DeiT (Data-efficient image Transformers)

**Problème résolu :** Entraîner ViT sans énormes datasets

**Innovations :**
1. **Distillation token** : token supplémentaire qui apprend d'un teacher CNN
2. **Augmentations fortes** : RandAugment, CutMix, Mixup
3. **Régularisation** : Stochastic Depth, Label Smoothing

```python
class DeiT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Architecture ViT standard
        self.vit = VisionTransformer(*args, **kwargs)
        
        # Token de distillation (en plus du CLS token)
        self.dist_token = nn.Parameter(torch.randn(1, 1, kwargs['embed_dim']))
        
    def forward(self, x):
        B = x.shape[0]
        
        # Embeddings + CLS token
        x = self.vit.embeddings(x)
        
        # Ajouter distillation token
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([x[:, :1], dist_tokens, x[:, 1:]], dim=1)  # [CLS, DIST, patches]
        
        # Encoder
        x = self.vit.encoder(x)
        
        # Deux sorties : classification et distillation
        cls_output = self.vit.head(x[:, 0])
        dist_output = self.vit.head(x[:, 1])
        
        return cls_output, dist_output
```

### 6.2 Swin Transformer

**Problème résolu :** Complexité quadratique de l'attention

**Idée : Hierarchical Vision Transformer**

Au lieu d'attention globale sur tous les patches :
1. **Windowed attention** : attention locale dans des fenêtres (ex: 7×7)
2. **Shifted windows** : décaler les fenêtres entre couches pour connecter les régions
3. **Patch merging** : réduire progressivement la résolution (comme CNNs)

**Complexité :** $$O(N)$$ au lieu de $$O(N^2)$$

```
Stage 1: 56×56 patches, dim=96
    ↓ (patch merging)
Stage 2: 28×28 patches, dim=192
    ↓ (patch merging)
Stage 3: 14×14 patches, dim=384
    ↓ (patch merging)
Stage 4: 7×7 patches, dim=768
```

### 6.3 MAE (Masked Autoencoders)

**Approche self-supervised** : BERT pour les images

1. Masquer aléatoirement 75% des patches
2. Encoder les patches visibles avec ViT
3. Décoder pour reconstruire les patches masqués
4. Loss : MSE entre pixels originaux et reconstruits

$$\mathcal{L} = \frac{1}{M}\sum_{i \in \text{masked}} ||\mathbf{x}_i - \hat{\mathbf{x}}_i||^2$$

**Pourquoi ça marche ?**

Forcer le modèle à comprendre la structure spatiale et sémantique pour reconstruire.

```python
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder  # ViT
        self.decoder = decoder  # ViT léger
        self.mask_ratio = mask_ratio
        
    def random_masking(self, x):
        """
        x: (B, N, D)
        Retourne : x_masked, mask, ids_restore
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Permutation aléatoire
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Garder seulement len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Masque binaire : 0 = keep, 1 = remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # Embeddings
        x = self.encoder.embeddings(x)  # (B, N+1, D)
        
        # Masquer (sans le CLS token)
        cls_token = x[:, :1, :]
        patches = x[:, 1:, :]
        patches_masked, mask, ids_restore = self.random_masking(patches)
        
        # Encoder
        x_encoded = self.encoder.encoder(torch.cat([cls_token, patches_masked], dim=1))
        
        # Decoder (avec mask tokens)
        x_decoded = self.decoder(x_encoded, ids_restore)
        
        return x_decoded, mask
```

---

## 7. Cas d'Usage Pratiques

### 7.1 Transfer Learning avec ViT

**Scénario typique :** Dataset médical avec 10k images

```python
import timm  # PyTorch Image Models library

# Charger ViT pré-entraîné
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=5  # Notre nombre de classes
)

# Option 1 : Fine-tune complet
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Option 2 : Freeze encoder, train head seulement
for param in model.parameters():
    param.requires_grad = False
model.head.weight.requires_grad = True
model.head.bias.requires_grad = True

# Option 3 : Unfreeze progressivement (best practice)
# Epoch 1-5 : head only
# Epoch 6-10 : + derniers 4 blocs
# Epoch 11+ : tout le modèle

def unfreeze_layers(model, num_blocks):
    # Freeze tout
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze head
    for param in model.head.parameters():
        param.requires_grad = True
    
    # Unfreeze derniers blocs
    for block in model.blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
```

### 7.2 Visualisation des Attention Maps

**Comprendre ce que le modèle "regarde" :**

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_attention_map(model, image, layer_idx=-1, head_idx=0):
    """
    Visualiser les attention maps d'une couche spécifique
    """
    model.eval()
    
    # Forward pass avec hook pour capturer l'attention
    attention_maps = []
    
    def hook_fn(module, input, output):
        # output[1] contient les attention weights si return_attention=True
        attention_maps.append(output[1])
    
    # Attacher le hook
    hook = model.blocks[layer_idx].attn.register_forward_hook(hook_fn)
    
    # Forward
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    hook.remove()
    
    # Récupérer l'attention du CLS token
    attn = attention_maps[0][0, head_idx, 0, 1:]  # (N,)
    
    # Reshaper en grille
    size = int(attn.shape[0] ** 0.5)
    attn = attn.reshape(size, size).cpu().numpy()
    
    # Visualiser
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image originale
    img_np = image.permute(1, 2, 0).cpu().numpy()
    ax1.imshow(img_np)
    ax1.set_title('Image Originale')
    ax1.axis('off')
    
    # Attention map
    ax2.imshow(img_np)
    ax2.imshow(attn, alpha=0.6, cmap='jet')
    ax2.set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
image = torch.randn(3, 224, 224)
visualize_attention_map(model, image, layer_idx=11, head_idx=0)
```

### 7.3 ViT pour d'autres tâches

**Segmentation sémantique (SETR) :**

```python
class ViTSegmentation(nn.Module):
    def __init__(self, vit_backbone, num_classes=21):
        super().__init__()
        self.backbone = vit_backbone
        
        # Decoder pour upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extraire features de toutes les patches
        features = self.backbone(x)  # (B, N+1, D)
        
        # Retirer CLS token, reshaper en 2D
        patch_features = features[:, 1:, :]  # (B, N, D)
        size = int((H // 16))
        patch_features = patch_features.transpose(1, 2).reshape(B, -1, size, size)
        
        # Decoder
        output = self.decoder(patch_features)  # (B, num_classes, H, W)
        
        return output
```

**Détection d'objets (DETR avec ViT backbone) :**

Remplacer ResNet par ViT dans DETR (Detection Transformer).

---

## 8. Limitations et Futures Directions

### 8.1 Limitations Actuelles

1. **Coût computationnel** :
   - $$O(N^2)$$ rend difficile l'utilisation sur haute résolution
   - Solutions : Swin Transformer, Linformer, Performer

2. **Besoin de grandes quantités de données** :
   - ViT-Base nécessite 14M+ images pour surpasser ResNet
   - Solutions : DeiT, MAE, self-supervised learning

3. **Interprétabilité limitée** :
   - Attention maps ≠ explication causale
   - Difficile de comprendre pourquoi une décision

4. **Sensibilité aux hyperparamètres** :
   - Choix du patch size critique
   - Learning rate, warmup schedule

### 8.2 Futures Directions (Recherche Active)

**1. Efficient Transformers**
   - Reduce attention complexity
   - Approximate attention mechanisms
   - Dynamic patch selection

**2. Self-Supervised Learning**
   - MAE, DINO, MoCo v3
   - Apprendre sans labels

**3. Multimodal Vision-Language**
   - CLIP, ALIGN, BLIP
   - ViT comme vision encoder

**4. Video Understanding**
   - TimeSformer, ViViT
   - Temporal attention

---

## 9. Implémentation Complète : Projet Pratique

### 9.1 Dataset : CIFAR-10

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT attend 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
```

### 9.2 Training Loop

```python
import torch.nn.functional as F
from tqdm import tqdm

# Modèle
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=10,  # CIFAR-10
    embed_dim=384,   # Plus petit que ViT-Base pour CIFAR
    depth=6,
    num_heads=6,
    mlp_ratio=4.0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training function
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(loader), 100. * correct / total

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

# Training loop
num_epochs = 100
best_acc = 0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    
    scheduler.step()
    
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
    print(f'Test Acc: {test_acc:.2f}%')
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_vit_cifar10.pth')
        print(f'Best model saved! Acc: {best_acc:.2f}%')
```

### 9.3 Résultats Attendus

| Modèle | Test Accuracy CIFAR-10 |
|--------|------------------------|
| ResNet-18 | ~95% |
| ResNet-50 | ~96% |
| ViT-Tiny (scratch) | ~85% |
| ViT-Base (pretrained) | ~98%+ |

**Pourquoi ViT scratch < ResNet ?**

CIFAR-10 = 50k images → pas assez pour ViT qui nécessite millions d'images.

**Solution : Transfer learning avec pre-trained ViT**

---

## 10. Ressources et Références

### 10.1 Papers Fondamentaux

1. **ViT Original** : *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
   - Dosovitskiy et al., ICLR 2021
   - arXiv:2010.11929

2. **DeiT** : *Training data-efficient image transformers*
   - Touvron et al., ICML 2021
   - arXiv:2012.12877

3. **Swin Transformer** : *Hierarchical Vision Transformer using Shifted Windows*
   - Liu et al., ICCV 2021
   - arXiv:2103.14030

4. **MAE** : *Masked Autoencoders Are Scalable Vision Learners*
   - He et al., CVPR 2022
   - arXiv:2111.06377

### 10.2 Implémentations

- **Official JAX** : https://github.com/google-research/vision_transformer
- **PyTorch (timm)** : https://github.com/huggingface/pytorch-image-models
- **Hugging Face** : https://huggingface.co/docs/transformers/model_doc/vit

### 10.3 Tutoriels Recommandés

1. The Illustrated Transformer (Jay Alammar)
2. Vision Transformers Explained (Yannic Kilcher)
3. Hugging Face ViT Documentation

---

## Conclusion

**Points clés à retenir :**

1. **Architecture** : ViT traite les images comme des séquences de patches, applique un Transformer encoder standard

2. **Self-Attention** : Mécanisme central permettant de capturer des dépendances globales dès la première couche

3. **Data Efficiency** : ViT nécessite beaucoup de données (millions) pour surpasser les CNNs, sinon utiliser pre-training

4. **Scalabilité** : Contrairement aux CNNs qui saturent, ViT continue d'améliorer avec plus de données et de compute

5. **Trade-offs** : 
   - ViT : flexible, scalable mais data-hungry
   - CNN : efficient, data-efficient mais moins scalable

**Quand utiliser ViT ?**

✅ Large dataset disponible (>1M images)  
✅ Accès à modèles pre-trained  
✅ Besoin de performances SOTA  
✅ Tâches nécessitant contexte global  

❌ Petit dataset (<100k images) sans pre-training  
❌ Contraintes de latence strictes  
❌ Ressources computationnelles limitées  

**L'avenir** : Hybrid models combinant les forces des CNNs et Transformers (ex: ConvNext, CoAtNet) représentent une direction prometteuse.
