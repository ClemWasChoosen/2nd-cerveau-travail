# BERT et Variantes Françaises : Architecture et Fondamentaux Théoriques

> **Résumé en une phrase** : Guide exhaustif sur l'architecture BERT (Bidirectional Encoder Representations from Transformers) et ses variantes françaises (CamemBERT, CamemBERTa, FlauBERT), couvrant les mécanismes d'attention, les stratégies de pré-entraînement et les hyperparamètres architecturaux.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-04-28 |
| **Dernière mise à jour** | 2026-04-28 |
| **Domaine** | Deep Learning / NLP |
| **Niveau** | Intermédiaire à Avancé |
| **Durée de lecture** | ~45 minutes |
| **Fichier** | `bert_architecture_variantes_francais.md` |
| **Emplacement** | `/02_deep_learning/01_architectures/` |
| **Tags** | `#bert` `#transformer` `#nlp` `#attention` `#camembert` `#french-nlp` `#pre-training` |

### Prérequis

- [x] Architecture Transformer (encodeur-décodeur, self-attention multi-têtes)
- [x] Tokenization (BPE, WordPiece, SentencePiece)
- [x] Embeddings (word embeddings, positional encoding)
- [x] Concepts de transfer learning et fine-tuning
- [x] Backpropagation et optimisation (Adam, learning rate scheduling)

### Cours connexes (Liens Zettelkasten)

- **Architecture similaire** : [[vit.md]] - Vision Transformer utilise le même encodeur Transformer que BERT
- **Complémentaires** : [[03_llm/distillation_llm.md]] - DistilBERT est une version distillée
- **Suite recommandée** : [[bert_fine_tuning_strategies.md]] (à créer) - Stratégies avancées de fine-tuning

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Ce cours constitue un **aide-mémoire complet** sur l'architecture BERT et ses variantes françaises. Vous y trouverez les formulations mathématiques détaillées, les choix architecturaux, les comparaisons entre variantes, et les hyperparamètres modulables. L'objectif est de répondre à la majorité des questions techniques que vous pourriez vous poser lors de l'utilisation ou de la modification de ces modèles.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Expliquer les mécanismes d'attention bidirectionnelle et les stratégies de pré-entraînement MLM/NSP
2. **Analyser** : Comparer les différences architecturales entre BERT, RoBERTa, CamemBERT et CamemBERTa
3. **Évaluer** : Choisir la variante appropriée selon votre corpus, langue et tâche
4. **Créer** : Modifier les hyperparamètres architecturaux pour adapter BERT à vos besoins spécifiques

---

## 🔍 Contexte et Motivation

### Pourquoi BERT est-il révolutionnaire ?

**Contexte historique (pré-2018)** :
- Les modèles de langage étaient **unidirectionnels** (GPT-1, ELMo left-to-right)
- Le transfer learning en NLP était limité (word2vec, GloVe : embeddings statiques)
- Les tâches NLP nécessitaient des architectures spécifiques par tâche

**L'innovation BERT (2018)** :
BERT a introduit le **pré-entraînement bidirectionnel profond** permettant de capturer le contexte à gauche ET à droite simultanément via le Masked Language Modeling (MLM).

**Pourquoi "bidirectionnel" change tout** :
- **Modèle unidirectionnel** : "La banque de la rivière" → "banque" prédit seulement par "La"
- **BERT bidirectionnel** : "La [MASK] de la rivière" → "banque" prédit par "La", "de", "la", "rivière"

Cela permet de capturer des représentations contextuelles riches dès le pré-entraînement.

### Quel problème résout-il ?

**Problème 1 : Représentations contextuelles limitées**
- **Avant** : Word embeddings statiques (word2vec) → même vecteur pour "banque" (financière) et "banque" (rivière)
- **Avec BERT** : Embeddings contextualisés → représentations différentes selon le contexte

**Problème 2 : Pré-entraînement unidirectionnel**
- **GPT-1** : Pré-entraîné left-to-right → ne peut pas utiliser le contexte futur
- **BERT** : Pré-entraîné bidirectionnellement → utilise tout le contexte

**Problème 3 : Transfer learning inefficace**
- **Avant** : Architectures task-specific, pré-entraînement limité
- **Avec BERT** : Architecture universelle, fine-tuning pour toute tâche NLP

### Applications dans le monde réel

1. **Classification de textes** : Analyse de sentiment, détection de spam, catégorisation de documents
2. **Named Entity Recognition (NER)** : Extraction d'entités (personnes, lieux, organisations)
3. **Question Answering** : SQuAD, Natural Questions → trouver des réponses dans un contexte
4. **Similarité sémantique** : Paraphrase detection, semantic textual similarity
5. **Analyse juridique/médicale en français** : CamemBERT pour domaines spécialisés

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Cette section construit les bases architecturales. Nous commençons par l'architecture globale, puis détaillons chaque composant (embeddings, attention, pré-entraînement).

### 1. Architecture Globale de BERT

#### 1.1 Composants Principaux

BERT est basé sur l'**encodeur Transformer** uniquement (pas de décodeur, contrairement à GPT).

**Architecture en couches** :

```python
Input Text
↓
Tokenization (WordPiece)
↓
Token Embeddings + Segment Embeddings + Position Embeddings
↓
Transformer Encoder Layer 1
├─ Multi-Head Self-Attention
├─ Add & Norm (Layer Normalization)
├─ Feed-Forward Network
└─ Add & Norm
↓
Transformer Encoder Layer 2
↓
...
↓
Transformer Encoder Layer L (12 ou 24 couches)
↓
Contextualized Embeddings pour chaque token
↓
[CLS] token → Classification tasks
[token] embeddings → Token-level tasks (NER, POS tagging)
```

#### 1.2 Variantes de BERT par Taille

| Modèle | Couches ($$L$$) | Hidden Size ($$H$$) | Attention Heads ($$A$$) | Paramètres | Complexité Mémoire |
|--------|-----------------|---------------------|-------------------------|------------|-------------------|
| **BERT-Base** | 12 | 768 | 12 | 110M | ~2 GB (inference) |
| **BERT-Large** | 24 | 1024 | 16 | 340M | ~6 GB (inference) |

**Pourquoi ces configurations ?**
- $$L = 12$$ (Base) : Compromis performance/coût, suffisant pour la plupart des tâches
- $$H = 768$$ : Divisible par $$A = 12$$ (chaque tête = 64 dimensions)
- $$A = 12$$ : Capture différents types de relations syntaxiques/sémantiques

**Relation mathématique** :
$$\text{Head dimension} = \frac{H}{A} = \frac{768}{12} = 64$$

Cette dimension de 64 par tête est un standard empirique (aussi utilisé dans le Transformer original).

---

### 2. Embeddings : Représentation de l'Input

BERT combine **trois types d'embeddings** pour chaque token.

#### 2.1 Token Embeddings

**Tokenization avec WordPiece** :
BERT utilise WordPiece (vocabulaire de ~30K tokens pour BERT anglais).

**Exemple** :
- Input : "playing"
- Tokens : ["play", "##ing"]

**Embedding matriciel** :
$$E_{\text{token}} \in \mathbb{R}^{V \times H}$$

Où :
- $$V$$ = taille du vocabulaire (~30 000)
- $$H$$ = dimension cachée (768 pour BERT-Base)

**Pourquoi WordPiece ?**
- **Vocabulaire fixe** : Gère les mots rares via sous-unités (out-of-vocabulary handling)
- **Compression** : Réduit la taille du vocabulaire vs mots entiers
- **Généralisation** : Partage de représentations entre mots morphologiquement similaires

#### 2.2 Segment Embeddings

Utilisés pour distinguer deux phrases dans les tâches de paires de phrases (ex: NLI, QA).

**Représentation** :
$$E_{\text{segment}} \in \mathbb{R}^{2 \times H}$$

Deux embeddings apprenables :
- $$E_A$$ pour la phrase A
- $$E_B$$ pour la phrase B

**Exemple** :
```
Input: [CLS] Quelle est la capitale ? [SEP] Paris est la capitale. [SEP]
Segments: [  A  ] [       A       ] [ A ] [    B    ] [  B ] [  B  ] [ B ]
```

#### 2.3 Position Embeddings

Contrairement au Transformer original (encodage sinusoïdal), BERT utilise des **position embeddings apprenables**.

**Représentation** :
$$E_{\text{position}} \in \mathbb{R}^{N_{\text{max}} \times H}$$

Où :
- $$N_{\text{max}} = 512$$ (longueur de séquence maximale)
- Chaque position a un vecteur appris

**Formulation finale** :
$$E_{\text{input}}^{(i)} = E_{\text{token}}^{(i)} + E_{\text{segment}}^{(i)} + E_{\text{position}}^{(i)}$$

**Pourquoi apprenables vs sinusoïdales ?**
- **Flexibilité** : Peuvent s'adapter aux patterns spécifiques du pré-entraînement
- **Limitation** : Fixées à 512 tokens max (extensions possibles via interpolation)
- **Empirique** : Meilleures performances observées dans le paper original

---

### 3. Multi-Head Self-Attention : Le Cœur de BERT

#### 3.1 Self-Attention Scalée (Scaled Dot-Product Attention)

**Formulation mathématique** :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Où :
- $$Q \in \mathbb{R}^{N \times d_k}$$ : Queries (requêtes)
- $$K \in \mathbb{R}^{N \times d_k}$$ : Keys (clés)
- $$V \in \mathbb{R}^{N \times d_v}$$ : Values (valeurs)
- $$N$$ : Longueur de séquence
- $$d_k = d_v = \frac{H}{A}$$ : Dimension par tête (64 pour BERT-Base)
- $$\sqrt{d_k}$$ : Facteur de normalisation pour stabilité numérique

**Intuition** :
1. **Scores d'attention** : $$QK^T$$ calcule la similarité entre chaque paire de tokens
2. **Normalisation** : $$\sqrt{d_k}$$ empêche les gradients de saturer (softmax sur grandes valeurs → gradients ~0)
3. **Softmax** : Convertit scores en probabilités (somme = 1 par ligne)
4. **Pondération** : Multiplie les valeurs $$V$$ par les poids d'attention

**Pourquoi $$\sqrt{d_k}$$ ?**
- Avec $$d_k$$ grand, $$QK^T$$ peut avoir de grandes valeurs
- Softmax sur grandes valeurs → poids concentrés (gradients faibles)
- Division par $$\sqrt{d_k}$$ → variance normalisée ≈ 1

**Dérivation de la variance** :
Si $$Q, K \sim \mathcal{N}(0, 1)$$, alors :
$$\text{Var}(QK^T) = d_k$$
Donc :
$$\text{Var}\left(\frac{QK^T}{\sqrt{d_k}}\right) = 1$$

#### 3.2 Multi-Head Attention

**Pourquoi plusieurs têtes ?**
Une seule tête d'attention capture un seul type de relation. Les têtes multiples permettent de capturer différents aspects :
- **Tête 1** : Relations syntaxiques (sujet-verbe)
- **Tête 2** : Coréférences (pronoms → antécédents)
- **Tête 3** : Relations sémantiques (synonymes, hyperonymes)

**Formulation mathématique** :

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_A) W^O$$

Où chaque tête est :
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Matrices de projection** :
- $$W_i^Q \in \mathbb{R}^{H \times d_k}$$ : Projection pour queries (tête $$i$$)
- $$W_i^K \in \mathbb{R}^{H \times d_k}$$ : Projection pour keys
- $$W_i^V \in \mathbb{R}^{H \times d_v}$$ : Projection pour values
- $$W^O \in \mathbb{R}^{Ad_v \times H}$$ : Projection finale de sortie

**Dimensions** :
- Input : $$\mathbb{R}^{N \times H}$$
- Chaque tête : $$\mathbb{R}^{N \times d_k}$$ ($$d_k = H/A = 64$$)
- Concaténation : $$\mathbb{R}^{N \times (A \cdot d_k)} = \mathbb{R}^{N \times H}$$
- Output : $$\mathbb{R}^{N \times H}$$ (après $$W^O$$)

**Complexité computationnelle** :
$$O(N^2 \cdot H)$$

Où :
- $$N^2$$ : Calcul de l'attention entre toutes les paires de tokens
- $$H$$ : Dimension des embeddings

**Pourquoi c'est un problème pour les longues séquences ?**
Pour $$N = 512$$ : $$512^2 = 262,144$$ opérations d'attention par couche.

#### 3.3 Bidirectionnalité dans BERT

**Différence cruciale avec GPT** :

| Modèle | Type d'attention | Masquage | Contexte utilisé |
|--------|------------------|----------|------------------|
| **GPT** | Causal (autoregressive) | Masque triangulaire | Tokens précédents uniquement |
| **BERT** | Bidirectional | Aucun masque | Tous les tokens de la séquence |

**Masque causal (GPT)** :
$$
\text{Mask} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

**BERT (pas de masque structurel)** :
$$
\text{Mask} = \begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

Chaque token peut "voir" tous les autres tokens (bidirectionnel).

---

### 4. Feed-Forward Network (FFN)

Après l'attention, chaque position passe par un **réseau feed-forward** identique.

**Formulation** :
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

Où :
- $$W_1 \in \mathbb{R}^{H \times 4H}$$ : Projection vers dimension intermédiaire
- $$W_2 \in \mathbb{R}^{4H \times H}$$ : Projection retour vers dimension originale
- $$b_1, b_2$$ : Biais
- $$4H = 3072$$ pour BERT-Base

**Pourquoi $$4H$$ (dimension intermédiaire) ?**
- **Capacité de représentation** : Augmente la non-linéarité
- **Empirique** : Ratio 4:1 trouvé optimal dans le Transformer original
- **Trade-off** : Plus grand = plus de paramètres mais meilleure expressivité

**Fonction d'activation GELU** :
$$\text{GELU}(x) = x \cdot \Phi(x)$$

Où $$\Phi(x)$$ est la CDF de la distribution normale standard.

**Approximation** :
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**Pourquoi GELU plutôt que ReLU ?**
- **Stochastique** : GELU est une version lissée de ReLU
- **Gradients non-nuls** : Contrairement à ReLU, GELU a des gradients partout
- **Performance empirique** : Meilleures performances en NLP (observé dans BERT, GPT-2)

---

### 5. Layer Normalization et Connexions Résiduelles

#### 5.1 Layer Normalization

**Formulation** :
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

Où :
- $$\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$$ : Moyenne sur la dimension cachée
- $$\sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2}$$ : Écart-type
- $$\gamma, \beta \in \mathbb{R}^H$$ : Paramètres apprenables (scale & shift)
- $$\odot$$ : Produit élément par élément

**Pourquoi Layer Norm plutôt que Batch Norm ?**

| Critère | Batch Normalization | Layer Normalization |
|---------|---------------------|---------------------|
| **Normalisation** | Sur le batch | Sur la dimension cachée |
| **Indépendance** | Dépend du batch | Indépendant du batch |
| **Séquences variables** | Problématique | Pas de problème |
| **Inference** | Nécessite statistiques globales | Identique à l'entraînement |

Pour les Transformers, Layer Norm est préféré car :
- **Indépendance** : Chaque exemple normalisé indépendamment
- **Stabilité** : Pas de dépendance aux statistiques du batch

#### 5.2 Connexions Résiduelles (Residual Connections)

**Formulation dans chaque sous-couche** :
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Où $$\text{Sublayer}(x)$$ est soit :
- Multi-Head Attention
- Feed-Forward Network

**Pourquoi les connexions résiduelles ?**
1. **Gradient flow** : Évite la disparition des gradients dans les réseaux profonds
2. **Identité** : Permet au réseau d'apprendre des transformations incrémentales
3. **Optimisation** : Facilite l'entraînement de réseaux très profonds (24 couches pour BERT-Large)

**Formulation complète d'une couche Transformer** :

$$
\begin{aligned}
z &= \text{LayerNorm}(x + \text{MultiHeadAttention}(x)) \\
\text{output} &= \text{LayerNorm}(z + \text{FFN}(z))
\end{aligned}
$$

---

### 6. Stratégies de Pré-Entraînement

BERT utilise deux tâches de pré-entraînement auto-supervisées.

#### 6.1 Masked Language Modeling (MLM)

**Principe** :
Masquer aléatoirement 15% des tokens et prédire les tokens masqués.

**Exemple** :
- **Input** : "Paris est la [MASK] de la France"
- **Target** : "capitale"

**Stratégie de masquage détaillée** :
Sur les 15% de tokens sélectionnés :
- **80%** : Remplacés par `[MASK]`
- **10%** : Remplacés par un token aléatoire
- **10%** : Inchangés

**Pourquoi cette stratégie asymétrique ?**

1. **80% [MASK]** : Force le modèle à prédire basé sur le contexte
2. **10% random** : Évite que le modèle ne se fie uniquement au token `[MASK]`
3. **10% unchanged** : Force le modèle à utiliser le contexte même quand le token est correct

**Fonction de perte** :
$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

Où :
- $$\mathcal{M}$$ : Ensemble des positions masquées
- $$x_{\backslash \mathcal{M}}$$ : Tokens non masqués

**Pourquoi 15% ?**
- **Trop peu (<10%)** : Signal d'apprentissage insuffisant
- **Trop (>20%)** : Contexte insuffisant pour prédire
- **15%** : Compromis empirique optimal

#### 6.2 Next Sentence Prediction (NSP)

**Principe** :
Prédire si la phrase B suit naturellement la phrase A dans le corpus.

**Format** :
```
Input: [CLS] Phrase A [SEP] Phrase B [SEP]
Label: IsNext (1) ou NotNext (0)
```

**Exemple** :
- **IsNext** : A = "Il pleut dehors." / B = "Je vais prendre un parapluie." → Label = 1
- **NotNext** : A = "Il pleut dehors." / B = "La Lune est un satellite." → Label = 0

**Construction des paires** :
- **50%** : Phrases consécutives réelles (label = 1)
- **50%** : Phrases aléatoires du corpus (label = 0)

**Fonction de perte** :
$$\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} | \text{[CLS]})$$

Le vecteur `[CLS]` est utilisé pour la classification binaire.

**Perte totale de pré-entraînement** :
$$\mathcal{L}_{\text{pre-train}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

**Critique de NSP** (RoBERTa, 2019) :
NSP est potentiellement **trop facile** (distinction topic vs cohérence). RoBERTa supprime NSP et améliore les performances.

---

### 7. Tokens Spéciaux

BERT introduit des tokens spéciaux avec des rôles spécifiques.

| Token | Rôle | Utilisation |
|-------|------|-------------|
| **[CLS]** | Classification | Premier token, représentation de la séquence entière pour classification |
| **[SEP]** | Séparateur | Sépare les phrases A et B, marque la fin de séquence |
| **[MASK]** | Masquage | Token de remplacement pour MLM |
| **[PAD]** | Padding | Complétion des séquences courtes à longueur fixe |
| **[UNK]** | Inconnu | Tokens hors vocabulaire |

**Pourquoi [CLS] pour la classification ?**
- **Agrégation globale** : [CLS] n'a pas de signification lexicale, il agrège l'information de toute la séquence via l'attention
- **Position fixe** : Toujours en première position, facilite l'extraction

---

## 🔬 Variantes de BERT

### 1. RoBERTa (Robustly Optimized BERT Approach)

**Paper** : *"RoBERTa: A Robustly Optimized BERT Pretraining Approach"* (Liu et al., Facebook AI, 2019) - [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)

#### 1.1 Modifications par rapport à BERT

| Aspect | BERT | RoBERTa |
|--------|------|---------|
| **NSP** | Utilisé | ✗ Supprimé |
| **Masquage** | Statique (même masque par epoch) | Dynamique (masque change chaque epoch) |
| **Batch size** | 256 | 8K (32x plus grand) |
| **Données** | 16GB (BookCorpus + Wikipedia) | 160GB (+ CC-News, OpenWebText, Stories) |
| **BPE** | WordPiece (30K) | Byte-level BPE (50K) |
| **Séquences** | 512 tokens max | Séquences complètes (pas de padding NSP) |

#### 1.2 Pourquoi ces changements améliorent les performances ?

**Suppression de NSP** :
- NSP apprend principalement la distinction de topics (facile)
- Entraînement sur phrases complètes/documents > paires artificielles

**Masquage dynamique** :
- **BERT** : Masque généré une fois, réutilisé chaque epoch → modèle voit toujours les mêmes exemples masqués
- **RoBERTa** : Nouveau masque à chaque passage → ×4 diversité si 4 epochs (40 epochs = 10× diversité)

**Byte-level BPE** :
- **Avantage** : Gère mieux les mots rares et les langues avec caractères spéciaux
- **Vocabulaire plus grand** : 50K vs 30K → meilleure couverture

**Plus de données** :
- BERT : 16GB (13M de paramètres appris sur peu de données)
- RoBERTa : 160GB (10× plus) → meilleure généralisation

#### 1.3 Résultats

RoBERTa surpasse BERT sur tous les benchmarks (GLUE, SQuAD, RACE) avec **la même architecture**, simplement grâce à l'optimisation du pré-entraînement.

**Leçon clé** : Le pré-entraînement est aussi important que l'architecture.

---

### 2. CamemBERT : BERT pour le Français

**Paper** : *"CamemBERT: a Tasty French Language Model"* (Martin et al., Inria/Facebook, 2020) - [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)

#### 2.1 Motivation

**Pourquoi pas BERT multilingue (mBERT) ?**
- mBERT est entraîné sur 104 langues → **dilution des capacités** par langue
- Vocabulaire partagé → sous-représentation du français
- Corpus français limité dans mBERT

**Objectif** : BERT monolingue français optimisé.

#### 2.2 Architecture et Spécificités

| Aspect | BERT (anglais) | CamemBERT |
|--------|----------------|-----------|
| **Architecture de base** | RoBERTa (optimisé) | RoBERTa (optimisé) |
| **Tokenization** | WordPiece | **SentencePiece** (BPE) |
| **Vocabulaire** | 30K | 32K |
| **Corpus** | BookCorpus + Wikipedia (16GB) | **OSCAR** (138GB français) |
| **Lowercase** | Oui (BERT-base-uncased) | Non (case-sensitive) |

#### 2.3 Tokenization avec SentencePiece

**Pourquoi SentencePiece ?**
- **Indépendant de la langue** : Pas besoin de pré-tokenization (espaces, ponctuation)
- **Gestion des espaces** : Traite les espaces comme des caractères spéciaux (`▁`)
- **Sous-mots optimaux** : BPE apprend les sous-unités directement du corpus

**Exemple de tokenization** :

```python
# Exemple conceptuel (pas d'exécution nécessaire)
from transformers import CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

text = "L'apprentissage automatique est fascinant."
tokens = tokenizer.tokenize(text)
# Résultat : ['▁L', "'", 'apprentissage', '▁automatique', '▁est', '▁fascinant', '.']

# ▁ représente un espace initial
```

#### 2.4 Corpus OSCAR

**OSCAR** (Open Super-large Crawled Aggregated coRpus) :
- **Taille** : 138GB de texte français (vs 4GB Wikipedia français)
- **Source** : CommonCrawl (web crawl)
- **Qualité** : Filtrage via perplexité et modèle de langue

**Impact** :
- **Diversité** : Variété de domaines (pas uniquement encyclopédique)
- **Taille** : ×34 plus de données que mBERT pour le français

#### 2.5 Configuration de Pré-Entraînement

| Hyperparamètre | Valeur |
|----------------|--------|
| **Couches** | 12 |
| **Hidden size** | 768 |
| **Attention heads** | 12 |
| **Paramètres** | 110M |
| **Batch size** | 2048 |
| **Steps** | 100K |
| **Learning rate** | 1e-4 (warmup + decay) |
| **MLM masking** | 15% (dynamique) |
| **NSP** | Non (suit RoBERTa) |

#### 2.6 Performances

**Benchmarks français** :
- **NER** : FTB, WikiNER
- **POS Tagging** : FTB, GSD
- **Dependency Parsing** : FTB
- **Classification** : CLS-FR, PAWS-X-FR

**Résultats** : CamemBERT > mBERT > FlauBERT (selon les tâches)

---

### 3. CamemBERTa : CamemBERT + RoBERTa Optimisations

**Observation** : CamemBERTa n'est pas un modèle officiel distinct, mais désigne généralement :
1. **CamemBERT v2** : Versions améliorées avec plus de données
2. **Approche RoBERTa pour français** : Combinaison des optimisations RoBERTa appliquées au français

**Si vous voyez "CamemBERTa" dans un contexte** :
- Vérifier s'il s'agit d'un fine-tuning spécifique de CamemBERT
- Ou d'une implémentation RoBERTa entraînée sur corpus français

**Clarification** :
Le terme n'est pas standardisé. **CamemBERT** suit déjà l'approche RoBERTa (sans NSP, masquage dynamique).

---

### 4. FlauBERT : Alternative Française

**Paper** : *"FlauBERT: Unsupervised Language Model Pre-training for French"* (Le et al., CNRS, 2020) - [arXiv:1912.05372](https://arxiv.org/abs/1912.05372)

#### 4.1 Différences avec CamemBERT

| Aspect | CamemBERT | FlauBERT |
|--------|-----------|----------|
| **Base** | RoBERTa (sans NSP) | BERT original (avec NSP) |
| **Corpus** | OSCAR (138GB) | Divers corpus (71GB) |
| **Tokenization** | SentencePiece BPE | SentencePiece BPE |
| **Vocabulaire** | 32K | 50K |
| **Tailles disponibles** | Base (110M) | Small (54M), Base (138M), Large (373M) |

#### 4.2 Variantes de FlauBERT

**FlauBERT-small** :
- 6 couches, 512 hidden size, 8 heads
- 54M paramètres
- Utile pour ressources limitées

**FlauBERT-base** :
- 12 couches, 768 hidden size, 12 heads
- 138M paramètres

**FlauBERT-large** :
- 24 couches, 1024 hidden size, 16 heads
- 373M paramètres
- Meilleure performance mais coût élevé

#### 4.3 Quand utiliser FlauBERT vs CamemBERT ?

**CamemBERT** (recommandé par défaut) :
- ✅ Optimisations RoBERTa (meilleur pré-entraînement)
- ✅ Plus de données (OSCAR)
- ✅ Performances généralement supérieures

**FlauBERT** :
- ✅ Variante Small pour ressources limitées
- ✅ Variante Large pour tâches exigeantes
- ✅ NSP si votre tâche nécessite compréhension de cohérence inter-phrases

---

## ⚖️ Comparaisons et Choix de Design

### 1. Tableau Comparatif Global

| Modèle | Langue | Architecture | NSP | Corpus | Taille Corpus | Vocab | Tokenizer | Paramètres |
|--------|--------|--------------|-----|--------|---------------|-------|-----------|------------|
| **BERT-base** | EN | BERT | ✓ | BookCorpus + Wiki | 16GB | 30K | WordPiece | 110M |
| **RoBERTa-base** | EN | RoBERTa | ✗ | +CC-News, etc. | 160GB | 50K | BPE | 125M |
| **mBERT** | 104 langues | BERT | ✓ | Wikipedia | ~GB/langue | 110K | WordPiece | 110M |
| **CamemBERT** | FR | RoBERTa | ✗ | OSCAR | 138GB | 32K | SentencePiece | 110M |
| **FlauBERT-base** | FR | BERT | ✓ | Divers | 71GB | 50K | SentencePiece | 138M |

### 2. Arbre de Décision pour Choisir un Modèle

```
Tâche NLP française ?
├─ OUI
│   └─ Ressources limitées (GPU faible) ?
│       ├─ OUI → FlauBERT-small (54M paramètres)
│       └─ NON
│           └─ Performance maximale ?
│               ├─ OUI → CamemBERT-large ou FlauBERT-large
│               └─ NON → CamemBERT-base (recommandé par défaut)
│
└─ NON (autre langue)
└─ Langue supportée par modèle monolingue ?
├─ OUI → Utiliser modèle monolingue (meilleur)
└─ NON → mBERT ou XLM-RoBERTa (multilingue)
```

### 3. Comparaison Hyperparamètres Modulables

#### 3.1 Nombre de Couches ($$L$$)

**Impact** :
- **Plus de couches** : Représentations plus abstraites, meilleure performance
- **Moins de couches** : Plus rapide, moins de mémoire

**Recommandations** :

| Tâche | Couches recommandées | Raison |
|-------|----------------------|--------|
| Classification simple | 6-12 | Suffisant pour capturer sémantique |
| NER, POS tagging | 12 | Nécessite syntaxe + sémantique |
| QA, NLI complexe | 12-24 | Raisonnement multi-sauts |
| Ressources limitées | 4-6 | DistilBERT (6 couches) = 97% BERT-base |

**Formule empirique** :
$$\text{Performance} \propto \log(L)$$
Au-delà de 24 couches, gains marginaux décroissants.

#### 3.2 Hidden Size ($$H$$)

**Impact** :
- **Plus grand** : Capacité de représentation accrue
- **Plus petit** : Moins de paramètres, plus rapide

**Contrainte** :
$$H$$ doit être divisible par $$A$$ (nombre de têtes) :
$$d_k = \frac{H}{A}$$

**Configurations typiques** :

| Taille | $$H$$ | $$A$$ | $$d_k$$ | Use case |
|--------|-------|-------|---------|----------|
| Tiny | 128 | 2 | 64 | Mobile, edge devices |
| Small | 512 | 8 | 64 | Ressources limitées |
| Base | 768 | 12 | 64 | Standard |
| Large | 1024 | 16 | 64 | Performance maximale |

**Pourquoi $$d_k = 64$$ est standard ?**
- **Empirique** : Trouvé optimal dans le Transformer original
- **Trade-off** : Assez grand pour expressivité, assez petit pour stabilité numérique

#### 3.3 Nombre de Têtes d'Attention ($$A$$)

**Rôles des têtes multiples** :
Chaque tête capture différents types de relations.

**Analyse empirique** (Voita et al., 2019) :
- **Têtes positionnelles** : Capturent relations locales (adjacence)
- **Têtes syntaxiques** : Relations grammaticales (sujet-verbe)
- **Têtes rares** : Tokens spéciaux, ponctuation

**Recommandations** :

| $$H$$ | $$A$$ recommandé | $$d_k$$ | Note |
|-------|------------------|---------|------|
| 256 | 4 | 64 | Minimum viable |
| 512 | 8 | 64 | Small model |
| 768 | 12 | 64 | BERT-base (standard) |
| 1024 | 16 | 64 | BERT-large |

**Pourquoi pas plus de têtes ?**
- Au-delà de 16 : redondance entre têtes
- Coût computationnel accru sans gain significatif

#### 3.4 Dimension Intermédiaire FFN ($$H_{\text{FFN}}$$)

**Standard** : $$H_{\text{FFN}} = 4H$$

**Impact** :

| $$H_{\text{FFN}}$$ | Paramètres FFN | Performance | Coût |
|-------------------|----------------|-------------|------|
| $$2H$$ | $$2H^2$$ | Réduite | Faible |
| $$4H$$ | $$8H^2$$ (standard) | Optimale | Moyen |
| $$8H$$ | $$16H^2$$ | Marginal | Élevé |

**Pourquoi $$4H$$ ?**
- **Capacité** : Assez grande pour non-linéarité complexe
- **Overfitting** : Pas trop grande (évite overfitting sur petits datasets)

#### 3.5 Longueur de Séquence Maximale ($$N_{\text{max}}$$)

**BERT standard** : 512 tokens

**Limitations** :
- **Mémoire** : $$O(N^2)$$ pour l'attention → 512² = 262K opérations
- **Séquences longues** : Documents, livres nécessitent plus

**Alternatives pour séquences longues** :

| Modèle | $$N_{\text{max}}$$ | Technique | Complexité |
|--------|-------------------|-----------|------------|
| BERT | 512 | Standard | $$O(N^2)$$ |
| Longformer | 4096 | Attention sparse | $$O(N \cdot k)$$ |
| BigBird | 4096 | Random + global attention | $$O(N \cdot k)$$ |
| Reformer | 16384 | LSH attention | $$O(N \log N)$$ |

**Pour modifier $$N_{\text{max}}$$ dans BERT** :
1. **Étendre position embeddings** : Interpoler ou apprendre nouvelles positions
2. **Réentraîner** : Fine-tuner sur séquences plus longues
3. **Coût mémoire** : $$512 \to 1024$$ = ×4 mémoire d'attention

#### 3.6 Dropout

**Utilisation dans BERT** :
- Après embeddings : 0.1
- Après attention : 0.1
- Après FFN : 0.1

**Impact** :

| Dropout rate | Overfitting | Performance | Quand utiliser |
|--------------|-------------|-------------|----------------|
| 0.0 | Élevé | Basse (overfit) | Très grands datasets |
| 0.1 | Optimal | Meilleure | Standard (BERT) |
| 0.2-0.3 | Faible | Réduite (underfit) | Petits datasets |

**Recommandation** :
- **Pré-entraînement** : 0.1 (standard)
- **Fine-tuning petit dataset** : Augmenter à 0.2-0.3
- **Fine-tuning grand dataset** : Garder 0.1

---

## 💡 Compréhension Intuitive

### Analogie : BERT comme un Réseau Social

**Imaginez un réseau social où chaque mot est une personne** :

1. **Self-Attention** = Chaque personne regarde TOUS ses "amis" (autres mots) et décide de l'importance de chacun
2. **Multi-Head Attention** = Chaque personne a 12 "perspectives" différentes (travail, famille, loisirs) pour évaluer ses amis
3. **Bidirectionnel** = Les amitiés sont symétriques (A → B et B → A)
4. **Masked LM** = Jeu du "devine qui" : on cache une personne, et les autres doivent deviner qui c'est basé sur le contexte

**Différence avec GPT (unidirectionnel)** :
- GPT = On ne peut voir que les amis "à gauche" (chronologiquement avant)
- BERT = On voit tous les amis (gauche + droite)

### Questions pour vérifier la compréhension

Avant de continuer, assurez-vous de pouvoir répondre :

1. **Q1** : Pourquoi BERT utilise-t-il $$\sqrt{d_k}$$ dans le calcul de l'attention ?
   - *Réponse attendue* : Pour normaliser la variance des scores et éviter la saturation du softmax, ce qui améliore le gradient flow.

2. **Q2** : Quelle est la différence fondamentale entre BERT et GPT en termes d'architecture ?
   - *Réponse attendue* : BERT utilise un encodeur bidirectionnel (voit tout le contexte), GPT utilise un décodeur unidirectionnel avec masque causal (voit seulement le contexte passé).

3. **Q3** : Pourquoi CamemBERT performe mieux que mBERT sur des tâches françaises ?
   - *Réponse attendue* : Corpus monolingue français plus large (138GB vs ~4GB), vocabulaire optimisé pour le français, et pré-entraînement sans dilution multi-langues.

4. **Q4** : Pourquoi RoBERTa supprime-t-il NSP ?
   - *Réponse attendue* : NSP apprend principalement la distinction de topics (facile), pas la cohérence inter-phrases. Entraîner sur documents complets est plus efficace.

---

## 💻 Implémentation Pratique (Concepts Clés)

> **Principe de modalité** : Code minimal pour illustrer les concepts architecturaux, pas un tutoriel complet.

### 1. Chargement et Inspection de BERT

```python
"""
Titre : Inspection de l'architecture BERT
Objectif : Comprendre la structure interne d'un modèle BERT
"""

from transformers import BertModel, BertConfig, CamembertModel
import torch

# Configuration BERT-base
config = BertConfig(
    vocab_size=30522,           # Taille du vocabulaire
    hidden_size=768,            # H : dimension cachée
    num_hidden_layers=12,       # L : nombre de couches
    num_attention_heads=12,     # A : nombre de têtes
    intermediate_size=3072,     # 4H : dimension FFN
    max_position_embeddings=512,# N_max : longueur séquence max
    hidden_dropout_prob=0.1,    # Dropout
    attention_probs_dropout_prob=0.1
)

# Instancier le modèle
model = BertModel(config)

# Afficher la structure
print(f"Nombre de paramètres : {model.num_parameters():,}")
# Résultat : ~110M paramètres pour BERT-base

# Structure des couches
print("\nStructure d'une couche Transformer :")
print(model.encoder.layer[0])  # Première couche

# Sortie exemple :
# BertLayer(
#   (attention): BertAttention(...)
#   (intermediate): BertIntermediate(...)
#   (output): BertOutput(...)
# )
```

**Explication** :
- `BertConfig` : Spécifie tous les hyperparamètres modulables
- `num_parameters()` : Calcule le total de paramètres (embeddings + couches + pooler)
- `encoder.layer[i]` : Accès à chaque couche Transformer

### 2. Modification des Hyperparamètres

```python
"""
Titre : Créer une variante custom de BERT
Objectif : Modifier l'architecture pour expérimentation
"""

# Exemple : BERT "Small" custom
config_small = BertConfig(
    vocab_size=32000,           # Vocabulaire français (CamemBERT-like)
    hidden_size=512,            # Réduit de 768 → 512
    num_hidden_layers=8,        # Réduit de 12 → 8
    num_attention_heads=8,      # Maintient d_k = 512/8 = 64
    intermediate_size=2048,     # 4H = 4*512
    max_position_embeddings=512
)

model_small = BertModel(config_small)
print(f"Paramètres BERT-Small custom : {model_small.num_parameters():,}")
# ~40M paramètres (vs 110M pour base)

# Exemple : Extension de la longueur de séquence
config_long = BertConfig.from_pretrained("bert-base-uncased")
config_long.max_position_embeddings = 1024  # 512 → 1024

# Note : Nécessite réentraînement ou interpolation des position embeddings
```

**Pourquoi ces modifications ?**
- **Moins de couches/hidden size** : Réduit le coût pour applications avec ressources limitées
- **Extension séquence** : Nécessaire pour traiter des documents longs (papiers, articles)

### 3. Inspection des Embeddings

```python
"""
Titre : Comprendre les trois types d'embeddings
Objectif : Visualiser token, segment et position embeddings
"""

# Charger CamemBERT
model_cam = CamembertModel.from_pretrained("camembert-base")

# Accéder aux embeddings
token_embeddings = model_cam.embeddings.word_embeddings
position_embeddings = model_cam.embeddings.position_embeddings

print(f"Token embeddings shape : {token_embeddings.weight.shape}")
# Résultat : torch.Size([32005, 768]) → 32K vocab × 768 dim

print(f"Position embeddings shape : {position_embeddings.weight.shape}")
# Résultat : torch.Size([514, 768]) → 514 positions × 768 dim
# (514 = 512 + 2 tokens spéciaux)

# Segment embeddings (pour NSP, non utilisé dans CamemBERT)
# BERT classique a : embeddings.token_type_embeddings
# Shape : [2, 768] → 2 segments (A/B)
```

**Observation** :
- CamemBERT a 514 position embeddings (512 + marge)
- Pas de segment embeddings car pas de NSP

### 4. Calcul de Complexité Mémoire

```python
"""
Titre : Estimation de la mémoire GPU requise
Objectif : Prédire les besoins mémoire pour différentes configurations
"""

def estimate_memory(num_layers, hidden_size, num_heads, seq_length, batch_size):
    """
    Estime la mémoire GPU (en GB) pour un modèle BERT-like.
    
    Formules simplifiées :
    - Attention : O(N^2 * H)
    - Paramètres : ~12 * L * H^2
    """
    # Paramètres du modèle
    params_per_layer = 12 * hidden_size**2  # Approximation
    total_params = num_layers * params_per_layer
    param_memory = total_params * 4 / 1e9  # 4 bytes (float32) → GB
    
    # Activations (attention)
    attention_memory = (batch_size * num_layers * seq_length**2 * hidden_size * 4) / 1e9
    
    # Total
    total_memory = param_memory + attention_memory
    
    return {
        "Paramètres (GB)": round(param_memory, 2),
        "Activations (GB)": round(attention_memory, 2),
        "Total (GB)": round(total_memory, 2)
    }

# Exemple : BERT-base
print("BERT-base (L=12, H=768, seq=512, batch=8) :")
print(estimate_memory(12, 768, 12, 512, 8))

# Exemple : BERT-large
print("\nBERT-large (L=24, H=1024, seq=512, batch=8) :")
print(estimate_memory(24, 1024, 16, 512, 8))

# Exemple : Séquence longue
print("\nBERT-base avec séquence 2048 (batch=8) :")
print(estimate_memory(12, 768, 12, 2048, 8))
```

**Résultats attendus** :
- BERT-base (512 tokens) : ~3-4 GB
- BERT-large : ~8-10 GB
- Séquence 2048 : ×16 mémoire d'attention ($$N^2$$)

---

## 🔬 Détails Mathématiques Avancés

### 1. Dérivation de l'Attention

**Objectif** : Comprendre pourquoi la formulation softmax(QK^T/√d_k)V est optimale.

**Problème** : Calculer une pondération de chaque token basée sur sa pertinence.

**Étape 1 : Score de similarité**
$$s_{ij} = q_i \cdot k_j = \sum_{d=1}^{d_k} q_i^{(d)} k_j^{(d)}$$

**Étape 2 : Normalisation par température**
Diviser par $$\sqrt{d_k}$$ agit comme une "température" :
$$s_{ij}' = \frac{s_{ij}}{\sqrt{d_k}}$$

**Justification statistique** :
Si $$q_i, k_j \sim \mathcal{N}(0, 1)$$ indépendants :
$$\mathbb{E}[s_{ij}] = 0$$
$$\text{Var}(s_{ij}) = d_k$$

Donc :
$$\text{Var}(s_{ij}') = \frac{d_k}{d_k} = 1$$

**Étape 3 : Softmax pour pondération probabiliste**
$$\alpha_{ij} = \frac{\exp(s_{ij}')}{\sum_{k=1}^{N} \exp(s_{ik}')}$$

Propriété : $$\sum_{j=1}^{N} \alpha_{ij} = 1$$ (distribution de probabilité)

**Étape 4 : Agrégation pondérée**
$$\text{output}_i = \sum_{j=1}^{N} \alpha_{ij} v_j$$

Résultat : Chaque token est une moyenne pondérée des valeurs, avec poids proportionnels à la similarité query-key.

### 2. Complexité Computationnelle Détaillée

**Complexité d'une couche BERT** :

**Multi-Head Attention** :
- Projections $$Q, K, V$$ : $$3NH^2$$
- Scores d'attention $$QK^T$$ : $$N^2H$$
- Softmax : $$N^2$$
- Pondération $$\alpha V$$ : $$N^2H$$
- Projection de sortie $$W^O$$ : $$NH^2$$
- **Total** : $$O(N^2H + NH^2)$$

**Feed-Forward Network** :
- $$W_1$$ : $$N \cdot H \cdot 4H = 4NH^2$$
- $$W_2$$ : $$N \cdot 4H \cdot H = 4NH^2$$
- **Total** : $$O(NH^2)$$

**Complexité pour $$L$$ couches** :
$$O(L \cdot (N^2H + NH^2))$$

**Dominance** :
- Pour $$N < H$$ : $$NH^2$$ domine (FFN)
- Pour $$N > H$$ : $$N^2H$$ domine (attention)

BERT-base : $$N=512, H=768$$ → $$N < H$$ → FFN domine en théorie, mais attention reste coûteuse en pratique.

### 3. Analyse de l'Apprentissage par Gradient

**Problème des gradients vanishing dans les réseaux profonds** :

Sans connexions résiduelles :
$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=1}^{L} \frac{\partial x_i}{\partial x_{i-1}}$$

Si $$\left\| \frac{\partial x_i}{\partial x_{i-1}} \right\| < 1$$ pour tout $$i$$ :
$$\left\| \frac{\partial \mathcal{L}}{\partial x_0} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial x_L} \right\| \cdot \rho^L$$

Avec $$\rho < 1$$ → gradient exponentiellement petit.

**Avec connexions résiduelles** :
$$x_{i+1} = x_i + F(x_i)$$

$$\frac{\partial x_{i+1}}{\partial x_i} = I + \frac{\partial F(x_i)}{\partial x_i}$$

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=1}^{L} \left(I + \frac{\partial F(x_i)}{\partial x_{i-1}}\right)$$

L'identité $$I$$ garantit un "chemin direct" pour le gradient (highway).

**Layer Normalization** stabilise également :
- Recentre et réduit les activations → évite explosions/disparitions

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreurs fréquentes

#### Erreur 1 : Confondre [CLS] et moyenne des tokens

**Description** :
Pour une tâche de classification, utiliser la moyenne de tous les tokens au lieu de [CLS].

**Pourquoi c'est problématique** :
- `[CLS]` est **spécifiquement entraîné** pour la représentation globale pendant le pré-entraînement NSP
- Moyenne naïve pondère également tous les tokens (y compris padding, ponctuation)

**Solution** :
```python
# ❌ MAUVAIS
outputs = model(input_ids)
mean_embedding = outputs.last_hidden_state.mean(dim=1)  # Moyenne de tous les tokens

# ✅ BON
outputs = model(input_ids)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # Token [CLS] seulement

# Ou utiliser le pooler (déjà optimisé pour [CLS])
cls_pooled = outputs.pooler_output
```

**Impact** : Utiliser [CLS] améliore généralement de 2-5% sur les tâches de classification.

**Source** : [BERT paper original](https://arxiv.org/abs/1810.04805) - Section 3.4

#### Erreur 2 : Learning rate trop élevé pour le fine-tuning

**Description** :
Utiliser le même learning rate que pour l'entraînement from scratch (ex: 1e-3).

**Pourquoi c'est problématique** :
- BERT pré-entraîné a déjà de bonnes représentations
- LR élevé → **catastrophic forgetting** (perte des connaissances pré-entraînées)

**Solution** :
```python
# ❌ MAUVAIS
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ✅ BON
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# Learning rates typiques : 1e-5, 2e-5, 3e-5, 5e-5

# Encore mieux : Discriminative learning rates
# Couches basses (près des embeddings) : LR plus faible
# Couches hautes + classifier : LR plus élevé
```

**Impact** : LR mal choisi peut dégrader les performances de 10-20%.

**Source** : [Howard & Ruder, 2018 - ULMFiT](https://arxiv.org/abs/1801.06146)

#### Erreur 3 : Ignorer le warmup

**Description** :
Démarrer directement avec le learning rate cible sans période de warmup.

**Pourquoi c'est problématique** :
- Les premiers steps ont des gradients instables (variance élevée)
- Démarrage brutal → divergence possible

**Solution** :
```python
# ❌ MAUVAIS
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# ✅ BON
from transformers import get_linear_schedule_with_warmup

num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Learning rate augmente linéairement pendant warmup, puis décroît
```

**Impact** : Warmup améliore la stabilité et peut gagner 1-3% de performance.

**Source** : [Transformer original](https://arxiv.org/abs/1706.03762) - Section 5.3

#### Erreur 4 : Tokenization incorrecte avec séquences tronquées

**Description** :
Tronquer les séquences sans tenir compte de l'information perdue.

**Exemple problématique** :
Question : "Quelle est la capitale de la France ?"
Contexte : "[300 tokens inutiles]... La capitale de la France est Paris."

Si on tronque à 512 tokens, on perd la réponse !

**Solution** :
```python
# ❌ MAUVAIS
inputs = tokenizer(
    question, 
    context, 
    max_length=512, 
    truncation=True  # Tronque arbitrairement
)

# ✅ BON (pour Question Answering)
inputs = tokenizer(
    question,
    context,
    max_length=512,
    truncation="only_second",  # Tronque seulement le contexte (2e séquence)
    stride=128,  # Overlap pour sliding window
    return_overflowing_tokens=True  # Récupère les tokens tronqués
)

# Pour d'autres tâches : stratégies de chunking
# Exemple : Découper un long document en passages de 512 tokens avec overlap
```

**Impact** : Stratégie de truncation adaptée est cruciale pour QA (peut améliorer F1 de 5-10%).

### ✅ Bonnes pratiques

#### Pratique 1 : Gradual Unfreezing pour Fine-Tuning

**Principe** :
Geler progressivement les couches du bas vers le haut pendant le fine-tuning.

**Justification** :
- **Couches basses** : Capturent des features générales (syntaxe, morphologie) → peu besoin de modification
- **Couches hautes** : Capturent des features task-specific → nécessitent adaptation

**Implémentation** :
```python
# Étape 1 : Geler toutes les couches sauf le classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Entraîner classifier seul (quelques epochs)
# ...

# Étape 2 : Dégeler les 2 dernières couches
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

# Entraîner avec LR plus faible
# ...

# Étape 3 : Dégeler tout avec LR encore plus faible
for param in model.parameters():
    param.requires_grad = True
```

**Sources** :
- [ULMFiT paper](https://arxiv.org/abs/1801.06146)
- [Howard, fastai course](https://course.fast.ai/)

#### Pratique 2 : Mixed Precision Training (FP16)

**Principe** :
Utiliser float16 au lieu de float32 pour réduire la mémoire et accélérer le calcul.

**Justification** :
- **Mémoire** : FP16 = 2 bytes vs FP32 = 4 bytes → division par 2
- **Vitesse** : GPUs modernes (Tensor Cores) optimisés pour FP16

**Implémentation** :
```python
from torch.cuda.amp import autocast, GradScaler

model = BertForSequenceClassification.from_pretrained("camembert-base")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()  # Pour éviter underflow numérique

for batch in train_dataloader:
    optimizer.zero_grad()
    
    # Forward pass en mixed precision
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Backward pass avec scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Impact** :
- Réduction mémoire : ~40-50%
- Accélération : 2-3× sur GPUs récents (V100, A100)

**Sources** :
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

#### Pratique 3 : Utiliser des Ensembles de Modèles

**Principe** :
Combiner les prédictions de plusieurs modèles pour améliorer la robustesse.

**Stratégies** :
1. **Même architecture, différentes seeds** : 3-5 runs avec initialisations différentes
2. **Différentes architectures** : BERT + RoBERTa + CamemBERT
3. **Différents hyperparamètres** : LR, batch size, epochs

**Implémentation** :
```python
# Entraîner 5 modèles avec seeds différentes
models = []
for seed in [42, 123, 456, 789, 1011]:
    set_seed(seed)
    model = train_model(...)  # Votre pipeline d'entraînement
    models.append(model)

# Prédiction par vote majoritaire (classification)
def ensemble_predict(inputs):
    predictions = []
    for model in models:
        pred = model(inputs).logits.argmax(dim=-1)
        predictions.append(pred)
    
    # Vote majoritaire
    predictions = torch.stack(predictions)
    final_pred = torch.mode(predictions, dim=0).values
    return final_pred

# Ou moyenne des probabilités (soft voting)
def ensemble_predict_soft(inputs):
    probs = []
    for model in models:
        prob = torch.softmax(model(inputs).logits, dim=-1)
        probs.append(prob)
    
    # Moyenne des probabilités
    avg_prob = torch.stack(probs).mean(dim=0)
    final_pred = avg_prob.argmax(dim=-1)
    return final_pred
```

**Impact** :
- Amélioration typique : 1-3% sur métriques (F1, accuracy)
- Coût : ×N inference time (N = nombre de modèles)

**Sources** :
- [Kaggle winning solutions](https://www.kaggle.com/) (ensembles quasi-systématiques)

### 📋 Checklist de Validation

Avant de considérer votre implémentation BERT complète :

- [ ] **Tokenization** : Utiliser le tokenizer correspondant au modèle (CamembertTokenizer pour CamemBERT, etc.)
- [ ] **Learning rate** : Tester 1e-5, 2e-5, 5e-5 (pas plus haut)
- [ ] **Warmup** : Inclure 5-10% de warmup steps
- [ ] **Longueur de séquence** : Vérifier la distribution des longueurs de vos données (padding/truncation optimal)
- [ ] **Batch size** : Adapter à votre GPU (8-32 typique), utiliser gradient accumulation si mémoire limitée
- [ ] **Epochs** : 2-4 epochs suffisent généralement (plus = risque overfitting)
- [ ] **Validation** : Monitorer performance sur validation set (early stopping si dégradation)
- [ ] **Métriques** : Choisir métrique adaptée (F1 pour classes déséquilibrées, accuracy pour équilibrées)
- [ ] **Reproductibilité** : Fixer les seeds (torch, numpy, random)
- [ ] **Mixed precision** : Utiliser FP16 si GPU compatible (gain mémoire/vitesse)

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques Fondamentaux

#### 1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- **Auteurs** : Devlin et al. (Google AI) (2018)
- **Publication** : NAACL 2019
- **URL** : [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Contribution clé** : Introduction du pré-entraînement bidirectionnel via Masked Language Modeling, révolutionnant le transfer learning en NLP
- **Pertinence** : **MUST READ** - Fondation de toute l'architecture, explique MLM et NSP en détail
- **Niveau** : Accessible (très bien écrit, exemples clairs)

#### 2. RoBERTa: A Robustly Optimized BERT Pretraining Approach

- **Auteurs** : Liu et al. (Facebook AI Research) (2019)
- **Publication** : arXiv
- **URL** : [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- **Contribution clé** : Démontre que les hyperparamètres de pré-entraînement sont aussi importants que l'architecture (suppression NSP, masquage dynamique, plus de données)
- **Pertinence** : Essentiel pour comprendre l'optimisation du pré-entraînement et les choix de CamemBERT
- **Niveau** : Accessible (focus expérimental)

#### 3. CamemBERT: a Tasty French Language Model

- **Auteurs** : Martin et al. (Inria, Facebook AI) (2020)
- **Publication** : ACL 2020
- **URL** : [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)
- **Contribution clé** : Adaptation de RoBERTa au français avec corpus OSCAR, démontrant l'importance des modèles monolingues
- **Pertinence** : **MUST READ** si vous travaillez en français - benchmarks, comparaisons avec mBERT et FlauBERT
- **Niveau** : Accessible (application directe de RoBERTa)

#### 4. FlauBERT: Unsupervised Language Model Pre-training for French

- **Auteurs** : Le et al. (CNRS) (2020)
- **Publication** : LREC 2020
- **URL** : [arXiv:1912.05372](https://arxiv.org/abs/1912.05372)
- **Contribution clé** : Modèle français avec variantes Small/Base/Large, comparaisons détaillées
- **Pertinence** : Utile pour comparaisons avec CamemBERT et choix de modèle
- **Niveau** : Accessible

#### 5. Attention is All You Need (Transformer Original)

- **Auteurs** : Vaswani et al. (Google Brain) (2017)
- **Publication** : NeurIPS 2017
- **URL** : [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Contribution clé** : Introduction du mécanisme d'attention et de l'architecture Transformer (base de BERT)
- **Pertinence** : Fondamental pour comprendre self-attention, multi-head, positional encoding
- **Niveau** : Technique (formulations mathématiques détaillées)

#### 6. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting

- **Auteurs** : Voita et al. (2019)
- **Publication** : ACL 2019
- **URL** : [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)
- **Contribution clé** : Analyse empirique du rôle de chaque tête d'attention (positionnelles, syntaxiques, rares)
- **Pertinence** : Compréhension approfondie du fonctionnement interne de BERT
- **Niveau** : Technique (analyse quantitative)

#### 7. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators

- **Auteurs** : Clark et al. (Stanford, Google) (2020)
- **Publication** : ICLR 2020
- **URL** : [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)
- **Contribution clé** : Alternative à MLM : détection de tokens remplacés (plus efficace en data efficiency)
- **Pertinence** : Exploration d'alternatives à BERT pour pré-entraînement
- **Niveau** : Technique

### 📚 Ressources Complémentaires

#### Articles de blog techniques

- **The Illustrated BERT** par Jay Alammar
  - [http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/)
  - 📌 **Pourquoi** : Visualisations exceptionnelles de l'architecture et du mécanisme d'attention
  - ⏱️ **Durée** : ~15 min

- **BERT Fine-Tuning Tutorial** par Chris McCormick
  - [https://mccormickml.com/2019/07/22/BERT-fine-tuning/](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
  - 📌 **Pourquoi** : Guide pratique détaillé avec code PyTorch
  - ⏱️ **Durée** : ~30 min

- **Hugging Face Course - Transformer models**
  - [https://huggingface.co/course/chapter1/1](https://huggingface.co/course/chapter1/1)
  - 📌 **Pourquoi** : Cours complet sur les Transformers et BERT avec exemples interactifs
  - ⏱️ **Durée** : 2-3 heures (cours complet)

#### Vidéos éducatives

- **BERT Explained (CodeEmporium)**
  - [https://www.youtube.com/watch?v=xI0HHN5XKDo](https://www.youtube.com/watch?v=xI0HHN5XKDo)
  - 📌 **Pourquoi** : Explication visuelle claire de l'architecture
  - ⏱️ **Durée** : 15 min

- **Attention is All You Need (Yannic Kilcher)**
  - [https://www.youtube.com/watch?v=iDulhoQ2pro](https://www.youtube.com/watch?v=iDulhoQ2pro)
  - 📌 **Pourquoi** : Explication détaillée du paper Transformer original
  - ⏱️ **Durée** : 50 min

#### Documentation officielle

- **Hugging Face Transformers - BERT**
  - [https://huggingface.co/docs/transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert)
  - 📌 **Section recommandée** : Model Architecture, Usage
  - 📌 **Pourquoi** : Référence complète pour l'implémentation

- **CamemBERT Model Card**
  - [https://huggingface.co/camembert-base](https://huggingface.co/camembert-base)
  - 📌 **Section recommandée** : Intended uses & limitations, Training data
  - 📌 **Pourquoi** : Détails sur le pré-entraînement et les benchmarks

### 🛠️ Outils et Frameworks

#### Outil 1 : Hugging Face Transformers

- **URL** : [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- **Description** : Bibliothèque Python pour charger, fine-tuner et utiliser BERT et variantes
- **Cas d'usage** : Accès facile à CamemBERT, FlauBERT, BERT multilingue
- **Installation** :
```bash
pip install transformers torch
```
- **Exemple rapide** :
```python
from transformers import CamembertModel, CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")

inputs = tokenizer("Bonjour, comment allez-vous ?", return_tensors="pt")
outputs = model(**inputs)
```

#### Outil 2 : BertViz (Visualisation d'Attention)

- **URL** : [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
- **Description** : Outil interactif pour visualiser les patterns d'attention dans BERT
- **Cas d'usage** : Comprendre quelles relations les têtes d'attention capturent, debugging
- **Installation** :
```bash
pip install bertviz
```
- **Exemple rapide** :
```python
from bertviz import head_view
from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("The cat sat on the mat", return_tensors='pt')
outputs = model(**inputs)

attention = outputs.attentions  # Tuple de tensors (12 couches)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Visualisation interactive
head_view(attention, tokens)
```

#### Outil 3 : Sentence Transformers

- **URL** : [https://www.sbert.net/](https://www.sbert.net/)
- **Description** : Framework pour créer des embeddings de phrases avec BERT (utile pour similarité sémantique)
- **Cas d'usage** : Recherche sémantique, clustering de documents, paraphrase detection
- **Installation** :
```bash
pip install sentence-transformers
```
- **Exemple rapide** :
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

sentences = [
    "Paris est la capitale de la France.",
    "La capitale française est Paris.",
    "Il fait beau aujourd'hui."
]

embeddings = model.encode(sentences)
# Calcul de similarité cosine
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
```

### 📖 Cours et Tutoriels Connexes

#### Dans votre repository (Liens Zettelkasten)

- **Architecture similaire** :
  - [[vit.md]] - Vision Transformer utilise la même architecture encodeur que BERT (application à la vision)
  - **Raison de la liaison** : Comprendre l'adaptation du Transformer à d'autres domaines (vision vs texte)

- **Approfondissement** :
  - [[bert_fine_tuning_strategies.md]] (à créer) - Stratégies avancées de fine-tuning (discriminative LR, gradual unfreezing)
  - [[attention_mechanisms_deep_dive.md]] (à créer) - Analyse mathématique approfondie de l'attention
  - **Ce qui sera couvert** : Variantes d'attention (sparse, linear, local), optimisations

- **Sujets parallèles** :
  - [[03_llm/distillation_llm.md]] - DistilBERT (compression de BERT)
  - **Lien thématique** : Techniques de compression pour déploiement (distillation, quantization, pruning)
  
  - [[tokenization_nlp.md]] (à créer) - WordPiece, BPE, SentencePiece en profondeur
  - **Lien thématique** : Fondamental pour comprendre les différences entre variantes

#### Cours externes recommandés

- **Stanford CS224N: Natural Language Processing with Deep Learning**
  - [http://web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)
  - 📌 **Modules pertinents** : Lecture 10 (Transformers), Lecture 11 (BERT)
  - ⏱️ **Durée** : 2×1h30

- **fast.ai - Practical Deep Learning for Coders (Part 2)**
  - [https://course.fast.ai/](https://course.fast.ai/)
  - 📌 **Modules pertinents** : Lesson 9-10 (NLP with Transformers)
  - ⏱️ **Durée** : 2×2h

- **Hugging Face NLP Course**
  - [https://huggingface.co/course](https://huggingface.co/course)
  - 📌 **Modules pertinents** : Chapter 1-3 (Transformer models, Fine-tuning)
  - ⏱️ **Durée** : 5-6 heures (complet)

---

## 📝 Résumé Rapide (Quick Reference)

> **Carte de référence** : À consulter rapidement pour se remémorer l'essentiel.

### Concepts Clés

| Concept | Formule/Définition | Cas d'usage |
|---------|-------------------|-------------|
| **Self-Attention** | $$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$ | Capturer relations contextuelles entre tokens |
| **Multi-Head** | $$\text{Concat}(\text{head}_1,...,\text{head}_A)W^O$$ | Différentes perspectives (syntaxe, sémantique) |
| **MLM** | Masquer 15% des tokens → Prédire | Pré-entraînement bidirectionnel |
| **Position Embeddings** | Vecteurs apprenables (512 positions) | Encoder l'ordre des tokens |
| **[CLS]** | Premier token → Classification | Représentation globale de la séquence |
| **Layer Norm** | $$\gamma \frac{x-\mu}{\sigma} + \beta$$ | Stabilisation de l'entraînement |

### Architectures Comparées

| Modèle | $$L$$ | $$H$$ | $$A$$ | Params | Meilleur pour |
|--------|-------|-------|-------|--------|---------------|
| BERT-base | 12 | 768 | 12 | 110M | Anglais, baseline |
| RoBERTa-base | 12 | 768 | 12 | 125M | Anglais, optimisé |
| CamemBERT | 12 | 768 | 12 | 110M | **Français (recommandé)** |
| FlauBERT-base | 12 | 768 | 12 | 138M | Français, alternative |
| FlauBERT-small | 6 | 512 | 8 | 54M | Français, ressources limitées |

### Code Minimal

```python
# Chargement CamemBERT pour classification
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch

# Modèle et tokenizer
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Tokenization
text = "Ce film est excellent !"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Prédiction
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.softmax(outputs.logits, dim=-1).argmax().item()
```

### Décisions Clés

**Quand utiliser BERT (et variantes) :**

```
Tâche NLP française ?
├─ OUI → CamemBERT (par défaut)
│   └─ Ressources limitées ? → FlauBERT-small
│   └─ Performance maximale ? → FlauBERT-large
│
└─ NON (autre langue)
└─ Langue supportée par modèle monolingue ?
├─ OUI → Utiliser modèle monolingue (meilleur)
└─ NON → mBERT ou XLM-RoBERTa
```
**Hyperparamètres de fine-tuning :**
```
Learning Rate : 2e-5 (tester 1e-5, 5e-5)
Batch Size    : 16-32 (ou gradient accumulation)
Epochs        : 3-4 (rarement plus)
Warmup        : 10% des steps
Optimizer     : AdamW
Scheduler     : Linear decay avec warmup
```

### Pièges à éviter

1. ⚠️ **LR trop élevé (>1e-4)** → Solution : 2e-5 standard, tester 1e-5 à 5e-5
2. ⚠️ **Pas de warmup** → Solution : 10% warmup steps
3. ⚠️ **Moyenne tokens au lieu de [CLS]** → Solution : Utiliser `outputs.pooler_output` ou `[:, 0, :]`
4. ⚠️ **Tronquer sans stratégie** → Solution : `truncation="only_second"` pour QA, stride pour documents longs
5. ⚠️ **Ignorer mixed precision** → Solution : Utiliser `torch.cuda.amp` (gain mémoire/vitesse)

