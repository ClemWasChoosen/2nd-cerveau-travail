# Transformers et LLM : cours complet, architecture, entraînement et utilisation

> **Résumé en une phrase** : ce cours relie les fondements des Transformers, le cycle de vie complet des grands modèles de langage, leur utilisation en production et une annexe datée sur l'état de l'art au 17 septembre 2026.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-09-17 |
| **Dernière mise à jour** | 2026-09-17 |
| **Domaine** | Deep Learning / NLP / LLM |
| **Niveau** | Intermédiaire à Avancé |
| **Durée de lecture** | ~2 h 30 pour le parcours principal, davantage avec les approfondissements |
| **Fichier** | `transformers_llm_cours_complet.md` |
| **Emplacement** | `/02_deep_learning/03_llm/` |
| **Tags** | `#transformer` `#llm` `#attention` `#pretraining` `#alignment` `#rag` `#agents` `#inference` `#evaluation` |
| **Cutoff état de l'art** | 2026-09-17 |

### Prérequis

- Algèbre linéaire : vecteurs, matrices, produit matriciel
- Probabilités : softmax, log-vraisemblance, entropie croisée
- Deep learning : gradient descent, backpropagation, Adam
- Python et PyTorch, au moins au niveau lecture de code
- Notions de GPU et de mémoire utiles pour les sections serving

### Cours connexes

Ce document est le cours d'entrée unique. Les fichiers suivants sont des approfondissements spécialisés :

- [[bert_architecture_variantes.md]] : encodeur Transformer, BERT et variantes françaises
- [[rag.md]] : RAG, recherche vectorielle, chunking, reranking et évaluation
- [[peft_lora.md]] : PEFT, LoRA, QLoRA, SFT et DPO en pratique
- [[methodes_entrainement_llm.md]] : SFT, PPO, DPO, GRPO, KTO et online RL
- [[apprentissage_renforcement_llm.md]] : bases de RL et RLHF
- [[distillation_llm.md]] : compression et transfert de capacités
- [[serving_llm.md]] : serving, quantification, KV cache et moteurs d'inférence
- [[infra_ia.md]] : infrastructure et choix de déploiement

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Ce cours suit le chemin complet d'un système LLM :

```text
texte brut
  → tokens et embeddings
  → attention et blocs Transformer
  → pré-entraînement
  → post-entraînement et alignement
  → génération autoregressive
  → serving et optimisation
  → prompting, outils, RAG et agents
  → évaluation, sécurité et choix du modèle
```

L'objectif n'est pas de mémoriser une liste de noms. Il s'agit de construire un modèle mental assez solide pour répondre à quatre questions :

1. **Que calcule réellement le modèle ?**
2. **Pourquoi une modification architecturale ou un algorithme d'entraînement change-t-il son comportement ?**
3. **Quelle technique faut-il utiliser pour une application donnée ?**
4. **Comment comparer des modèles sans confondre benchmark, marketing et performance métier ?**

### Objectifs d'apprentissage, taxonomie de Bloom

À la fin du cours, vous serez capable de :

1. **Comprendre** le passage d'un texte à une distribution de probabilités sur le prochain token.
2. **Expliquer** self-attention, multi-head attention, masques causaux, FFN, résidus et normalisation avec les formes des tenseurs.
3. **Comparer** encodeur-only, decoder-only, encoder-decoder, modèles denses et MoE.
4. **Analyser** un pipeline de pré-entraînement, SFT, RLHF/DPO/GRPO, PEFT et distillation.
5. **Dimensionner** grossièrement mémoire, KV cache, coût de génération et débit.
6. **Construire** une application avec prompting, sortie structurée, tool calling, RAG et workflows contrôlés.
7. **Évaluer** qualité, factualité, retrieval, tool use, coût, latence et sécurité.
8. **Choisir** un modèle par tâche et contraintes, sans chercher un classement global intemporel.

### TL;DR en trois phrases

Un LLM est principalement un modèle qui estime `P(token suivant | tokens précédents)` en faisant passer une séquence de vecteurs dans des blocs Transformer. Le pré-entraînement lui donne des régularités linguistiques et de nombreuses capacités latentes, tandis que le post-entraînement lui apprend à suivre des instructions, préférences et contraintes. En production, la qualité finale vient du couple **modèle + données/contexte + outils + orchestration + évaluation**, pas du nom du modèle seul.

---

## 1. Intuition générale : un LLM comme compresseur génératif

### 1.1 Ce qu'un LLM apprend réellement

Prenons une séquence de tokens `x_1, ..., x_T`. Un modèle autoregressif apprend une factorisation :

$`P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})`$

À chaque position, il produit un vecteur de logits de taille `V`, où `V` est le vocabulaire. Le softmax transforme ces logits en probabilités :

$`P(x_t=i \mid x_{<t}) = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}`$

Le modèle ne « récupère » pas une réponse déjà stockée. Il calcule une distribution conditionnelle et génère progressivement. Cela explique :

- sa capacité à recombiner des motifs vus dans les données ;
- sa sensibilité au contexte fourni ;
- ses erreurs plausibles mais fausses ;
- l'importance du décodage, du retrieval et de la validation externe.

### 1.2 Intelligence, mémoire et vérité sont trois axes différents

Un modèle peut être très bon en raisonnement abstrait mais mauvais pour une information récente. Il peut connaître un fait sans savoir le citer, ou produire une réponse élégante sans preuve. Il faut séparer :

| Propriété | Question | Technique associée |
|---|---|---|
| Capacité | Le modèle sait-il résoudre la classe de problème ? | Modèle, raisonnement, entraînement |
| Mémoire paramétrique | Le fait est-il présent dans les poids ? | Pré-entraînement, continued pretraining |
| Contexte | Le modèle peut-il exploiter l'information fournie ? | Prompt, fenêtre de contexte, attention |
| Accès | Peut-il consulter une source ou un système ? | RAG, outils, APIs |
| Fiabilité | Sait-il s'abstenir, citer et respecter un format ? | Post-entraînement, garde-fous, validation |
| Coût | Peut-on l'utiliser à l'échelle ? | Petit modèle, quantification, caching, batching |

### 1.3 Analogie pédagogique

Imaginez un lecteur extrêmement rapide qui a lu une bibliothèque immense :

- le **pré-entraînement** lui apprend les régularités de la langue et des domaines ;
- le **post-entraînement** lui apprend à répondre comme un assistant ;
- le **prompt** lui donne la mission et le contexte immédiat ;
- le **RAG** lui ouvre une bibliothèque externe ;
- les **outils** lui permettent de calculer ou d'agir ;
- l'**évaluation** vérifie que ses réponses satisfont le besoin réel.

Un lecteur peut être cultivé sans être fiable dans une procédure métier. C'est exactement pourquoi l'application doit être conçue autour du modèle, et pas seulement autour de lui.

---

## 2. Du texte aux tenseurs

### 2.1 Tokenisation

Un modèle ne reçoit pas directement des mots. Il reçoit des identifiants entiers correspondant à des tokens. Un tokenizer segmente le texte selon un vocabulaire appris.

Exemple conceptuel :

```text
"Les transformers sont puissants"
→ ["Les", " transform", "ers", " sont", " puissant", "s"]
→ [412, 9821, 71, 203, 8711, 19]
```

La tokenisation est un compromis :

- un vocabulaire trop petit produit beaucoup de tokens et augmente le coût ;
- un vocabulaire trop grand augmente la taille de la matrice d'embeddings ;
- les langues, le code, les nombres et les caractères rares ne sont pas segmentés de la même façon.

Les grandes familles historiques sont BPE, WordPiece et SentencePiece. Le token n'est pas forcément un mot, une syllabe ou un caractère. Il peut contenir un espace initial, un fragment de mot ou une séquence de caractères fréquente.

**Conséquence pratique :** le nombre de tokens est le vrai coût d'entrée d'un LLM, pas le nombre de mots. Il faut mesurer les tokens sur les données réelles.

### 2.2 Embeddings

On transforme chaque identifiant en vecteur dense :

$$E \in \mathbb{R}^{V \times d_{model}}$$

Pour un token `x_t`, l'embedding est la ligne `E[x_t]`, de dimension `d_model`.

Pour un batch :

```text
input_ids : [B, T]
embeddings : [B, T, d_model]
```

- `B` : batch size
- `T` : longueur de séquence
- `d_model` : dimension cachée

### 2.3 Information de position

L'attention seule ne sait pas si un token apparaît avant ou après un autre. Il faut injecter l'ordre. Les approches principales sont :

- embeddings positionnels appris ;
- sinus et cosinus ;
- RoPE, rotation relative des requêtes et clés ;
- ALiBi, biais dépendant de la distance ;
- variantes destinées à extrapoler vers de longues séquences.

Avec RoPE, la position modifie la représentation de façon à ce que le produit scalaire entre une requête et une clé dépende de leur différence de position. Cette propriété est particulièrement utile dans l'attention causale et les extensions de contexte.

### 2.4 Décomposition mentale d'un input

```text
texte
  → tokenizer
input_ids [B, T]
  → embedding lookup
X [B, T, d_model]
  → information de position
H0 [B, T, d_model]
  → N blocs Transformer
HN [B, T, d_model]
  → projection vers vocabulaire
logits [B, T, V]
```

---

## 3. Self-attention, le cœur du Transformer

### 3.1 Pourquoi l'attention ?

Pour comprendre un token, il faut sélectionner les autres tokens pertinents. Dans « La banque près de la rivière », le mot « rivière » aide à désambiguïser « banque ». Dans « La banque a publié ses résultats », « publié » et « résultats » orientent vers le sens financier.

L'attention apprend dynamiquement quelles positions consulter et avec quelle intensité.

![Image Mechanisme attention](https://github.com/ClemWasChoosen/2nd-cerveau-travail/blob/main/divers/image/attention-mechanism.jpg)

### 3.2 Requêtes, clés et valeurs

À partir de `X`, trois projections linéaires produisent :

$`Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V`$

Si `X` a la forme `[B, T, d_model]`, et si les projections conservent `d_model` :

```text
Q, K, V : [B, T, d_model]
```

Intuition :

- **Query** : ce que cherche la position courante ;
- **Key** : ce que chaque position propose comme indice ;
- **Value** : le contenu transmis si la position est pertinente.

### 3.3 Scaled dot-product attention

L'attention est :

$`Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V`$

- `QK^T` compare toutes les requêtes à toutes les clés ;
- `sqrt(d_k)` stabilise l'échelle des produits ;
- `M` est un masque éventuel ;
- le softmax transforme les scores en poids ;
- la multiplication par `V` agrège les informations.

Formes typiques :

```text
Q : [B, H, T, d_head]
K : [B, H, T, d_head]
V : [B, H, T, d_head]
QKᵀ : [B, H, T, T]
output : [B, H, T, d_head]
```

La matrice `[T, T]` explique le coût quadratique de l'attention dense en longueur de séquence.

### 3.4 Masque causal

Pour un modèle decoder-only, la position `t` ne doit pas voir les tokens futurs. Le masque est triangulaire :

```text
position 1 : ✓ · · ·
position 2 : ✓ ✓ · ·
position 3 : ✓ ✓ ✓ ·
position 4 : ✓ ✓ ✓ ✓
```

Les positions interdites reçoivent un score proche de `-∞` avant le softmax. Le modèle peut donc être entraîné en parallèle sur toute la séquence, tout en respectant la causalité.

### 3.5 Multi-head attention

Au lieu d'une seule attention, on utilise plusieurs têtes :

$$
head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
$$

$$
MHA(X) = Concat(head_1, ..., head_h)W_O
$$

Chaque tête peut apprendre un type de relation différent : accord grammatical, dépendance longue, structure de code, ponctuation, entités ou hiérarchie de document. Il ne faut toutefois pas surinterpréter chaque tête comme une règle sémantique propre et stable.

### 3.6 MHA, MQA et GQA

La différence porte sur le nombre de têtes `K,V` :

| Variante | `H_q` | `H_kv` | Effet |
|---|---:|---:|---|
| MHA | `H` | `H` | Qualité et mémoire KV maximales |
| MQA | `H` | `1` | KV cache minimal, compromis de qualité possible |
| GQA | `H` | intermédiaire | Bon compromis qualité, mémoire et débit |

GQA réduit surtout le coût du KV cache et du decode. Il ne réduit pas directement le nombre de paramètres des poids principaux.

### 3.7 Pourquoi FlashAttention ne rend pas l'attention linéaire

FlashAttention est une implémentation exacte qui réduit les accès mémoire grâce au tiling et à une meilleure localité. Elle évite de matérialiser toute la matrice intermédiaire en mémoire HBM, mais le calcul dense reste conceptuellement quadratique en `T`.

Piège fréquent : « FlashAttention rend l'attention `O(T)` ». La bonne formulation est : elle réduit fortement les mouvements mémoire et la mémoire auxiliaire pour l'attention dense, sans modifier sa complexité arithmétique fondamentale.

---

## 4. Le bloc Transformer complet

### 4.1 Architecture pré-normale

Une variante moderne d'un bloc decoder-only ressemble à :

```text
H
 ├─ LayerNorm / RMSNorm
 ├─ Self-Attention causale
 ├─ connexion résiduelle
 ├─ LayerNorm / RMSNorm
 ├─ Feed-Forward Network ou MoE
 ├─ connexion résiduelle
 ↓
H'
```

Formellement :

$$
H' = H + Attention(Norm(H))
$$

$$
H_{out} = H' + FFN(Norm(H'))
$$

La connexion résiduelle donne au gradient un chemin direct et permet d'empiler de nombreuses couches. La normalisation stabilise les activations.

### 4.2 RMSNorm et LayerNorm

LayerNorm centre et normalise par la moyenne et la variance. RMSNorm utilise principalement la norme quadratique moyenne et peut être plus simple et efficace. Le choix fait partie de la famille architecturale, mais il ne suffit pas à expliquer les performances d'un modèle.

### 4.3 Feed-Forward Network

Le FFN applique la même transformation indépendamment à chaque position :

$$
FFN(x) = W_2\,\sigma(W_1x + b_1) + b_2
$$

Dans beaucoup de LLM récents, on utilise une variante gated, par exemple SwiGLU :

$$
FFN(x) = W_2\left(SiLU(W_gx) \odot W_upx\right)
$$

L'attention mélange les positions. Le FFN transforme le contenu de chaque position. Une intuition utile est donc :

- attention = communication entre tokens ;
- FFN = calcul local et transformation de représentation.

### 4.4 Modèles denses et Mixture-of-Experts

Dans un modèle dense, chaque token traverse toutes les matrices principales. Dans un MoE, un routeur sélectionne quelques experts parmi beaucoup :

```text
représentation du token
        ↓
      routeur
   ↙    ↓    ↘
expert  expert  expert
        ↓
combinaison pondérée
```

Un modèle MoE peut avoir beaucoup de paramètres totaux tout en activant seulement une fraction par token. Cela améliore potentiellement le rapport capacité/coût de calcul, mais ajoute des problèmes de routage, équilibrage, communication entre GPU et mémoire des poids.

**À retenir :** paramètres actifs ne signifie pas mémoire totale minimale. En serving, il faut souvent garder ou charger l'ensemble des experts.

---

## 5. Les grandes familles d'architectures

### 5.1 Encoder-only : BERT et représentations

Un encoder-only voit généralement le contexte bidirectionnel. Il est bien adapté à :

- classification ;
- NER ;
- embeddings et retrieval ;
- extraction ;
- compréhension de texte.

Le pré-entraînement historique utilise le Masked Language Modeling. Pour approfondir : [[bert_architecture_variantes.md]].

### 5.2 Decoder-only : GPT et LLM génératifs

Un decoder-only utilise une attention causale et prédit le token suivant. Il est adapté à :

- génération ;
- conversation ;
- code ;
- tool calling ;
- raisonnement et workflows textuels ;
- génération structurée.

C'est la famille dominante des assistants généralistes actuels.

### 5.3 Encoder-decoder : T5 et séquence-à-séquence

Un encoder lit la source et un decoder génère la sortie en cross-attentionnant la représentation source. C'est naturel pour :

- traduction ;
- résumé ;
- transformation de texte ;
- certaines tâches contrôlées.

Les APIs généralistes utilisent souvent des decoder-only, mais le paradigme encoder-decoder reste important pour comprendre le design des modèles.

### 5.4 Multimodalité

Un modèle multimodal doit transformer image, audio, vidéo ou document en représentations compatibles avec son backbone linguistique. Les architectures peuvent utiliser :

- encodeur de vision + projection vers l'espace du LLM ;
- tokens visuels intercalés ;
- encodeurs spécialisés ;
- fusion précoce ou tardive ;
- modèles unifiés avec plusieurs têtes de sortie.

Une fiche « multimodal » ne signifie pas que toutes les modalités, résolutions, longueurs vidéo et sorties sont identiques. Vérifiez toujours les entrées, sorties et limites exactes du modèle.

### 5.5 Long contexte

Augmenter la fenêtre de contexte n'implique pas automatiquement une compréhension parfaite de toute la fenêtre. Les limites viennent de :

- coût et mémoire de l'attention ;
- capacité réelle à retrouver une information noyée ;
- entraînement sur des longueurs comparables ;
- position encoding ;
- bruit et redondance ;
- coût du prefill et du KV cache.

La règle pratique est de préférer un contexte court et pertinent à une fenêtre énorme remplie de documents non filtrés.

---

## 6. Pré-entraînement : apprendre le langage et les régularités

### 6.1 Loss autoregressive

Pour un corpus de tokens `x_1, ..., x_T` :

$$
\mathcal{L}_{LM}(\theta) = -\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})
$$

En pratique, on décale les labels d'un token : les logits à la position `t` prédisent le token à la position `t+1`.

```text
entrée :  [Je, lis, un, cours]
label  :  [lis, un, cours, de]
```

La cross-entropy moyenne est un signal d'entraînement efficace, mais elle ne mesure pas directement l'utilité, la factualité, la sécurité ou le respect des préférences humaines.

### 6.2 Données

La qualité du corpus dépend de :

- déduplication ;
- filtrage de toxicité et de données personnelles ;
- qualité linguistique ;
- mélange de domaines ;
- code et documents structurés ;
- provenance et licences ;
- équilibre entre données générales et spécialisées ;
- contamination des benchmarks.

Une grande quantité de tokens de mauvaise qualité peut être moins utile qu'un corpus plus petit mais mieux sélectionné. La composition des données est souvent aussi importante que la taille du modèle.

### 6.3 Optimisation

Le pipeline comprend généralement :

1. initialisation des poids ;
2. forward pass ;
3. calcul de la cross-entropy ;
4. backward pass ;
5. accumulation éventuelle des gradients ;
6. mise à jour AdamW ou variante ;
7. scheduler du learning rate ;
8. checkpointing et évaluation.

Pour entraîner de grands modèles, il faut distribuer :

- les données ;
- les couches ;
- les tenseurs des matrices ;
- les experts MoE ;
- les états d'optimisation.

### 6.4 Lois d'échelle

Les lois d'échelle décrivent des relations empiriques entre loss, taille du modèle, nombre de tokens et budget de calcul. Elles ne donnent pas une règle universelle pour choisir le meilleur modèle d'application.

Deux idées à retenir :

- augmenter seulement le nombre de paramètres peut être inefficace si le modèle est sous-entraîné ;
- un petit modèle très bien entraîné et post-entraîné peut battre un grand modèle mal adapté à une tâche.

### 6.5 Continued pretraining

Le continued pretraining, ou adaptation de domaine, continue la loss autoregressive sur un corpus spécialisé. Il est utile pour le vocabulaire, le style et certaines connaissances distribuées. Il peut toutefois dégrader le suivi d'instructions ou les capacités générales s'il n'y a pas de mélange de données générales et de phase de ré-alignement.

Ne pas confondre :

- **CPT** : adapter des connaissances et représentations ;
- **SFT** : apprendre un comportement à partir de réponses exemples ;
- **RAG** : fournir des connaissances externes sans modifier les poids.

---

## 7. Post-entraînement et alignement

### 7.1 Pourquoi un modèle de base n'est pas encore un assistant

Le pré-entraînement optimise la vraisemblance d'un texte. Il n'apprend pas directement :

- à répondre brièvement à une question ;
- à respecter un format JSON ;
- à refuser une demande dangereuse ;
- à suivre une procédure ;
- à utiliser un outil au bon moment.

Le post-entraînement transforme une capacité de complétion en comportement utile.

### 7.2 SFT

Le Supervised Fine-Tuning entraîne le modèle sur des conversations ou paires `instruction → réponse idéale` :

$$\mathcal{L}_{SFT}(\theta) = -\sum_{t \in \text{réponse}} \log \pi_\theta(y_t \mid x, y_{<t})$$

La loss est généralement masquée sur le prompt et calculée principalement sur les tokens de réponse.

Pièges :

- données incohérentes ;
- réponses trop longues et répétitives ;
- entraînement sur des raisonnements confidentiels non souhaités ;
- oubli des comportements généraux ;
- format de messages incompatible avec le tokenizer du modèle.

### 7.3 RLHF, RLAIF et reward model

RLHF suit historiquement :

```text
modèle base
  → SFT sur démonstrations
  → réponses candidates
  → préférences humaines
  → reward model
  → optimisation de la policy avec RL
```

RLAIF remplace ou complète les préférences humaines par un juge IA ou des principes explicites. Cela augmente la scalabilité, mais reporte le risque sur les biais du juge et la qualité du signal.

### 7.4 DPO

DPO utilise des triplets `(prompt, chosen, rejected)` sans entraîner séparément une policy RL complète. Son intuition est d'augmenter la probabilité relative de la réponse préférée par rapport à une réponse rejetée, sous contrainte d'un modèle de référence.

Il est souvent plus simple et stable que PPO, mais reste sensible à la qualité des paires et au décalage entre le dataset et les sorties produites en production.

### 7.5 PPO, REINFORCE, RLOO et GRPO

Ces méthodes génèrent des sorties et les évaluent avec un reward :

- PPO stabilise la mise à jour avec une contrainte de ratio et un modèle de valeur ;
- REINFORCE optimise directement le gradient de la récompense ;
- RLOO utilise des baselines leave-one-out ;
- GRPO compare un groupe de sorties pour estimer un avantage relatif sans critic classique.

Elles deviennent particulièrement intéressantes quand la tâche dispose d'un signal vérifiable : tests unitaires, solveur mathématique, vérificateur de preuve ou environnement.

Elles sont coûteuses parce qu'il faut générer plusieurs réponses, calculer les récompenses et maintenir une boucle online. Un reward facilement manipulable entraîne du reward hacking.

### 7.6 PEFT, LoRA et QLoRA

Pour une couche `W_0`, LoRA approxime la mise à jour :

$$
W' = W_0 + \Delta W
$$

$$
\Delta W = BA
$$

avec un rang `r` petit. La base reste gelée et seules `A` et `B` sont entraînées. QLoRA ajoute une quantification de la base pour réduire la mémoire.

Choisir PEFT quand :

- la base possède déjà les capacités générales ;
- le comportement ou le domaine est spécifique ;
- le budget GPU est limité ;
- plusieurs adaptateurs doivent coexister.

Choisir RAG plutôt que LoRA quand le besoin principal est une connaissance privée, changeante et traçable. Pour la pratique détaillée : [[peft_lora.md]].

### 7.7 Distillation

La distillation transfère un comportement de teacher vers un student :

- logits ;
- représentations ;
- réponses ;
- traces de raisonnement ou résumés de raisonnement ;
- données synthétiques vérifiées.

Elle permet de réduire coût et latence, mais le student hérite des erreurs et peut perdre des capacités rares. Une sortie convaincante du teacher n'est pas automatiquement une donnée d'entraînement correcte.

---

## 8. Génération et inférence

### 8.1 Décodage autoregressif

Le modèle produit un token, l'ajoute au contexte, puis recommence :

$`P(y_{1:T}\mid x) = \prod_{t=1}^{T}P(y_t \mid x,y_{<t})`$

Le décodage transforme la distribution en choix :

- greedy : choisir le token le plus probable ;
- temperature : modifier la concentration ;
- top-k : limiter aux `k` tokens ;
- top-p : limiter à une masse de probabilité ;
- beam search : explorer plusieurs séquences, plutôt pour certaines tâches seq2seq.

Pour les assistants modernes, le choix de decoding dépend du besoin : déterminisme pour extraction et code, diversité pour créativité, contrôle explicite pour raisonnement et agents.

### 8.2 Prefill et decode

Le **prefill** lit le prompt en parallèle et remplit le KV cache. Le **decode** produit les tokens de sortie un par un en réutilisant ce cache.

- prefill : plutôt compute-bound ;
- decode : souvent memory-bandwidth-bound ;
- TTFT : time to first token ;
- TPOT : time per output token ;
- throughput : tokens ou requêtes par seconde.

### 8.3 KV cache

Pour `L` couches, `H_kv` têtes K/V, dimension par tête `d_h`, longueur `S` et `b` octets par élément :

$$
M_{KV} \approx 2 \times L \times H_{kv} \times d_h \times S \times b
$$

Le facteur 2 vient de `K` et `V`. Avec plusieurs séquences, la mémoire se multiplie par la concurrence. GQA et MQA réduisent `H_kv`, donc la mémoire et le trafic du cache.

### 8.4 Quantification

La mémoire brute des poids est approximativement :

$$
M_{poids} \approx N_{paramètres} \times \frac{q}{8}
$$

`q` est le nombre de bits par poids. Les échelles, métadonnées, activations, buffers et KV cache s'ajoutent.

Exemples :

- BF16 : qualité robuste, mémoire élevée ;
- FP8 : intéressant sur GPU récents ;
- INT8 : compromis courant ;
- W4A16 : poids 4-bit et activations haute précision ;
- GGUF : format adapté à certains runtimes locaux.

Quantifier les poids ne signifie pas automatiquement quantifier les activations ou le KV cache.

### 8.5 Serving

Un moteur de serving moderne doit gérer :

- batching continu ;
- scheduling prefill/decode ;
- allocation efficace du KV cache ;
- parallélisme tensor, pipeline ou data ;
- quantification et kernels optimisés ;
- streaming ;
- timeouts et backpressure ;
- métriques de qualité et latence.

vLLM, SGLang, TensorRT-LLM et llama.cpp font des compromis différents selon GPU, taille du modèle, quantification, batch et objectifs de latence. Approfondissement : [[serving_llm.md]].

---

## 9. Utiliser un LLM : du prompt à l'agent

### 9.1 L'échelle d'autonomie

Commencer par la plus petite architecture suffisante :

```text
appel LLM simple
  → sortie structurée
  → tool calling contrôlé
  → RAG
  → workflow déterministe multi-étapes
  → agent borné
```

Un agent ajoute de l'autonomie, mais aussi de la latence, du coût, des erreurs cumulées et des risques d'action. Une tâche dont le chemin est connu est généralement mieux servie par un workflow déterministe.

### 9.2 Prompt robuste

Un bon prompt définit :

1. objectif ;
2. contexte et données ;
3. critères de réussite ;
4. contraintes ;
5. format de sortie ;
6. comportement en cas d'information manquante ;
7. exemples si nécessaire.

Exemple conceptuel :

````text
Tu extrais les incidents techniques d'un ticket.

Objectif : retourner uniquement les faits présents dans le ticket.

Règles :
- ne rien inventer ;
- utiliser null si une information manque ;
- conserver les dates et identifiants exactement ;
- séparer symptôme, cause et action corrective.

Sortie : JSON conforme au schéma fourni.
````

Le prompting n'est pas une sécurité suffisante. Les sorties doivent être validées par du code lorsque le système prend une décision ou déclenche une action.

### 9.3 Sorties structurées

Une sortie structurée impose un contrat entre le modèle et le programme :

```json
{
  "severity": "high",
  "components": ["database"],
  "confidence": 0.82,
  "needs_human_review": true
}
```

Il faut distinguer :

- JSON syntaxiquement valide ;
- schéma valide ;
- contenu sémantiquement correct ;
- décision acceptable pour le métier.

### 9.4 Tool calling

Le modèle ne doit pas recevoir directement les secrets ni exécuter arbitrairement une action. Le pattern sûr est :

```text
modèle propose un appel
  → application valide nom et arguments
  → application vérifie permissions et limites
  → application exécute l'outil
  → résultat marqué comme donnée externe
  → modèle formule la suite
```

Appliquer le moindre privilège, valider les arguments, limiter les effets de bord et demander une approbation humaine pour les actions sensibles.

### 9.5 RAG

Un pipeline RAG comprend :

```text
documents
  → parsing
  → nettoyage
  → chunking
  → embeddings
  → index vectoriel ou hybride
  → retrieval
  → reranking
  → contexte
  → génération avec citations
```

Pour une question `q`, un retriever produit `D_k(q)`, puis le générateur calcule :

$$
P(a \mid q, D_k(q))
$$

Les erreurs se répartissent entre :

- document absent ;
- mauvais chunking ;
- query mal reformulée ;
- retriever insuffisant ;
- reranker mauvais ;
- contexte trop long ou contradictoire ;
- génération non fidèle.

Évaluer séparément retrieval et génération. Pour le cours détaillé : [[rag.md]].

### 9.6 Agents

Un agent sélectionne dynamiquement des étapes et des outils. Il est utile lorsque :

- le chemin n'est pas connu à l'avance ;
- l'agent doit explorer ;
- les outils ont des résultats intermédiaires ;
- le problème est difficile à décomposer statiquement.

Il faut borner : nombre d'étapes, coût, durée, outils accessibles, types de fichiers, destinations réseau et actions irréversibles.

### 9.7 Mémoire

Distinguer :

- mémoire de conversation courte ;
- mémoire utilisateur persistante ;
- mémoire de travail de l'agent ;
- base documentaire RAG ;
- état d'un workflow.

Une mémoire n'est pas automatiquement vraie. Elle peut être obsolète, empoisonnée ou contraire à une permission actuelle. Les droits présents doivent toujours primer sur une mémoire passée.

---

## 10. Évaluation, fiabilité et sécurité

### 10.1 Évaluer le système, pas uniquement le modèle

Une évaluation utile contient :

- dataset représentatif ;
- cas fréquents et cas limites ;
- critères d'acceptation ;
- coût et latence ;
- taux d'abstention ;
- erreurs d'outils ;
- qualité des citations ;
- revue humaine sur les cas ambigus.

Les benchmarks externes sont des signaux, pas des garanties de performance sur vos propres données.

### 10.2 Évaluation RAG

Séparer :

| Niveau | Question | Exemples de métriques |
|---|---|---|
| Retrieval | Les bons documents sont-ils récupérés ? | recall@k, precision@k, MRR, nDCG |
| Contexte | Le contexte est-il suffisant et non contradictoire ? | contextual recall, relevancy, precision |
| Génération | La réponse est-elle fidèle et utile ? | faithfulness, answer relevancy |
| Système | La tâche métier est-elle réussie ? | exactitude, résolution, escalade, coût |

### 10.3 Évaluation des agents

Mesurer :

- succès de tâche de bout en bout ;
- exactitude des appels ;
- arguments corrects ;
- nombre d'étapes ;
- appels inutiles ;
- latence ;
- coût ;
- violation de permissions ;
- robustesse à l'échec d'un outil.

Un score de tool calling ne prédit pas nécessairement la réussite d'un agent complet : le harness, les outils, la mémoire et le simulateur comptent.

### 10.4 Juges LLM

Un LLM-as-judge permet d'annoter à l'échelle, mais il peut avoir :

- biais de position ;
- biais de style et de longueur ;
- préférence pour une réponse qui lui ressemble ;
- difficulté à vérifier des faits externes ;
- mauvaise calibration.

Permuter l'ordre des réponses, calibrer sur des annotations humaines et conserver un échantillon audité manuellement.

### 10.5 Prompt injection et confidentialité

Une injection peut être directe ou indirecte dans un document, e-mail, page web ou résultat RAG. Défense en profondeur :

- séparer instructions et contenu non fiable ;
- ne jamais considérer un document comme une instruction système ;
- valider chaque appel d'outil ;
- appliquer le moindre privilège ;
- sandboxer les outils ;
- filtrer secrets et données sensibles ;
- demander une approbation humaine pour les actions à impact ;
- tester attaques multilingues, obfusquées et indirectes.

---

## 11. Choisir un modèle

### 11.1 Les critères qui comptent

| Critère | Question |
|---|---|
| Capacité | Le modèle réussit-il votre tâche représentative ? |
| Raisonnement | Le budget de raisonnement améliore-t-il le résultat ? |
| Modalités | Textes, images, audio, vidéo, documents ? |
| Tool use | Les appels sont-ils exacts et conformes au schéma ? |
| Contexte | Quelle longueur utile, pas seulement annoncée ? |
| Latence | TTFT et débit sous votre concurrence ? |
| Coût | Prix tokens, cache, outils, stockage et orchestration ? |
| Données | Rétention, région, entraînement, DPA ? |
| Ouverture | Poids, licence, reproductibilité et possibilité de local ? |
| Cycle de vie | Alias, snapshots, dépréciation et migration ? |

### 11.2 Pourquoi il n'existe pas de meilleur modèle universel

Un modèle peut être meilleur en raisonnement scientifique, un autre en coding agentique, un autre en coût/latence ou en déploiement local. Les résultats changent avec :

- le prompt ;
- le budget de reasoning ;
- les outils ;
- le nombre de retries ;
- la longueur de contexte ;
- le harness d'agent ;
- la version exacte ;
- le protocole de benchmark.

La bonne méthode est de partir d'un petit jeu de requêtes réelles, de tester plusieurs modèles dans le même harness, puis de choisir selon le coût par tâche réussie.

### 11.3 Arbre de décision

```text
Avez-vous besoin d'informations privées ou récentes ?
  oui → RAG / outils avant fine-tuning
  non → testez un appel simple

La sortie doit-elle être consommée par du code ?
  oui → structured output + validation

Le modèle doit-il agir sur un système ?
  oui → tool calling avec permissions et approbation

Le chemin est-il connu à l'avance ?
  oui → workflow déterministe
  non → agent borné

Le comportement reste-t-il insuffisant après prompt, contexte et outils ?
  oui → évaluer PEFT/fine-tuning

La contrainte principale est-elle le coût ou la latence ?
  oui → petit modèle, distillation, routing ou quantification
```

---

## 12. État de l'art au 17 septembre 2026

> Cette section est volontairement datée. Les modèles, prix, aliases et benchmarks changent. Elle doit être mise à jour séparément du tronc fondamental.

### 12.1 Ce qui caractérise la période actuelle

Les tendances les plus importantes sont :

1. **Raisonnement configurable** : le budget de réflexion devient un paramètre de coût et de latence.
2. **Agents et computer use** : les modèles ne se limitent plus à produire du texte, ils sélectionnent des outils et interagissent avec des environnements.
3. **Contextes très longs** : les fenêtres atteignent des centaines de milliers ou environ un million de tokens, mais la qualité utile dépend du retrieval et de la structure du contexte.
4. **Modèles hybrides et MoE** : beaucoup de paramètres totaux, peu de paramètres actifs par token.
5. **Multimodalité native** : texte, image, audio, vidéo et documents convergent dans les APIs.
6. **Open-weight compétitif** : Qwen, Mistral, DeepSeek, GLM, Llama et d'autres familles réduisent l'écart sur certaines tâches tout en offrant contrôle et déploiement.
7. **Évaluation agentique** : SWE-bench, BFCL, OSWorld et les suites d'usage réel complètent les benchmarks de connaissance.
8. **Coût d'orchestration** : la qualité dépend souvent plus du pipeline, des outils et de l'évaluation que d'un changement marginal de modèle.

### 12.2 Familles propriétaires et API-first

| Famille observée | Positionnement pratique | Pourquoi elle peut être choisie | Réserves |
|---|---|---|---|
| **OpenAI GPT-6 Astra** | Modèle frontier pour travail complexe, raisonnement, computer use et tâches professionnelles | Très haut niveau généraliste annoncé ; Artificial Analysis le compare favorablement à Gemini 3.8 Flash sur son index v4.3 | Coût et latence élevés ; preuves indépendantes encore dépendantes du snapshot et du réglage |
| **Anthropic Claude Fable 5.1 / Opus 5 / Sonnet 5 / Haiku 4.5** | Raisonnement long, coding agentique, compromis vitesse/intelligence, haut débit | Catalogue clair, contexte jusqu'à 1M pour les modèles haut de gamme, outils et vision | Les labels de positionnement viennent du fournisseur ; tester le harness réel |
| **Google Gemini 3.8 Flash / Gemini 3.1 Pro** | Long horizon, multimodalité, agents, recherche et écosystème Google | Flash pour débit et coût, Pro pour raisonnement avancé et coding | Différence stable vs preview, modalités et prix à vérifier par endpoint |
| **xAI Grok 4.6** | Code, tool calling et raisonnement configurable | Offre API claire et contexte annoncé de 500K | Moins de preuves indépendantes homogènes selon tâches |
| **DeepSeek V4.1-Flash / V4 Pro** | Rapport capacité/coût, code, agents et raisonnement | Prix API compétitifs, évolution rapide, versions thinking/non-thinking | Disponibilité, multimodalité et compatibilité des versions doivent être vérifiées |

Les noms, caractéristiques et prix sont issus des catalogues officiels consultés à la date du cutoff : [OpenAI](https://openai.com/index/gpt-6-astra-next-generation-work/), [Anthropic](https://platform.claude.com/docs/en/models/overview), [Google](https://ai.google.dev/gemini-api/docs/models), [xAI](https://docs.x.ai/developers/models), [DeepSeek](https://api-docs.deepseek.com/updates/).

### 12.3 Familles open-weight ou localisables

| Famille | Atout principal | Pourquoi elle peut être choisie | Réserves |
|---|---|---|---|
| **Meta Llama 4** | Écosystème open-weight, multimodalité et nombreux runtimes | Large adoption, intégrations, choix de tailles et d'hébergeurs | Licence Llama Community, contraintes de déploiement et résultats dépendants du modèle exact |
| **Mistral Small 4 / Medium 3.5 / Large 3 / Ministral 3** | Famille européenne, modèles efficaces, certaines licences permissives | Small et Ministral pour coût/local, Large pour capacité multimodale | Les modèles et statuts changent rapidement ; ne pas confondre API et poids |
| **Qwen3 et Qwen3.8** | Très bon rapport capacité/localisation, code, multilingue et tool use | Versions de tailles variées, Apache 2.0 pour plusieurs checkpoints | Les résultats de modèles open-weight dépendent fortement du runtime, quantification et prompt template |
| **DeepSeek V3.2 et V4** | Raisonnement, code, efficacité et licences ouvertes pour certaines versions | Alternative open-weight et/ou API très compétitive | Les versions V4 exactes et la disponibilité des poids doivent être vérifiées séparément |
| **GLM** | Tool use et modèles open-weight performants sur certaines évaluations | BFCL montre que certains GLM ont un profil fort en function calling | Benchmark spécialisé, licence et modèle exact à inspecter |

Sources principales : [Meta Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Mistral Models](https://docs.mistral.ai/models), [Qwen3.8](https://github.com/QwenLM/Qwen3.8), [DeepSeek V4.1-Flash](https://www.deepseek.com/en/news/deepseek-v4-1-flash/).

### 12.4 Que disent les benchmarks ?

- **Artificial Analysis** fournit un index multidimensionnel, avec par exemple comparaison entre GPT-6 Astra et Gemini 3.8 Flash. Il mesure plusieurs évaluations, mais son score composite ne remplace pas un test sur vos données.
- **OSWorld** mesure l'action dans un environnement informatique réel. Le snapshot consulté place Claude Fable 5.1 et Claude Opus 5 très haut, mais le résultat appartient au couple modèle + environnement + nombre maximal d'étapes.
- **SWE-bench** mesure la résolution d'issues de code avec un agent. Les résultats montrent pourquoi il faut écrire `modèle + agent + version du harness + date`, plutôt que citer un score comme une propriété isolée du LLM.
- **BFCL** mesure function calling et tool use. Le classement publié en avril 2026 contient des modèles propriétaires et open-weight, avec des différences importantes entre coût, précision, multi-turn et hallucinations.
- **LiveCodeBench** vise une évaluation de code avec problèmes récents. Il est utile pour suivre le coding, mais son classement n'est pas un classement généraliste.

Références : [Artificial Analysis](https://artificialanalysis.ai/models/comparisons/gpt-6-astra-high-vs-gemini-3-8-flash), [OSWorld](https://os-world.github.io/), [SWE-bench](https://www.swebench.com/), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [LiveCodeBench](https://livecodebench.github.io/leaderboard.html).

### 12.5 Guide pratique par scénario

| Scénario | Première famille à tester | Pourquoi | Test indispensable |
|---|---|---|---|
| Raisonnement général difficile | GPT-6 Astra, Claude Fable/Opus 5, Gemini 3.1 Pro | Budget de raisonnement et capacités générales | Jeu de problèmes réels, coût par tâche réussie |
| Coding agentique | Claude Opus/Sonnet, GPT Codex ou équivalent, Gemini Flash/Pro, DeepSeek/Qwen | Boucles d'édition, outils et exécution | Même dépôt, même agent, tests et limites d'étapes |
| Computer use | Claude Fable/Opus 5, Gemini avec computer use, modèles spécialisés | Vision de l'écran et actions multi-étapes | OSWorld-like sur votre environnement, sécurité des actions |
| RAG privé | Un modèle fiable avec structured outputs et citations | La qualité vient surtout du retrieval, reranking et grounding | Recall@k, faithfulness et cas sans réponse |
| Haut volume faible coût | Gemini Flash, Haiku, petits Qwen/Mistral ou modèles locaux | Débit, prix et latence | p95 sous concurrence et coût par requête |
| Local, contrôle et confidentialité | Ministral, Qwen, Mistral, DeepSeek, Llama, GLM selon licence | Poids, quantification, déploiement interne | VRAM, licence, débit et qualité après quantification |
| Tool calling | Modèle avec support natif et bon score BFCL | Arguments et schémas fiables | BFCL-like sur vos outils, permissions et erreurs |

**Conclusion :** le « meilleur modèle » est celui qui minimise le coût total par tâche acceptée sous vos contraintes. Une démo ou un score isolé ne suffit pas.

---

## 13. Implémentation minimale

### 13.1 Un appel simple avec sortie structurée

````python
from pydantic import BaseModel

class Incident(BaseModel):
    severity: str
    components: list[str]
    needs_human_review: bool

# Pseudocode indépendant du fournisseur
response = client.responses.parse(
    model="model-id-versioned",
    input=[
        {"role": "system", "content": "Extrais uniquement les faits présents."},
        {"role": "user", "content": ticket_text},
    ],
    text_format=Incident,
)
incident = response.output_parsed
````

La validation Pydantic vérifie la forme. Elle ne prouve pas que la valeur est vraie. Pour cela, il faut des règles métier, des sources et éventuellement une revue humaine.

### 13.2 RAG minimal conceptuel

````python
def answer(question, vector_store, llm):
    candidates = vector_store.search(question, top_k=20)
    context = rerank(question, candidates)[:5]
    prompt = build_grounded_prompt(question, context)
    answer = llm.generate(prompt)
    return validate_citations(answer, context)
````

Testez séparément `search`, `rerank`, `build_grounded_prompt`, `generate` et `validate_citations`. Un pipeline monolithique rend les régressions difficiles à diagnostiquer.

### 13.3 Fine-tuning ou RAG ?

````text
Besoin = informations privées ou changeantes
→ RAG

Besoin = style, format ou comportement stable
→ SFT / PEFT après création d'un jeu d'évaluation

Besoin = connaissances distribuées très spécialisées
→ envisager continued pretraining, puis ré-alignement

Besoin = coût et latence
→ routing, distillation, quantification ou modèle plus petit
````

---

## 14. Pièges courants

### Architecture

- Croire que tous les Transformers sont des LLM génératifs.
- Confondre encoder bidirectionnel et decoder causal.
- Dire que l'attention « comprend » sans vérifier les représentations et le contexte.
- Confondre paramètres totaux et paramètres actifs dans un MoE.
- Croire que FlashAttention supprime le coût quadratique.

### Entraînement

- Comparer un modèle base et un modèle instruct comme s'ils avaient le même objectif.
- Fine-tuner sur peu de données sans jeu de validation.
- Utiliser du continued pretraining brut sur un modèle aligné sans tester l'instruction following.
- Croire que LoRA est une loss, alors que c'est une paramétrisation de la mise à jour.
- Optimiser une métrique proxy vulnérable au reward hacking.

### Application

- Commencer par un agent alors qu'un appel structuré suffit.
- Utiliser RAG sans mesurer le recall du retriever.
- Passer trop de documents au modèle au lieu de filtrer et reranker.
- Donner des outils puissants sans validation, permissions ou approbation.
- Confondre sortie JSON valide et réponse vraie.

### Évaluation

- Utiliser un leaderboard comme vérité générale.
- Comparer des modèles avec des prompts, budgets de raisonnement et harnesses différents.
- Mesurer uniquement la qualité et ignorer coût, latence, taux d'échec et sécurité.
- Ne pas tester les cas sans réponse, contradictoires ou adversariaux.
- Stocker des prompts et documents sensibles dans les traces sans politique de rétention.

---

## 15. Exercices d'auto-évaluation

### Questions de rappel

1. Pourquoi un decoder-only peut-il être entraîné en parallèle malgré une génération séquentielle ?
2. Quelles sont les formes de `Q`, `K`, `V` dans une multi-head attention ?
3. Quel est le rôle du masque causal ?
4. Pourquoi divise-t-on `QKᵀ` par `sqrt(d_k)` ?
5. Quelle différence entre MHA, MQA et GQA ?
6. Pourquoi le KV cache accélère-t-il le decode ?
7. Pourquoi quantifier les poids ne suffit-il pas à prédire la mémoire runtime ?
8. Quand préférer RAG à LoRA ?
9. Que mesure exactement un reward model ?
10. Pourquoi un score SWE-bench appartient-il aussi à l'agent et au harness ?

### Exercices de calcul

1. Pour `L=32`, `H_kv=8`, `d_h=128`, `S=8192`, BF16, calculez l'ordre de grandeur du KV cache d'une séquence.
2. Comparez la mémoire brute des poids d'un modèle 7B en BF16, INT8 et INT4.
3. Pour une couche `W` de forme `[4096, 4096]`, comparez le nombre de paramètres d'une mise à jour complète avec LoRA de rang `r=16`.
4. Estimez le coût d'une requête contenant 4 000 tokens d'entrée et 1 000 tokens de sortie pour deux modèles ayant des prix différents.

### Exercices de conception

1. Concevez un pipeline pour répondre à des questions sur une base documentaire privée avec citations.
2. Remplacez un agent multi-outils par l'architecture la plus simple possible, puis justifiez chaque outil conservé.
3. Construisez un jeu d'évaluation de 50 requêtes pour comparer trois modèles sur votre travail réel.
4. Définissez les conditions qui déclenchent une revue humaine.
5. Écrivez une fiche modèle contenant version, contexte, prix, licence, modalités, latence, coûts et date de mesure.

---

## 16. Fiche de révision en une page

```text
LLM = modèle de probabilité sur des tokens

Transformer :
  embeddings + positions
  → self-attention Q/K/V
  → FFN
  → résidus + normalisation
  → logits vocabulaire

Pré-entraînement : prédire le prochain token
Post-entraînement : apprendre le comportement utile

RAG : connaissance externe et citations
LoRA : mise à jour compacte des poids
Tool calling : le modèle propose, le programme valide et exécute
Agent : sélection dynamique d'étapes et outils

Serving : prefill, decode, KV cache, batching, quantification
Évaluation : tâche réelle, coût, latence, sécurité et robustesse
Choix modèle : pas de meilleur absolu, seulement un meilleur compromis
```

---

## 17. Sources et références

### Papiers fondateurs

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805), 2018
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020
- Lewis et al., [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401), 2020
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), 2022
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), 2023
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025

### Documentation technique

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [Hugging Face TRL](https://huggingface.co/docs/trl/)
- [vLLM](https://docs.vllm.ai/)
- [SGLang](https://docs.sglang.ai/)
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [OpenAI model catalogue](https://developers.openai.com/api/docs/models)
- [Anthropic model overview](https://platform.claude.com/docs/en/models/overview)
- [Google Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Mistral models](https://docs.mistral.ai/models)
- [DeepSeek changelog](https://api-docs.deepseek.com/updates/)
- [xAI models](https://docs.x.ai/developers/models)

### Évaluation et sécurité

- [Artificial Analysis](https://artificialanalysis.ai/)
- [SWE-bench](https://www.swebench.com/)
- [LiveCodeBench](https://livecodebench.github.io/)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [OSWorld](https://os-world.github.io/)
- [OWASP GenAI Security Project](https://genai.owasp.org/)
- [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

### Méthodologie des cours du dépôt

Ce cours reprend la structure utilisée dans les cours existants : métadonnées, prérequis, liens Zettelkasten, objectifs Bloom, TL;DR, intuition, théorie, équations, formes de tenseurs, schémas, implémentation minimale, tableaux de comparaison, guide de décision, pièges, points clés, exercices et sources. Les documents spécialisés restent conservés comme approfondissements plutôt que recopiés intégralement.

---

## 🔗 Liens connexes

- **Fondations deep learning** : [[../INDEX.md]]
- **Architecture BERT** : [[bert_architecture_variantes.md]]
- **RAG** : [[rag.md]]
- **Fine-tuning efficace** : [[peft_lora.md]]
- **Méthodes d'entraînement** : [[methodes_entrainement_llm.md]]
- **RLHF et apprentissage par renforcement** : [[apprentissage_renforcement_llm.md]]
- **Distillation** : [[distillation_llm.md]]
- **Serving** : [[../../03_infrastructure/serving_llm.md]]
- **Infrastructure IA** : [[../../03_infrastructure/infra_ia.md]]

> **Prochaine mise à jour recommandée** : conserver le tronc fondamental stable et mettre à jour uniquement la section « État de l'art » avec une nouvelle date de cutoff, les versions exactes, les prix, les licences et les benchmarks reproduits.
