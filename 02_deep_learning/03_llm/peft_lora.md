# PEFT, LoRA et Fine-tuning Efficace des LLMs

## Table des Matières

0. [Avant de commencer](#0-avant-de-commencer)
1. [Le Problème : pourquoi le fine-tuning complet est prohibitif](#1-le-problème--pourquoi-le-fine-tuning-complet-est-prohibitif)
2. [LoRA : Low-Rank Adaptation](#2-lora--low-rank-adaptation)
3. [QLoRA : Quantization + LoRA](#3-qlora--quantization--lora)
4. [PEFT au sens large : variantes et évolutions](#4-peft-au-sens-large--variantes-et-évolutions)
5. [Implémentation Pratique](#5-implémentation-pratique)
6. [Alignment : SFT et DPO avec TRL](#6-alignment--sft-et-dpo-avec-trl)
7. [Guide de Choix](#7-guide-de-choix)
8. [⚠️ Pièges Courants](#8-️-pièges-courants)
9. [🎯 Exercices d'Auto-évaluation](#9--exercices-dauto-évaluation)
10. [Sources et Références](#10-sources-et-références)

---

## 0. Avant de commencer

### Prérequis

- Architecture Transformer : attention multi-têtes, couches FFN, layer norm
- Gradient descent et backpropagation (niveau conceptuel)
- Notions de base GPU/VRAM
- Python et PyTorch (niveau débutant suffisant pour la section implémentation)

### Objectifs d'apprentissage *(taxonomie de Bloom)*

| Niveau | Objectif |
|--------|---------|
| **Comprendre** | Expliquer pourquoi LoRA réduit les paramètres entraînables via la décomposition de faible rang |
| **Analyser** | Comparer LoRA, QLoRA, DoRA, Adapters selon les contraintes GPU et de qualité |
| **Appliquer** | Implémenter un fine-tuning QLoRA sur un modèle 7B avec `peft` + `bitsandbytes` |
| **Évaluer** | Choisir entre RAG, LoRA et fine-tuning complet selon le cas d'usage |
| **Synthétiser** | Concevoir un pipeline SFT → DPO pour aligner un LLM interne |

### TL;DR en 3 phrases

> LoRA réduit de ~99% les paramètres entraînables en approximant les mises à jour de poids par des matrices de très faible rang. QLoRA combine cette approche avec la quantization 4-bit du modèle de base, rendant le fine-tuning accessible sur une seule GPU grand public. PEFT est la librairie Hugging Face qui implémente LoRA et ses variantes.

---

## 1. Le Problème : pourquoi le fine-tuning complet est prohibitif

### 1.1 Analogie

Imaginons que tu veuilles enseigner à un pianiste virtuose à jouer dans le style de Debussy. Deux approches :

- **Full fine-tuning** : réapprendre à jouer du piano depuis zéro. Coûteux, risque d'oublier les autres styles.
- **LoRA** : lui donner un set d'instructions stylistiques compactes. Il garde toutes ses compétences, et applique ces ajustements à la volée.

Le pianiste (modèle de base) reste intact. Seule une fine "couche d'adaptation" est entraînée.

### 1.2 Le coût du fine-tuning complet

Un fine-tuning complet modifie **tous** les paramètres du modèle. Pour un LLM moderne, cela pose des problèmes majeurs :

> **Conditions du tableau** : seq_len = 2048, batch_size = 1, gradient checkpointing désactivé, optimiseur Adam (stocke 2 moments par paramètre en FP32). Ces chiffres varient significativement avec ces paramètres : le gradient checkpointing peut réduire la VRAM d'entraînement de 60-70% au prix d'un ralentissement de ~20-30%.

| Modèle | Paramètres | VRAM inférence (BF16) | VRAM entraînement Adam (approx.) |
|--------|-----------|----------------------|----------------------------------|
| GPT-2 (117M) | 117M | ~0.2 Go | ~1 Go |
| LLaMA 3.1 8B | 8B | ~16 Go | ~60-80 Go |
| LLaMA 3.1 70B | 70B | ~140 Go | ~500+ Go |

**Autres problèmes du full fine-tuning :**
- **Catastrophic forgetting** : risque de dégradation des capacités générales sur les domaines absents du dataset de fine-tuning. LoRA réduit significativement ce risque car la base reste gelée, mais ne l'élimine pas entièrement (les adaptateurs peuvent eux-mêmes sur-fitter).
- **Coût prohibitif** pour l'expérimentation : chaque essai coûte des dizaines de GPU-heures
- **Un modèle par tâche** : stocker N copies complètes d'un 70B est impraticable en production
- **Instabilité** : les grands learning rates peuvent faire diverger le modèle

### 1.3 L'hypothèse fondamentale du PEFT

**PEFT (Parameter-Efficient Fine-Tuning)** repose sur une hypothèse validée empiriquement :

> *"Les tâches NLP peuvent être résolues dans un sous-espace de très faible dimension intrinsèque, bien plus petit que la dimensionnalité apparente des paramètres du modèle."*
>
> — Aghajanyan, Zettlemoyer & Gupta, **ACL 2021** *(Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning, arXiv:2012.13255)*

En pratique : les mises à jour utiles lors d'un fine-tuning spécialisé peuvent être approximées par des matrices de faible rang. PEFT exploite cette propriété pour n'entraîner qu'une fraction infime des paramètres.

---

> 📌 **Points clés — Section 1**
> - Le fine-tuning complet d'un 8B exige ~60-80 Go de VRAM (Adam, seq_len=2048), contre ~16 Go pour l'inférence
> - Les gains utiles lors du fine-tuning se produisent dans un sous-espace de faible dimension intrinsèque (Aghajanyan et al., ACL 2021)
> - LoRA réduit le nombre de paramètres entraînables de 100× à 10 000×, tout en atténuant (sans éliminer) le catastrophic forgetting

---

## 2. LoRA : Low-Rank Adaptation

### 2.1 Principe mathématique

Lors d'un fine-tuning standard, on modifie les poids d'une couche linéaire :

$$W' = W_0 + \Delta W$$

où $$W_0 \in \mathbb{R}^{d \times k}$$ est la matrice initiale **gelée** et $$\Delta W \in \mathbb{R}^{d \times k}$$ est la mise à jour entraînable.

**L'hypothèse LoRA** : $$\Delta W$$ est de rang intrinsèquement faible. On l'approxime par :

$$\Delta W = BA$$

avec :
- $$B \in \mathbb{R}^{d \times r}$$, initialisée à **zéro**
- $$A \in \mathbb{R}^{r \times k}$$, initialisée avec une distribution gaussienne aléatoire
- $$r \ll \min(d, k)$$ : le **rang LoRA**

La sortie d'une couche devient :

$$h = W_0 x + \frac{\alpha}{r} BAx$$

où $$\frac{\alpha}{r}$$ est le **facteur de scaling** ($$\alpha$$ est un hyperparamètre indépendant de $$r$$).

> **Pourquoi $$B = 0$$ à l'init ?** Pour que $$\Delta W = BA = 0$$ au départ : le modèle préserve exactement le comportement de la base et s'adapte progressivement. Si $$B$$ était aléatoire, le modèle serait perturbé dès le premier forward pass.

**Réduction des paramètres :**

$$N_{LoRA} = r(d + k) \ll dk = N_{full}$$

**Exemple concret** pour $$W_q \in \mathbb{R}^{4096 \times 4096}$$, $$r = 16$$ :
- Full fine-tuning : $$4096 \times 4096 = 16.8M$$ paramètres
- LoRA : $$16 \times (4096 + 4096) = 131K$$ paramètres → **0.78%**

### 2.2 Où appliquer LoRA ?

LoRA peut s'appliquer à toute couche linéaire. Les cibles usuelles :

| Matrice | Rôle | Recommandé |
|---------|------|-----------|
| $$W_q, W_k, W_v$$ | Queries/Keys/Values de l'attention | ✅ Toujours |
| $$W_o$$ | Output projection de l'attention | ✅ Souvent |
| $$W_{gate}, W_{up}, W_{down}$$ | Couches FFN | ✅ Pour rangs plus élevés |
| Embeddings | Lookup table | ❌ Rarement utile |

> **Conseil pratique** : cibler `"all-linear"` (paramètre PEFT) donne souvent de meilleurs résultats qu'une sélection manuelle, au prix d'une légère augmentation mémoire.

### 2.3 Hyperparamètres clés

| Hyperparamètre | Rôle | Valeurs typiques |
|----------------|------|-----------------|
| `r` (rang) | Capacité d'adaptation (taille des matrices A et B) | 8, 16, 32 |
| `lora_alpha` | Détermine le scaling $$\frac{\alpha}{r}$$ appliqué aux mises à jour | alpha=r → scaling=1 |
| `lora_dropout` | Régularisation sur les matrices LoRA | 0.0 à 0.1 |
| `target_modules` | Couches linéaires ciblées | `"all-linear"` ou liste explicite |

> **Sur `lora_alpha` et `r`** : ce qui compte, c'est le **ratio** $$\frac{\alpha}{r}$$, pas les valeurs absolues. `alpha=16, r=16` (scaling=1) est le paramétrage le plus simple à interpréter. `alpha=32, r=16` (scaling=2) amplifie l'effet LoRA, ce qui peut aider ou déstabiliser selon le cas.

**Guide pour choisir r :**

| Objectif | r recommandé |
|----------|-------------|
| Ajustement de style/format léger | 4 - 8 |
| Spécialisation domaine métier | 16 - 32 |
| Fine-tuning de capacités complexes | 64 |
| Budget VRAM très limité | 4 |

> ⚠️ Des rangs très élevés (r > 128) n'améliorent généralement pas la qualité et peuvent dégrader la généralisation. La littérature récente converge vers r ∈ [8, 64] pour la grande majorité des cas pratiques.

### 2.4 Fusion (merging) pour l'inférence

Pour l'inférence, on peut **fusionner** les poids LoRA dans le modèle de base :

$$W_{merged} = W_0 + \frac{\alpha}{r} BA$$

**Avantage** : latence identique au modèle de base, zéro surcoût LoRA au serving.
**Inconvénient** : perd la modularité (impossible de switcher d'adaptateur sans recharger le modèle de base).

---

> 📌 **Points clés — Section 2**
> - LoRA gèle $$W_0$$ et ajoute deux petites matrices A (gaussien) et B (zéro) par couche ciblée
> - Le scaling $$\frac{\alpha}{r}$$ contrôle l'intensité de l'adaptation, pas la capacité
> - Fusionner LoRA avant serving supprime tout surcoût de latence
> - r ∈ [8, 32] couvre la majorité des cas pratiques ; r > 128 est rarement justifié

---

## 3. QLoRA : Quantization + LoRA

### 3.1 Principe

**QLoRA** (Dettmers et al., 2023) combine deux techniques :
1. Le modèle de base est chargé en **4-bit NF4** (NormalFloat4 : distribution optimisée pour les poids LLM à distribution gaussienne)
2. Les adaptateurs LoRA restent entraînés en **BFloat16** pleine précision

```
Modèle de base (gelé, 4-bit NF4)        ← réduit la VRAM de ~4×
          +
Adaptateurs LoRA (entraînables, BF16)   ← quelques dizaines de Mo seulement
```

### 3.2 Gains mémoire

| Configuration | VRAM pour LLaMA 3.1 8B |
|---------------|------------------------|
| Full FP16 + Adam | ~60-80 Go |
| LoRA FP16 (base gelée) | ~25 Go |
| **QLoRA (base 4-bit NF4)** | **~6-8 Go** |

QLoRA a rendu le fine-tuning de LLMs accessible sur une **RTX 3090/4090** (24 Go VRAM).

### 3.3 Double quantization

QLoRA introduit la **double quantization** : les constantes de quantization (normalement en FP32, soit ~0.5 bits/paramètre) sont elles-mêmes quantifiées en 8-bit, réduisant encore l'empreinte de ~0.37 bits/paramètre.

### 3.4 Compromis

| Aspect | Impact |
|--------|--------|
| Vitesse d'entraînement | Légèrement plus lente qu'un LoRA FP16 (~10-20% overhead de dequantization) |
| Qualité finale | Légère dégradation possible, souvent négligeable pour les tâches standard |
| VRAM | Réduction très significative (~4× vs LoRA FP16) |

---

> 📌 **Points clés — Section 3**
> - QLoRA = base quantifiée 4-bit NF4 + adaptateurs LoRA en BF16
> - Permet de fine-tuner un 8B sur ~6-8 Go de VRAM (vs ~25 Go pour LoRA FP16)
> - La dégradation de qualité due à NF4 est faible pour la plupart des tâches standards

---

## 4. PEFT au sens large : variantes et évolutions

### 4.1 Adapters (Houlsby et al., ICML 2019)

Les Adapters insèrent de petits modules entre les couches du Transformer :

```
Input → [LN] → [Attention] → [Adapter] → [LN] → [FFN] → [Adapter] → Output
```

Un Adapter standard = projection $$d \to r$$ + non-linéarité + projection $$r \to d$$ + connexion résiduelle.

**Avantages :** bons pour le multi-tâche (un adapter par tâche, activable/désactivable).
**Inconvénients :** légère latence supplémentaire due aux couches séquentielles ajoutées ; moins utilisés que LoRA dans l'écosystème LLM actuel.

### 4.2 Prefix Tuning (Li & Liang, ACL 2021)

On apprend des tokens virtuels (vecteurs K/V entraînables) ajoutés au début de chaque séquence. Le modèle reste strictement inchangé.

**Avantages :** très peu de paramètres.
**Inconvénients :** performances généralement inférieures à LoRA ; instable sur les petits modèles.

### 4.3 IA³ (Liu et al., 2022)

Apprend des vecteurs de scaling $$l \in \mathbb{R}^d$$ qui modifient les activations : $$h' = l \odot h$$. Moins de 0.01% de paramètres entraînables.

### 4.4 DoRA — Weight-Decomposed LoRA (Liu et al., ICML 2024)

**DoRA** décompose les poids en composantes **magnitude** et **direction**, puis applique LoRA uniquement sur la composante directionnelle :

$$W = \underbrace{\frac{W}{\|W\|_c}}_{\text{direction (LoRA ici)}} \times \underbrace{\|W\|_c}_{\text{magnitude (scalaire entraînable)}}$$

**Résultats** : surpasse LoRA de manière consistante sur plusieurs benchmarks. Les gains sont plus marqués sur les tâches de raisonnement et de suivi d'instructions complexes.

### 4.5 LoRA+ (Hayou et al., 2024)

Modification simple et efficace : utiliser des **learning rates différents** pour A et B. La matrice B (qui projette dans l'espace de sortie du modèle) mérite un LR plus élevé que A.

**Gain** : +1 à 2% sur les benchmarks standards avec zéro coût supplémentaire et une modification minimale du code.

### 4.6 LoftQ (Liu et al., 2023)

Alternative à QLoRA : au lieu d'initialiser LoRA aléatoirement sur un modèle quantifié, LoftQ optimise conjointement la quantization et l'initialisation LoRA pour minimiser l'erreur d'approximation dès le départ.

**Utilité** : meilleur point de départ pour le fine-tuning, surtout sur les tâches complexes sensibles à la quantization.

### 4.7 Comparaison PEFT

| Méthode | % param. entraînés | Performance relative* | Latence inférence | Recommandation |
|---------|--------------------|-----------------------|-------------------|----------------|
| Full fine-tuning | 100% | Référence | ×1 | Gros budgets GPU |
| **LoRA** | 0.1% - 1% | Dépend fortement de la tâche | ×1 (si merged) | **Choix par défaut** |
| **QLoRA** | 0.1% - 1% | Légèrement < LoRA FP16 | ×1 (si merged) | GPU limités |
| **DoRA** | ~0.1% - 1% | ≥ LoRA (souvent meilleur) | ×1 (si merged) | Alternative recommandée à LoRA |
| **LoRA+** | 0.1% - 1% | Légèrement > LoRA standard | ×1 (si merged) | Upgrade quasi-gratuit de LoRA |
| **LoftQ** | 0.1% - 1% | ≥ QLoRA | ×1 (si merged) | QLoRA sur tâches complexes |
| Adapters | 3% - 5% | Comparable à LoRA | Légèrement + lente | Multi-tâches modulaires |
| Prefix Tuning | 0.01% - 0.1% | Inférieure à LoRA | Légèrement + lente | Contraintes mémoire extrêmes |
| IA³ | < 0.01% | Variable | ×1 | Recherche |

> *⚠️ Ces performances relatives n'ont **pas** de valeur numérique universelle. LoRA peut être à 99% du full fine-tuning sur une tâche de classification simple, mais significativement inférieur sur une tâche nécessitant une forte réorganisation des représentations. Toujours évaluer sur ta tâche spécifique.

---

> 📌 **Points clés — Section 4**
> - DoRA et LoRA+ sont des améliorations récentes de LoRA avec gains mesurés sur benchmarks
> - LoftQ est une alternative plus robuste à QLoRA pour les tâches complexes
> - Les performances relatives entre PEFT et full fine-tuning dépendent fortement de la tâche : ne pas généraliser des chiffres hors contexte
> - Adapters restent utiles pour le multi-tâche modulaire mais sont moins populaires que LoRA

---

## 5. Implémentation Pratique

### 5.1 Installation

```bash
pip install transformers peft bitsandbytes accelerate trl datasets wandb
```

### 5.2 LoRA avec PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. Charger le modèle de base en BF16
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 : meilleur que FP16 pour la stabilité des LLMs
    device_map="auto"             # distribue sur les GPUs disponibles
)

# 2. Configurer LoRA
lora_config = LoraConfig(
    r=16,                          # rang : 16 est un bon point de départ équilibré
    lora_alpha=16,                 # alpha=r → scaling factor=1, plus simple à interpréter
    target_modules="all-linear",   # cible toutes les couches linéaires (plus robuste que liste manuelle)
    lora_dropout=0.05,             # légère régularisation
    bias="none",                   # ne pas entraîner les biais (rarement utile avec LoRA)
    task_type=TaskType.CAUSAL_LM
)

# 3. Appliquer LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# >> trainable params: ~21M || all params: ~8.03B || trainable%: ~0.26%
```

### 5.3 QLoRA avec bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Configuration de la quantization 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # double quantization : ~0.37 bits/param en moins
    bnb_4bit_quant_type="nf4",         # NF4 : optimal pour distributions gaussiennes des poids LLM
    bnb_4bit_compute_dtype=torch.bfloat16  # les calculs restent en BF16 malgré le stockage 4-bit
)

# 2. Charger le modèle en 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# ⚠️ ÉTAPE OBLIGATOIRE avec QLoRA
# Cast les Layer Norms en FP32 pour stabiliser les gradients.
# Sans cette étape, les gradients peuvent exploser ou s'annuler silencieusement.
model = prepare_model_for_kbit_training(model)

# 3. Appliquer LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

### 5.4 Entraînement avec SFTTrainer (TRL)

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Format ChatML recommandé
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
dataset = load_dataset("json", data_files="mon_dataset.jsonl")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # batch effectif = 16
        warmup_ratio=0.05,               # ~5% des steps pour le warmup LR
        num_train_epochs=3,
        learning_rate=1e-4,              # 1e-4 typique pour LoRA ; 2e-4 peut être trop élevé
        fp16=False,
        bf16=True,                       # BF16 sur GPU Ampere+ (A100, H100, RTX 30xx/40xx)
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",     # activer pour détecter le surapprentissage
        load_best_model_at_end=True,
        report_to="wandb",
    ),
)
trainer.train()
```

### 5.5 Fusion et sauvegarde

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

# Recharger le modèle de base en précision complète pour merger proprement
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Charger l'adaptateur LoRA entraîné
model = PeftModel.from_pretrained(base_model, "./output/checkpoint-final")

# Fusionner les poids LoRA dans W₀ : W_merged = W₀ + (α/r) * BA
model = model.merge_and_unload()

# Sauvegarder le modèle fusionné pour serving sans surcoût
model.save_pretrained("./model_merged")
tokenizer.save_pretrained("./model_merged")
```

---

## 6. Alignment : SFT et DPO avec TRL

> Pour le détail complet de RLHF, PPO et reward modeling, voir [apprentissage_renforcement_llm.md](./apprentissage_renforcement_llm.md).

### 6.1 SFT (Supervised Fine-Tuning)

Première étape de tout pipeline d'alignment : fine-tuner le modèle sur des exemples de bonnes réponses au format instruction-following. Produit le "SFT model" qui sert de base pour DPO.

```python
# Dataset format ChatML
# {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
trainer = SFTTrainer(model=model, train_dataset=sft_dataset, ...)
```

### 6.2 DPO (Direct Preference Optimization)

DPO (Rafailov et al., NeurIPS 2023) apprend directement depuis des paires de préférences, **sans reward model séparé** :

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Dataset DPO :
```python
{
    "prompt": "Explique le gradient descent",
    "chosen": "Le gradient descent est un algorithme...",  # réponse préférée
    "rejected": "C'est un truc d'optimisation..."          # réponse moins bonne
}
```

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,          # modèle à aligner (SFT model + LoRA)
    ref_model=ref_model,  # SFT model gelé : sert de référence pour la régularisation KL
    tokenizer=tokenizer,
    train_dataset=dpo_dataset,
    args=DPOConfig(
        beta=0.1,            # régularisation KL : plus fort = plus proche du modèle de référence
        learning_rate=5e-5,  # plus faible qu'en SFT
        per_device_train_batch_size=2,
    )
)
```

### 6.3 DPO vs PPO : comparaison nuancée

| Critère | DPO | PPO (RLHF complet) |
|---------|-----|-------------------|
| Complexité | Simple (pas de reward model) | Élevée (4 modèles simultanés) |
| Stabilité | Bonne | Difficile à stabiliser |
| Coût | Moyen | Élevé |
| Tâches simples (style, chat) | ✅ Excellent | ✅ Excellent |
| Raisonnement complexe (math, code) | ⚠️ Peut sous-performer | ✅ Généralement meilleur |
| Recommandation pratique | Premier choix | Si DPO se révèle insuffisant |

> **Nuance (littérature 2024)** : plusieurs études (Xu et al., "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study", 2024 ; Guo et al., 2024) montrent que PPO avec un reward model bien entraîné surpasse DPO de manière significative sur les tâches de raisonnement complexe. DPO souffre d'un problème de **distribution shift** : il s'entraîne sur des paires statiques générées par d'autres modèles et n'explore pas activement ses propres sorties. Pour un assistant métier standard, DPO est généralement suffisant. Pour des capacités de raisonnement avancées (niveau MATH, compétitions de code), PPO reste la référence.

---

## 7. Guide de Choix

### 7.1 LoRA vs Full fine-tuning

| Besoin | LoRA | Full Fine-tuning |
|--------|------|-----------------|
| Budget GPU limité | ✅ Idéal | ❌ Trop coûteux |
| Plusieurs tâches/domaines | ✅ Un adaptateur par tâche | ❌ N copies complètes |
| Acquisition de capacités radicalement nouvelles | ⚠️ Limité | ✅ Plus adapté |
| Rollback vers le modèle de base | ✅ Trivial | ❌ Difficile |
| Déploiement agile (swap d'adaptateurs) | ✅ | ❌ Lourd |
| Qualité absolue maximale | ⚠️ Dépend de la tâche | ✅ Optimal |

### 7.2 RAG vs LoRA

> Voir aussi [rag.md](./rag.md)

| Besoin | RAG | LoRA |
|--------|-----|------|
| Connaissances factuelles changeantes | ✅ | ❌ |
| Documents internes citables avec sources | ✅ | ❌ |
| Modification du style ou du comportement | ❌ | ✅ |
| Format de sortie spécifique et stable | ❌ | ✅ |
| Mise à jour en temps réel | ✅ | ❌ |
| Réduction des hallucinations | ✅ (sur faits) | ⚠️ (partiel) |

**Règle pratique : RAG pour les faits, LoRA pour le comportement. Les deux sont complémentaires.**

### 7.3 Pipeline recommandé pour une entreprise

```
Base Model open-source (Llama / Mistral / Qwen)
         │
         ▼
   SFT avec LoRA
   (données instruction internes, format de sortie attendu)
         │
         ▼
   DPO avec LoRA
   (paires préférences collectées via feedbacks utilisateurs)
         │
         ▼
   Merge LoRA → Serving vLLM
   (voir serving_llm.md)
         │
         ▼
   Monitoring & itération
   (W&B, feedbacks, A/B test, win rate)
```

---

## 8. ⚠️ Pièges Courants

### Fine-tuning / LoRA

**Piège 1 : Oublier `prepare_model_for_kbit_training()` avec QLoRA**
Sans cette étape, les Layer Norms restent en 4-bit → gradients instables ou silencieusement nuls. **Appeler cette fonction avant `get_peft_model()`**, toujours.

**Piège 2 : Confondre `lora_alpha` avec un learning rate**
`lora_alpha` ne contrôle pas la vitesse d'apprentissage mais l'**amplitude** des mises à jour LoRA. Augmenter alpha amplifie l'effet, ce qui peut déstabiliser l'entraînement. Commencer avec `alpha = r`.

**Piège 3 : Noms de couches incorrects selon le modèle**
Sur Mistral, Qwen ou Gemma, les noms des couches diffèrent de LLaMA. `"all-linear"` est plus robuste que lister manuellement `["q_proj", "v_proj", ...]`.

**Piège 4 : Learning rate trop élevé**
`2e-4` est souvent trop élevé pour LoRA et peut dégrader le modèle. Commencer par `1e-4` ou `5e-5`, surtout pour DPO.

**Piège 5 : Absence d'early stopping sur petit dataset**
Avec quelques centaines d'exemples, le surapprentissage est rapide. Utiliser `evaluation_strategy="epoch"` et `load_best_model_at_end=True`.

**Piège 6 : Rang trop élevé sans justification**
r=256 n'est pas "mieux que" r=32. Des rangs élevés peuvent même dégrader la généralisation et augmentent la VRAM inutilement.

### DPO

**Piège 7 : Beta trop faible**
`beta=0.01` laisse le modèle trop libre de s'éloigner de la référence → effondrement de la diversité ou réponses dégénérées. Commencer avec `beta=0.1`.

**Piège 8 : Dataset chosen/rejected de mauvaise qualité**
DPO est très sensible à la qualité des paires. Des "rejected" qui ne sont pas clairement pires que les "chosen" polluent le signal d'apprentissage. Vérifier manuellement un échantillon avant d'entraîner.

---

## 9. 🎯 Exercices d'Auto-évaluation

### Questions de révision (retrieval practice)

1. Pourquoi initialise-t-on $$B = 0$$ dans LoRA et pas $$A$$ ?
2. Quelle est la différence de rôle entre `r` et `lora_alpha` ?
3. Pour `r=8` et `lora_alpha=16`, quel est le facteur de scaling appliqué aux mises à jour LoRA ?
4. Dans quel cas pratique DPO peut-il sous-performer PPO ?
5. Quelle étape de code est obligatoire pour QLoRA et souvent oubliée ?
6. Quelle est la différence principale entre DoRA et LoRA ?

### Exercice de calcul

Calculer le nombre de paramètres entraînables pour un fine-tuning LoRA d'un LLaMA 3.1 8B, sachant :
- 32 couches Transformer
- Matrices ciblées : $$W_q, W_v$$ (dimensions $$4096 \times 4096$$ chacune)
- $$r = 8$$

*Solution attendue :* $$32 \times 2 \times 8 \times (4096 + 4096) = 32 \times 2 \times 8 \times 8192 = 4{,}194{,}304 \approx 4.2M$$ paramètres, soit ~0.05% du modèle.

<details>
<summary>Réponses aux questions de révision</summary>

1. Pour que $$\Delta W = BA = 0$$ à l'initialisation, préservant exactement le comportement du modèle de base. Si B était aléatoire, le modèle serait perturbé dès le premier forward pass.
2. `r` contrôle la **capacité** (taille des matrices A et B). `lora_alpha` contrôle l'**amplitude** des mises à jour via le scaling $$\frac{\alpha}{r}$$.
3. $$\frac{\alpha}{r} = \frac{16}{8} = 2$$. Les mises à jour LoRA sont amplifiées d'un facteur 2.
4. Sur des tâches de raisonnement complexe (mathématiques, code compétitif) où un reward model bien entraîné fourni un signal plus riche et plus exploratoire. DPO souffre de distribution shift.
5. `prepare_model_for_kbit_training(model)` — cast les Layer Norms en FP32 pour stabiliser les gradients.
6. DoRA décompose les poids en magnitude et direction, et n'applique LoRA qu'à la composante directionnelle. Cette décomposition produit des gains mesurés sur les benchmarks de raisonnement.

</details>

---

## 10. Sources et Références

### Papers fondateurs

- **LoRA** : Hu et al., 2021 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA** : Dettmers et al., NeurIPS 2023 — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Intrinsic Dimensionality** : Aghajanyan, Zettlemoyer & Gupta, ACL 2021 — [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- **DoRA** : Liu et al., ICML 2024 — [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- **LoRA+** : Hayou et al., 2024 — [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
- **LoftQ** : Liu et al., 2023 — [arXiv:2310.08659](https://arxiv.org/abs/2310.08659)
- **DPO** : Rafailov et al., NeurIPS 2023 — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **DPO vs PPO** : Xu et al., 2024 — "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study"
- **Prefix Tuning** : Li & Liang, ACL 2021 — [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- **Adapters** : Houlsby et al., ICML 2019 — [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)

### Documentation technique

- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 🔗 Liens connexes

- [RAG →](./rag.md)
- [RLHF et Apprentissage par Renforcement (PPO, reward modeling) →](./apprentissage_renforcement_llm.md)
- [Distillation LLM →](./distillation_llm.md)
- [Serving et Optimisation d'Inférence →](../../03_infrastructure/serving_llm.md)

---

*Dernière mise à jour : Juin 2026*
