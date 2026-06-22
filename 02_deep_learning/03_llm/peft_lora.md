# PEFT, LoRA et Fine-tuning Efficace des LLMs

## Table des Matières

1. [Introduction et Problème](#1-introduction-et-problème)
2. [LoRA : Low-Rank Adaptation](#2-lora--low-rank-adaptation)
3. [QLoRA : Quantization + LoRA](#3-qlora--quantization--lora)
4. [PEFT au sens large](#4-peft-au-sens-large)
5. [Implémentation Pratique](#5-implémentation-pratique)
6. [Alignment : SFT, DPO, PPO avec TRL](#6-alignment--sft-dpo-ppo-avec-trl)
7. [Guide de Choix et Comparaisons](#7-guide-de-choix-et-comparaisons)
8. [Sources et Références](#8-sources-et-références)

---

## 1. Introduction et Problème

### 1.1 Le coût du fine-tuning complet

Un fine-tuning **full fine-tuning** modifie l'intégralité des paramètres d'un modèle. Pour un LLM moderne, cela pose des problèmes majeurs :

| Modèle | Paramètres | VRAM (FP16) | VRAM entraînement (Adam) |
|--------|-----------|-------------|--------------------------|
| GPT-2 (117M) | 117M | ~0.2 Go | ~1 Go |
| LLaMA 7B | 7B | ~14 Go | ~60 Go |
| LLaMA 13B | 13B | ~26 Go | ~100 Go |
| LLaMA 70B | 70B | ~140 Go | ~500+ Go |

> L'optimiseur Adam stocke deux moments par paramètre. Un modèle 7B en FP32 avec Adam nécessite environ 4× la taille des poids rien que pour l'optimiseur.

**Autres problèmes du full fine-tuning :**
- **Catastrophic forgetting** : le modèle oublie ses capacités générales
- **Coût prohibitif** pour des expérimentations
- **Un modèle par tâche** : impossible en pratique de garder N copies complètes
- **Instabilité** : le modèle peut diverger facilement

### 1.2 L'intuition du PEFT

**PEFT (Parameter-Efficient Fine-Tuning)** désigne une famille de techniques qui cherchent à adapter un modèle pré-entraîné en n'entraînant qu'une **fraction infime** des paramètres.

Idée clé :

> Les adaptations utiles d'un LLM se produisent dans un sous-espace de faible dimension intrinsèque.

En d'autres termes : les mises à jour des poids lors d'un fine-tuning spécialisé peuvent être approximées par des matrices de faible rang.

---

## 2. LoRA : Low-Rank Adaptation

### 2.1 Principe mathématique

Lors d'un fine-tuning standard, on modifie les poids d'une couche :

$$W' = W_0 + \Delta W$$

où $$W_0 \in \mathbb{R}^{d \times k}$$ est la matrice de poids initiale (gelée) et $$\Delta W \in \mathbb{R}^{d \times k}$$ est la mise à jour entraînable.

**L'hypothèse LoRA** : $$\Delta W$$ est de rang intrinsèquement faible. On peut donc l'approximer par :

$$\Delta W = BA$$

avec :
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ → le **rang LoRA**

La sortie d'une couche linéaire devient alors :

$$h = W_0 x + \frac{\alpha}{r} BAx$$

où $$\alpha$$ est un **facteur de scaling** (hyperparamètre distinct de $$r$$).

**Nombre de paramètres entraînables :**

$$N_{LoRA} = r(d + k) \ll dk = N_{full}$$

**Exemple concret :**

Pour une couche d'attention $$W_q \in \mathbb{R}^{4096 \times 4096}$$ et $$r=16$$ :
- Full fine-tuning : $$4096 \times 4096 = 16,7M$$ paramètres
- LoRA : $$16 \times (4096 + 4096) = 131K$$ paramètres
- Ratio : **0,78% des paramètres d'origine**

### 2.2 Initialisation

L'initialisation est critique pour la stabilité :
- $$A$$ est initialisée avec une **distribution gaussienne** aléatoire
- $$B$$ est initialisée à **zéro**

Ainsi, au début de l'entraînement : $$\Delta W = BA = 0$$, ce qui préserve exactement le comportement du modèle de base.

### 2.3 Où appliquer LoRA ?

LoRA peut être appliqué à n'importe quelle couche linéaire. En pratique, on l'applique aux **matrices d'attention** du Transformer :

| Matrice | Description | Souvent ciblée ? |
|---------|-------------|-----------------|
| $$W_q$$ | Query | ✅ Oui |
| $$W_k$$ | Key | ✅ Oui |
| $$W_v$$ | Value | ✅ Oui |
| $$W_o$$ | Output projection | Parfois |
| $$W_{up}, W_{down}$$ | FFN | Parfois |

Le paramètre `target_modules` dans PEFT contrôle quelles couches sont ciblées.

### 2.4 Hyperparamètres clés

| Hyperparamètre | Rôle | Valeurs typiques |
|----------------|------|-----------------|
| `r` (rang) | Capacité des adaptateurs. Plus r est grand, plus le modèle peut s'adapter | 4, 8, 16, 32, 64 |
| `lora_alpha` | Facteur de scaling ($$\frac{\alpha}{r}$$ contrôle l'intensité) | Souvent = r ou 2r |
| `lora_dropout` | Régularisation | 0.0 à 0.1 |
| `target_modules` | Matrices ciblées | `q_proj,v_proj` ou all-linear |

**Règle pratique :**
- $r = 8$ ou $r = 16$ est un bon point de départ
- $\alpha = r$ simplifie le tuning (scaling factor = 1)
- Augmenter $r$ améliore la capacité d'adaptation mais augmente la VRAM

### 2.5 Fusion (merging)

Pour l'inférence, on peut **fusionner** les poids LoRA dans le modèle de base :

$$W_{merged} = W_0 + \frac{\alpha}{r} BA$$

Avantage : latence d'inférence identique au modèle de base, aucun surcoût LoRA au serving.

---

## 3. QLoRA : Quantization + LoRA

### 3.1 Principe

**QLoRA** combine deux techniques :
1. Le modèle de base est chargé en **4-bit NF4** (NormalFloat4, une quantization optimisée pour les distributions gaussiennes des poids)
2. Les adaptateurs LoRA restent entraînés en **BFloat16** pleine précision

```
Modèle de base (gelé, 4-bit NF4) ──── réduit la VRAM
          +
Adaptateurs LoRA (entraînables, BF16) ──── quelques Mo
```

### 3.2 Gains mémoire

| Configuration | VRAM pour LLaMA 7B |
|---------------|---------------------|
| Full FP16 + Adam | ~60 Go |
| LoRA FP16 | ~25 Go |
| QLoRA (4-bit) | **~6 Go** |

QLoRA a rendu possible le fine-tuning de LLMs sur une **seule GPU grand public** (RTX 3090, RTX 4090).

### 3.3 Double quantization

QLoRA introduit aussi la **double quantization** : les constantes de quantization sont elles-mêmes quantifiées, ce qui réduit encore l'empreinte mémoire de ~0.37 bits par paramètre.

### 3.4 Compromis

- **Vitesse d'entraînement** : légèrement plus lente qu'un LoRA FP16 (dequantization overhead)
- **Qualité** : légère dégradation possible mais souvent négligeable en pratique
- **VRAM** : très significativement réduite → c'est l'avantage principal

---

## 4. PEFT au sens large

### 4.1 Adapters

Les **Adapters** insèrent de petits modules entre les couches existantes du Transformer :

```
Input → [Layer Norm] → [Attention] → [Adapter] → [Layer Norm] → [FFN] → [Adapter] → Output
```

Un Adapter est typiquement composé de :
- Une projection descendante : $$\mathbb{R}^d \rightarrow \mathbb{R}^{r}$$
- Une non-linéarité
- Une projection montante : $$\mathbb{R}^{r} \rightarrow \mathbb{R}^{d}$$
- Une connexion résiduelle

**Avantages :**
- Bonne pour le multi-tâches (un adapter par tâche)
- Activable / désactivable facilement

**Inconvénients :**
- Légère latence supplémentaire (couches supplémentaires)
- Moins dominant que LoRA dans l'écosystème LLM actuel

### 4.2 Prefix Tuning

Au lieu de modifier les poids, on apprend des **tokens virtuels** ajoutés au début de chaque séquence.

Formellement, on ajoute un préfixe entraînable $$P = [p_1, ..., p_l]$$ aux séquences de clés et valeurs d'attention.

**Avantages :**
- Très peu de paramètres (seulement le préfixe)
- Le modèle de base reste strictement inchangé

**Inconvénients :**
- Performances généralement inférieures à LoRA
- Moins intuitif à déboguer

### 4.3 IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

IA³ apprend des **vecteurs de scaling** qui modifient les activations internes :

$$h' = l \odot h$$

où $$l \in \mathbb{R}^d$$ est un vecteur entraînable et $$\odot$$ est le produit élément par élément.

**Avantages :** Très peu de paramètres (~0.01% du modèle)
**Inconvénients :** Moins standard, moins utilisé en production

### 4.4 Comparaison PEFT

| Méthode | % paramètres entraînés | Performance relative | Latence inférence | Usage recommandé |
|---------|----------------------|---------------------|-------------------|-----------------|
| **Full fine-tuning** | 100% | Référence | ×1 | Gros budgets GPU |
| **LoRA** | 0.1% - 1% | ≈95-99% | ×1 (si merged) | **Choix par défaut** |
| **QLoRA** | 0.1% - 1% | ≈93-98% | ×1 (si merged) | GPU limités |
| **Adapters** | 3% - 5% | ≈95% | Légèrement + lente | Multi-tâches |
| **Prefix Tuning** | 0.01% - 0.1% | ≈85-90% | Légèrement + lente | Contraintes extrêmes |
| **IA³** | ~0.01% | ≈85-90% | ×1 | Recherche |

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

# 1. Charger le modèle de base
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Configurer LoRA
lora_config = LoraConfig(
    r=16,                          # rang LoRA
    lora_alpha=32,                 # alpha = 2*r => scaling factor = 2
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 3. Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# >> trainable params: 6,815,744 || all params: 8,037,900,288 || trainable%: 0.0848
```

### 5.3 QLoRA avec bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Config quantization 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # double quantization
    bnb_4bit_quant_type="nf4",         # NormalFloat4 optimal pour LLM
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Charger en 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Préparer pour l'entraînement kbit (cast des norms en FP32)
model = prepare_model_for_kbit_training(model)

# 4. Appliquer LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",   # cible toutes les couches linéaires
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

### 5.4 Entraînement avec SFTTrainer (TRL)

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="mon_dataset.jsonl")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",          # tracking W&B
    ),
)
trainer.train()
```

### 5.5 Fusion et sauvegarde

```python
from peft import PeftModel

# Charger le modèle de base + adaptateur
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "./output/checkpoint-best")

# Fusionner les poids LoRA dans la base
model = model.merge_and_unload()

# Sauvegarder le modèle fusionné (pour serving sans surcoût)
model.save_pretrained("./model_merged")
tokenizer.save_pretrained("./model_merged")
```

---

## 6. Alignment : SFT, DPO, PPO avec TRL

> Pour le détail complet de RLHF, DPO et PPO, voir [apprentissage_renforcement_llm.md](./apprentissage_renforcement_llm.md).

### 6.1 SFT (Supervised Fine-Tuning)

Le SFT est la première étape de tout pipeline d'alignment : on fine-tune le modèle sur des exemples de bonnes réponses.

```python
from trl import SFTTrainer

# Dataset format instruction-following
# {"prompt": "...", "completion": "..."}
# ou Alpaca-style: {"instruction": ..., "input": ..., "output": ...}

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # ...
)
```

### 6.2 DPO (Direct Preference Optimization)

DPO simplifie l'alignment en apprenant directement depuis des paires de préférences, **sans reward model séparé** :

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Dataset DPO :

```python
# Format attendu par DPOTrainer
{
    "prompt": "Quelle est la capitale de la France ?",
    "chosen": "La capitale de la France est Paris.",     # bonne réponse
    "rejected": "La capitale est Lyon, une grande ville." # mauvaise réponse
}
```

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,   # modèle de référence (SFT model gelé)
    tokenizer=tokenizer,
    train_dataset=dpo_dataset,
    args=DPOConfig(
        beta=0.1,           # régularisation KL (force de l'alignment)
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        max_prompt_length=512,
        max_length=1024,
    )
)
trainer.train()
```

**Avantages de DPO vs PPO :**
- Plus simple : pas de reward model séparé
- Plus stable : pas de RL
- Compatible LoRA directement
- Résultats souvent équivalents pour les cas standards

---

## 7. Guide de Choix et Comparaisons

### 7.1 Choisir son rang LoRA

| Objectif | Rang recommandé | Alpha recommandé |
|----------|----------------|-----------------|
| Adaptation style/format léger | 4 - 8 | = r |
| Spécialisation domaine | 16 - 32 | = r ou 2r |
| Fine-tuning capacités complexes | 64 - 128 | 64 - 128 |
| Budget mémoire très faible | 4 | 4 |

### 7.2 LoRA vs Fine-tuning selon le cas d'usage

| Besoin | LoRA | Full Fine-tuning |
|--------|------|-----------------|
| Budget GPU limité | ✅ Idéal | ❌ Trop coûteux |
| Plusieurs tâches/domaines | ✅ Un adaptateur par tâche | ❌ N copies |
| Apprentissage de capacités entièrement nouvelles | ⚠️ Limité | ✅ Mieux |
| Rollback vers le modèle de base | ✅ Trivial | ❌ Difficile |
| Déploiement agile | ✅ Swap d'adaptateurs | ❌ Lent |
| Qualité maximale absolue | ⚠️ Proche mais pas égal | ✅ Optimal |

### 7.3 RAG vs LoRA (pour enrichir les connaissances)

> Voir aussi [rag.md](./rag.md) pour le détail du RAG.

| Besoin | RAG | LoRA |
|--------|-----|------|
| Ajouter des faits changeants | ✅ Excellent | ❌ Coûteux à mettre à jour |
| Documents internes citables | ✅ Excellent | ❌ Pas de traçabilité |
| Modifier le comportement/style | ❌ Non | ✅ Excellent |
| Apprendre un format de sortie | ❌ Non | ✅ Excellent |
| Mise à jour en temps réel | ✅ Oui | ❌ Non |
| Coût de mise en œuvre initial | Moyen | Moyen |

**Recommandation pratique :** RAG pour les faits, LoRA pour le comportement. Les deux sont complémentaires.

### 7.4 Pipeline recommandé pour une entreprise

```
Base Model (Llama / Mistral / Qwen)
         │
         ▼
   SFT avec LoRA
   (données internes, format)
         │
         ▼
   DPO avec LoRA
   (paires préférences collectées)
         │
         ▼
   Merge LoRA + Serving vLLM
   (voir serving_llm.md)
         │
         ▼
   Monitoring & itération
   (W&B, feedbacks, A/B test)
```

---

## 8. Sources et Références

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning](https://arxiv.org/abs/2101.00190)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)
- [TRL documentation](https://huggingface.co/docs/trl)

---

## 🔗 Liens connexes

- [RAG →](./rag.md)
- [RLHF et Apprentissage par Renforcement →](./apprentissage_renforcement_llm.md)
- [Distillation LLM →](./distillation_llm.md)
- [Serving et Optimisation d'Inférence →](../../03_infrastructure/serving_llm.md)

---

*Dernière mise à jour : Juin 2026*
