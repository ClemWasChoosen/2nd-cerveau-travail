# Serving et Optimisation d'Inférence des LLMs

## Table des Matières

1. [Introduction et Problèmes de l'Inférence LLM](#1-introduction-et-problèmes-de-linférence-llm)
2. [Concepts Fondamentaux](#2-concepts-fondamentaux)
3. [vLLM et PagedAttention](#3-vllm-et-pagedattention)
4. [Hugging Face TGI](#4-hugging-face-tgi)
5. [TensorRT-LLM](#5-tensorrt-llm)
6. [llama.cpp et GGUF](#6-llamacpp-et-gguf)
7. [Quantization](#7-quantization)
8. [FlashAttention](#8-flashattention)
9. [Speculative Decoding](#9-speculative-decoding)
10. [Continuous Batching](#10-continuous-batching)
11. [Guide de Choix et Déploiement](#11-guide-de-choix-et-déploiement)
12. [Sources et Références](#12-sources-et-références)

---

## 1. Introduction et Problèmes de l'Inférence LLM

### 1.1 Pourquoi l'inférence LLM est difficile

Contrairement à un modèle ML classique (résultat en <10ms, coût quasi nul), un LLM présente des propriétés très particulières :

| Propriété | ML Classique | LLM |
|-----------|-------------|-----|
| Latence | < 10ms | 100ms - plusieurs secondes |
| VRAM nécessaire | Quelques Mo | Plusieurs Go |
| Génération | Prédiction unique | Autoregressive token par token |
| Parallélisme | Facile | Complexe (dépendance séquentielle) |
| Coût GPU | Faible | Élevé |

> Voir [infra_ia.md](./infra_ia.md) pour le contexte général sur l'infrastructure IA.

### 1.2 La génération autoregressive

Un LLM génère du texte **token par token** : chaque nouveau token dépend de tous les précédents.

$$P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t | w_1, ..., w_{t-1})$$

**Conséquence :** impossible de paralléliser la génération d'une séquence. Le modèle doit faire T passes forward pour T tokens.

### 1.3 Le KV Cache

Pour éviter de recalculer tout le contexte à chaque token, les moteurs d'inférence stockent un **KV cache** (Key-Value cache) : les tenseurs K et V de chaque couche d'attention, calculés pour tous les tokens précédents.

**Taille du KV cache :**

$$\text{KV cache} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times \text{dtype\_bytes}$$

Pour LLaMA 3.1 8B (32 layers, 32 heads, dim 128, FP16) avec une séquence de 8192 tokens :
$$= 2 \times 32 \times 32 \times 128 \times 8192 \times 2 \approx 8.6 \text{ Go}$$

**Le problème :** avec de nombreuses requêtes simultanées et des longueurs variables, la gestion de ce cache est un goulot d'étranglement majeur.

---

## 2. Concepts Fondamentaux

### 2.1 Métriques de performance

Avant de choisir un serveur d'inférence, il faut savoir mesurer :

| Métrique | Définition | Unité |
|----------|-----------|-------|
| **TTFT** | Time To First Token | ms |
| **TPOT** | Time Per Output Token | ms/token |
| **Throughput** | Tokens générés par seconde | tokens/s |
| **Latence P50/P95/P99** | Percentiles de latence | ms |
| **MFU** | Model FLOPs Utilization | % |
| **Concurrence** | Requêtes simultanées supportées | n |

### 2.2 Parallélisme d'inférence

Pour servir de grands modèles, plusieurs stratégies de parallélisme existent :

| Type | Description | Quand l'utiliser |
|------|-------------|-----------------|
| **Tensor Parallelism** | Découpe les tenseurs entre GPU | Modèles > VRAM d'un GPU |
| **Pipeline Parallelism** | Distribue les couches entre GPU | Très grands modèles |
| **Data Parallelism** | Copies du modèle sur plusieurs GPU | Scaling du throughput |

### 2.3 Sampling strategies

Les paramètres de génération affectent la qualité et la vitesse :

| Paramètre | Rôle | Valeur typique |
|-----------|------|----------------|
| `temperature` | Créativité (0 = déterministe) | 0.0 - 1.0 |
| `top_p` | Nucleus sampling | 0.9 - 0.95 |
| `top_k` | Top-K sampling | 20 - 50 |
| `max_new_tokens` | Longueur max de la réponse | 256 - 4096 |

---

## 3. vLLM et PagedAttention

### 3.1 Le problème résolu

Les serveurs d'inférence classiques allouent de la mémoire **statique et contiguë** pour le KV cache. Problèmes :
- **Fragmentation** : des blocs mémoire sont réservés mais partiellement utilisés
- **Gaspillage** : la longueur maximale est réservée dès le départ
- **Faible concurrence** : peu de requêtes en parallèle

**vLLM** résout ce problème avec **PagedAttention**.

### 3.2 PagedAttention

PagedAttention s'inspire de la **mémoire virtuelle** des systèmes d'exploitation (paging).

```
Gestion classique :
┌─────────────────────────────┐
│ Requête 1 : [████████░░░░] │  ← mémoire réservée mais partiellement utilisée
│ Requête 2 : [████░░░░░░░░] │
│ Requête 3 : [██████████░░] │
└─────────────────────────────┘

PagedAttention (vLLM) :
┌────────────────────────────────────────┐
│ Page 1 │ Page 2 │ Page 3 │ Page 4 ... │
│  R1    │  R1    │  R2    │  R1        │  ← pages allouées dynamiquement
│  R3    │  R2    │  R3    │  R3        │
└────────────────────────────────────────┘
```

**Principe :**
- Le KV cache est divisé en **blocs (pages) de taille fixe**
- L'allocation est **dynamique** : on alloue des pages au fur et à mesure
- Différentes requêtes peuvent **partager des pages** (ex. même préfixe système)
- Plus de fragmentation : chaque page est soit pleine, soit la dernière d'une séquence

**Résultat :** utilisation mémoire quasi parfaite, beaucoup plus de requêtes simultanées.

### 3.3 Gains de vLLM

Les benchmarks de l'article original montrent :

- **2-4× plus de throughput** vs HuggingFace Transformers naïf
- **~20× plus de requêtes simultanées** possibles
- **Compatible OpenAI API** : migration triviale

### 3.4 Installation et usage

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# Charger le modèle
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,   # nombre de GPUs
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9
)

# Générer
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Explique le machine learning en 3 phrases."], sampling_params)
print(outputs[0].outputs[0].text)
```

**Serveur OpenAI-compatible :**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port 8000
```

### 3.5 Fonctionnalités avancées

- **Multi-LoRA serving** : charger et switcher dynamiquement entre plusieurs adaptateurs LoRA
- **Prefix caching** : partage automatique des KV des préfixes système identiques
- **Speculative decoding** : support intégré
- **Quantization** : AWQ, GPTQ, FP8 supportés
- **Vision models** : LLaVA, Qwen-VL, etc.

---

## 4. Hugging Face TGI

### 4.1 Présentation

**Text Generation Inference (TGI)** est le serveur d'inférence officiel de Hugging Face. Il est utilisé en production sur Hugging Face Inference Endpoints.

### 4.2 Fonctionnalités clés

- Continuous batching
- Token streaming
- Tensor parallelism
- Quantization (bitsandbytes, GPTQ, AWQ)
- FlashAttention intégré
- OpenAI-compatible API
- Intégration Hub HuggingFace

### 4.3 Lancement via Docker

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --num-shard 1 \
  --dtype bfloat16 \
  --max-input-length 4096 \
  --max-total-tokens 8192
```

### 4.4 Usage Python

```python
from huggingface_hub import InferenceClient

client = InferenceClient(base_url="http://localhost:8080")

response = client.text_generation(
    "Explique le gradient boosting.",
    max_new_tokens=256,
    temperature=0.7,
    stream=True
)
for chunk in response:
    print(chunk, end="", flush=True)
```

---

## 5. TensorRT-LLM

### 5.1 Présentation

**TensorRT-LLM** est le framework NVIDIA pour optimiser l'inférence des LLMs sur GPU NVIDIA. Il opère à un niveau bien plus bas que vLLM ou TGI.

### 5.2 Optimisations appliquées

- **Kernel fusion** : fusion des opérations pour réduire les transferts mémoire
- **In-flight batching** : continuous batching natif
- **Quantization** : INT8, FP8, INT4 AWQ
- **Tensor/Pipeline parallelism**
- **Paged KV cache** (similaire à vLLM)
- **Speculative decoding**

### 5.3 Pipeline de compilation

TensorRT-LLM nécessite une **compilation du modèle** avant le serving :

```python
# 1. Conversion du modèle en format TensorRT-LLM
import tensorrt_llm

# Construction du moteur optimisé
# (spécifique au GPU cible, taille de batch, longueur max...)
engine = tensorrt_llm.build(
    model="llama-3.1-8b",
    max_batch_size=32,
    max_input_len=2048,
    max_output_len=512,
    dtype="float16",
    quantization="awq"  # ou "fp8", "int8"
)
engine.save("./trt_engine")

# 2. Serving avec Triton Inference Server
# (voir section 11.3)
```

### 5.4 Comparaison avec vLLM

| Critère | vLLM | TensorRT-LLM |
|---------|------|--------------|
| Facilité de démarrage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Performance brute | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Portabilité | GPU NVIDIA + AMD | NVIDIA uniquement |
| Compilation nécessaire | Non | Oui |
| Support de modèles | Très large | Large, mais parfois limité |
| Production rapide | ✅ | Plus long |
| Optimisations NVIDIA | Bonnes | Maximales |

**Recommandation :** commencer avec vLLM. Migrer vers TensorRT-LLM si les 5-20% de performance supplémentaire justifient la complexité.

---

## 6. llama.cpp et GGUF

### 6.1 Présentation

**llama.cpp** est une implémentation C++ des LLMs qui permet de les exécuter localement, sur CPU ou GPU, à partir de modèles **quantifiés au format GGUF**.

### 6.2 Format GGUF

GGUF (GPT-Generated Unified Format) est un format binaire qui contient :
- Les poids du modèle (quantifiés)
- Les métadonnées du modèle
- Les paramètres de tokenization

**Niveaux de quantization GGUF courants :**

| Quantization | Bits/param | VRAM 7B | Qualité |
|-------------|-----------|---------|---------|
| Q2_K | ~2.6 | ~3.1 Go | Médiocre |
| Q4_K_M | ~4.8 | ~5.0 Go | Bon compromis |
| Q5_K_M | ~5.7 | ~5.8 Go | Bon |
| Q6_K | ~6.6 | ~6.6 Go | Très bon |
| Q8_0 | 8.0 | ~8.5 Go | Excellent |
| F16 | 16.0 | ~14 Go | Référence |

### 6.3 Usage

```bash
# Télécharger un modèle GGUF depuis HuggingFace
# Ex: llama3-8b-q4_k_m.gguf

# Lancer une inférence locale
./llama-cli \
    -m ./models/llama3-8b-q4_k_m.gguf \
    -n 512 \
    -p "Explique le machine learning :"

# Lancer un serveur HTTP local
./llama-server \
    -m ./models/llama3-8b-q4_k_m.gguf \
    --port 8080 \
    -ngl 35   # couches sur GPU
```

**Via Python (avec llama-cpp-python) :**

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama3-8b-q4_k_m.gguf",
    n_ctx=4096,      # taille du contexte
    n_gpu_layers=35, # couches sur GPU (0 = CPU only)
)

output = llm("Explique les réseaux de neurones:", max_tokens=256)
print(output["choices"][0]["text"])
```

### 6.4 Cas d'usage

| Cas | llama.cpp adapté ? |
|-----|-------------------|
| Machine personnelle / laptop | ✅ Idéal |
| Application offline / edge | ✅ Idéal |
| Confidentialité maximale | ✅ Idéal |
| Apple Silicon | ✅ Excellent (Metal) |
| Production haute concurrence | ❌ Préférer vLLM/TGI |
| Entraînement | ❌ Non |

---

## 7. Quantization

### 7.1 Principe

La **quantization** réduit la précision numérique des poids, des activations, ou des deux.

**Formats courants :**

| Format | Bits | Description |
|--------|------|-------------|
| FP32 | 32 | Pleine précision, référence |
| BF16 | 16 | Float 16 brain (mieux que FP16 pour les LLMs) |
| FP16 | 16 | Float 16 standard |
| INT8 | 8 | Entier 8-bit |
| FP8 | 8 | Float 8-bit (H100) |
| INT4 / NF4 | 4 | Entier ou NormalFloat 4-bit |

**Gains mémoire pour un modèle 7B :**

$$\text{VRAM} \approx N_{params} \times \frac{\text{bits}}{8} \text{ octets}$$

| Précision | Calcul | VRAM 7B |
|-----------|--------|---------|
| FP32 | 7B × 4 | ~28 Go |
| FP16/BF16 | 7B × 2 | ~14 Go |
| INT8 | 7B × 1 | ~7 Go |
| INT4 | 7B × 0.5 | ~3.5 Go |

### 7.2 Méthodes de quantization

#### Post-Training Quantization (PTQ)

Quantization **après** entraînement, sans accès aux données d'entraînement (ou avec un petit dataset de calibration).

**GPTQ** : quantization par blocs avec compensation des erreurs via la matrice hessienne.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  # granularité de la quantization
    damp_percent=0.1
)

model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(examples)  # dataset de calibration
model.save_quantized("./model_gptq")
```

**AWQ (Activation-Aware Quantization)** : préserve les poids les plus importants identifiés par l'analyse des activations. Souvent meilleure qualité que GPTQ.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./model_awq")
```

### 7.3 Quantization dynamique vs statique

| Type | Quand | Avantage |
|------|-------|----------|
| **Statique** | Activations quantifiées avec calibration | Vitesse maximale |
| **Dynamique** | Activations quantifiées à la volée | Plus flexible |
| **Weights-only** | Seuls les poids sont quantifiés | Simple, bon compromis |

### 7.4 Impact sur la qualité

Règle générale (pour des LLMs bien entraînés) :

- **FP8 / BF16** : dégradation négligeable
- **INT8** : dégradation très faible (<1%)
- **INT4 / AWQ** : dégradation légère (1-3%)
- **INT4 / GPTQ naïf** : dégradation modérée
- **INT2/3** : dégradation significative

---

## 8. FlashAttention

### 8.1 Problème de l'attention standard

L'attention classique du Transformer nécessite de matérialiser la **matrice d'attention complète** en mémoire :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Pour une séquence de longueur $$n$$, cette matrice est de taille $$n \times n$$. La complexité mémoire est donc **$$O(n^2)$$**, ce qui devient très coûteux pour les longs contextes.

De plus, le bottleneck est la **mémoire HBM** (High Bandwidth Memory) du GPU : les aller-retours entre SRAM (très rapide mais petite) et HBM (plus lente mais grande) sont coûteux.

### 8.2 Solution FlashAttention

FlashAttention réorganise le calcul pour minimiser les accès HBM :

1. **Tiling** : découpe Q, K, V en blocs qui tiennent en SRAM
2. **Fusion des kernels** : le softmax est calculé en ligne (online softmax) en même passe
3. **Pas de matérialisation** de la matrice $$n \times n$$

**Résultats :**
- Complexité mémoire réduite de $$O(n^2)$$ à $$O(n)$$
- 2-4× plus rapide que l'attention standard
- Permettre des contextes beaucoup plus longs

### 8.3 Versions

| Version | Améliorations | Support |
|---------|--------------|---------|
| FlashAttention 1 | Base, tiling, io-awareness | A100 |
| FlashAttention 2 | Meilleur parallélisme, 2× plus rapide | Ampere+ |
| FlashAttention 3 | H100 optimisé, FP8, speculative | Hopper (H100) |

### 8.4 Intégration

```python
# Activé automatiquement dans Transformers récent
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="flash_attention_2",  # ← activer FA2
    torch_dtype=torch.bfloat16,
)
```

FlashAttention est déjà intégré dans : vLLM, TGI, TensorRT-LLM, PyTorch >= 2.0 (via `scaled_dot_product_attention`).

---

## 9. Speculative Decoding

### 9.1 Principe

La génération autoregressive est séquentielle par nature. Le speculative decoding accélère l'inférence en utilisant **deux modèles** :

1. Un **draft model** (petit, rapide) propose $$\gamma$$ tokens
2. Le **target model** (grand) vérifie ces propositions en **un seul passage forward parallèle**

```
Draft model (ex: LLaMA 68M) :
  prompt → [token1, token2, token3, token4] (4 tokens proposés en 4 passes)

Target model (ex: LLaMA 70B) :
  prompt + draft → [accepté, accepté, rejeté] (1 seule passe, mais 3 tokens produits)
```

**Condition d'acceptation** (pour garder l'exactitude de la distribution) :

Pour un token proposé $$x$$ avec probabilité draft $$q(x)$$ et target $$p(x)$$ :
$$\text{Accepter avec probabilité} = \min\left(1, \frac{p(x)}{q(x)}\right)$$

Si rejeté, on resample depuis la distribution corrigée $$p - q$$ (normalisée).

**Résultat théorique :** la distribution finale est exactement celle du target model, mais plus vite.

### 9.2 Gains

- **1.5× à 3× d'accélération** selon le taux d'acceptation et le ratio de taille
- Particulièrement efficace pour des outputs prévisibles (code, JSON, résumés)
- Moins efficace pour des outputs très créatifs

### 9.3 Variantes

| Variante | Description |
|---------|-------------|
| **Speculative decoding** | Draft model séparé |
| **Medusa** | Têtes prédictives multiples sur le même modèle |
| **EAGLE** | Draft model entraîné à partir du target |
| **Lookahead decoding** | N-grams du contexte pour les propositions |

---

## 10. Continuous Batching

### 10.1 Problème du batching statique

En batching classique, on attend d'avoir un batch complet avant de lancer l'inférence. Problèmes :
- Séquences de longueurs très différentes → padding = gaspillage
- Si une séquence finit avant les autres, le GPU attend

### 10.2 Continuous Batching (Iteration-level scheduling)

Le continuous batching traite les requêtes **itération par itération** (token par token). Dès qu'une séquence se termine, une nouvelle peut prendre sa place dans le batch.

```
Batch statique :
Iter 1: [R1_t1, R2_t1, R3_t1]
Iter 2: [R1_t2, R2_t2, R3_t2]
Iter 3: [R1_t3, ------, R3_t3]  ← R2 finie mais slot gaspillé
Iter 4: [R1_t4, ------, R3_t4]

Continuous batching :
Iter 1: [R1_t1, R2_t1, R3_t1]
Iter 2: [R1_t2, R2_t2, R3_t2]
Iter 3: [R1_t3, R4_t1, R3_t3]  ← R2 finie, R4 démarre immédiatement
Iter 4: [R1_t4, R4_t2, R5_t1]  ← R3 finie, R5 démarre
```

**Impact** : bien meilleure utilisation GPU, throughput multiplié (typiquement 5-10× vs serving naïf).

Tous les serveurs modernes (vLLM, TGI, TensorRT-LLM) implémentent le continuous batching.

---

## 11. Guide de Choix et Déploiement

### 11.1 Comparaison globale

| Critère | vLLM | TGI | TensorRT-LLM | llama.cpp |
|---------|------|-----|-------------|----------|
| **Throughput** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Facilité** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU requis** | NVIDIA / AMD | NVIDIA | NVIDIA | CPU ou GPU |
| **Multi-LoRA** | ✅ | ❌ | ✅ | ❌ |
| **Local/edge** | ❌ | ❌ | ❌ | ✅ |
| **Production** | ✅ | ✅ | ✅ | Limité |
| **API OpenAI compat.** | ✅ | ✅ | Via Triton | ✅ |

### 11.2 Arbres de décision

**Choisir le bon serveur :**

```
Déploiement local / laptop ?
    ├── Oui → llama.cpp
    └── Non
         │
         ├── Stack HuggingFace existante ?
         │       ├── Oui → TGI
         │       └── Non
         │               │
         │               ├── Optimisation NVIDIA max. ?
         │               │       ├── Oui → TensorRT-LLM
         │               │       └── Non → vLLM (choix par défaut)
```

**Optimiser les coûts d'inférence :**

```
Budget VRAM suffisant ?
    ├── Oui (FP16) → Pas de quantization nécessaire
    └── Non
         │
         ├── Légère perte de qualité acceptable ?
         │       ├── Oui → INT4/AWQ
         │       └── Non → INT8
         │
         ├── Fine-tuning nécessaire sur le modèle quantifié ?
         │       ├── Oui → QLoRA (voir peft_lora.md)
         │       └── Non → GPTQ ou AWQ post-training
```

### 11.3 Architecture de production recommandée

```
           ┌─────────────────────────────────────────┐
           │          Load Balancer / API GW          │
           └──────────────┬──────────────────────────┘
                          │
           ┌──────────────▼──────────────┐
           │       vLLM Server           │
           │   (ou TGI / TRT-LLM)        │
           │                             │
           │  ┌─────────────────────┐    │
           │  │  Model + KV Cache   │    │
           │  │  (PagedAttention)   │    │
           │  └─────────────────────┘    │
           │  ┌─────────────────────┐    │
           │  │  LoRA adapters pool │    │  ← optionnel
           │  └─────────────────────┘    │
           └─────────────────────────────┘
                          │
           ┌──────────────▼──────────────┐
           │     Monitoring              │
           │  Prometheus + Grafana       │
           │  Métriques : latence P95,   │
           │  throughput, GPU util,      │
           │  TTFT, tokens/s             │
           └─────────────────────────────┘
```

### 11.4 Configuration vLLM recommandée en production

```python
# docker-compose.yml

services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --tensor-parallel-size 1
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --enable-chunked-prefill
      --max-num-seqs 256
    ports:
      - "8000:8000"
```

### 11.5 Métriques à monitorer en production

```python
# Métriques clés à exposer via Prometheus

metrics = {
    # Latence
    "vllm:time_to_first_token_seconds": "TTFT (p50, p95, p99)",
    "vllm:time_per_output_token_seconds": "TPOT",
    "vllm:e2e_request_latency_seconds": "Latence totale",

    # Throughput
    "vllm:generation_tokens_total": "Tokens générés/s",
    "vllm:prompt_tokens_total": "Tokens prompt/s",

    # Ressources
    "vllm:gpu_cache_usage_perc": "Utilisation KV cache",
    "vllm:num_requests_running": "Requêtes en cours",
    "vllm:num_requests_waiting": "Requêtes en attente",
}
```

---

## 12. Sources et Références

- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression](https://arxiv.org/abs/2306.00978)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)

---

## 🔗 Liens connexes

- [Infrastructure IA →](./infra_ia.md)
- [PEFT, LoRA et Fine-tuning Efficace →](../02_deep_learning/03_llm/peft_lora.md)
- [RAG →](../02_deep_learning/03_llm/rag.md)
- [Distillation LLM →](../02_deep_learning/03_llm/distillation_llm.md)

---

*Dernière mise à jour : Juin 2026*
