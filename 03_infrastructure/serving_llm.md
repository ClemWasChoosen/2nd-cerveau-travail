# Serving et Optimisation d'Inférence des LLMs

## Table des Matières

0. [Avant de commencer](#0-avant-de-commencer)
1. [Pourquoi l'inférence LLM est difficile](#1-pourquoi-linférence-llm-est-difficile)
2. [Concepts Fondamentaux](#2-concepts-fondamentaux)
3. [vLLM et PagedAttention](#3-vllm-et-pagedattention)
4. [Hugging Face TGI](#4-hugging-face-tgi)
5. [TensorRT-LLM](#5-tensorrt-llm)
6. [SGLang](#6-sglang)
7. [llama.cpp et GGUF](#7-llamacpp-et-gguf)
8. [Quantization](#8-quantization)
9. [FlashAttention](#9-flashattention)
10. [Speculative Decoding](#10-speculative-decoding)
11. [Continuous Batching et Chunked Prefill](#11-continuous-batching-et-chunked-prefill)
12. [Guide de Choix et Déploiement](#12-guide-de-choix-et-déploiement)
13. [⚠️ Pièges Courants](#13-️-pièges-courants)
14. [🎯 Exercices d'Auto-évaluation](#14--exercices-dauto-évaluation)
15. [Sources et Références](#15-sources-et-références)

---

## 0. Avant de commencer

### Prérequis

- Architecture Transformer : attention, FFN, couches séquentielles
- Notions de base GPU : VRAM, SRAM, HBM, parallélisme
- Python et Docker de base
- Concepts de serving web (API REST, latence, throughput)

### Objectifs d'apprentissage *(taxonomie de Bloom)*

| Niveau | Objectif |
|--------|---------|
| **Comprendre** | Expliquer pourquoi la génération autoregressive est difficile à servir à grande échelle |
| **Analyser** | Comparer vLLM, TGI, TensorRT-LLM, SGLang selon les contraintes de déploiement |
| **Appliquer** | Déployer un LLM avec vLLM en mode production via Docker |
| **Évaluer** | Choisir la bonne stratégie de quantization selon le trade-off qualité/VRAM |
| **Synthétiser** | Concevoir une architecture de serving LLM avec monitoring pour la production |

### TL;DR en 3 phrases

> Servir un LLM efficacement nécessite de gérer le KV cache (mémoire des calculs d'attention passés) sur de nombreuses requêtes concurrentes de longueurs variables. vLLM résout ce problème avec PagedAttention (inspiration OS paging). FlashAttention, la quantization et le speculative decoding complètent l'arsenal d'optimisation.

---

## 1. Pourquoi l'inférence LLM est difficile

### 1.1 Analogie

Imagine un chef cuisinier qui doit préparer des plats pour une salle pleine. Chaque plat (requête) a une durée de préparation inconnue à l'avance. Si le chef attend que tous les plats d'une table soient finis avant de servir, la salle tourne très mal. Un serveur d'inférence LLM naïf fait exactement cette erreur.

De plus, pour chaque plat, le chef doit se souvenir de toutes les étapes précédentes (le KV cache). Si cette mémoire de travail est mal gérée, beaucoup d'espace est gaspillé.

### 1.2 Propriétés qui rendent l'inférence LLM atypique

Contrairement à un modèle ML classique (résultat en <10ms, coût quasi nul), un LLM présente des propriétés très particulières :

| Propriété | ML Classique | LLM |
|-----------|-------------|-----|
| Latence | < 10ms | 100ms - plusieurs secondes |
| VRAM nécessaire | Quelques Mo | Plusieurs Go |
| Génération | Prédiction unique | Autoregressive, token par token |
| Parallélisme | Facile (indépendant) | Complexe (dépendance séquentielle) |
| Longueur des sorties | Fixe | Variable et imprévisible |

> Voir [infra_ia.md](./infra_ia.md) pour le contexte général sur l'infrastructure IA.

### 1.3 La génération autoregressive

Un LLM génère du texte **token par token** : chaque nouveau token dépend de tous les précédents.

$$P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, ..., w_{t-1})$$

**Conséquence fondamentale :** il est impossible de paralléliser la génération d'une séquence. Pour T tokens, le modèle doit faire T passes forward séquentielles.

---

## 2. Concepts Fondamentaux

### 2.1 Le KV Cache

Pour éviter de recalculer tout le contexte à chaque token, les moteurs d'inférence stockent un **KV cache** : les tenseurs K (Keys) et V (Values) de chaque couche d'attention, pour tous les tokens précédents.

**Formule générale du KV cache :**

$$\text{KV cache} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times seq\_len \times \text{dtype\_bytes}$$

> ⚠️ **Important** : depuis l'adoption de GQA (Grouped Query Attention), `n_kv_heads` est souvent **inférieur** à `n_heads`. Voir section 2.4.

**Exemple concret : LLaMA 3.1 8B** (32 couches, **8 KV heads GQA**, dim 128, BF16) pour 8192 tokens :

$$= 2 \times 32 \times 8 \times 128 \times 8192 \times 2 \approx \mathbf{1.1 \text{ Go}}$$

*Note : un calcul erroné avec 32 KV heads (comme si le modèle était MHA classique) donnerait ~8.6 Go, soit une surestimation d'un facteur 4. LLaMA 3.1 8B utilise GQA avec 8 KV heads.*

**Le problème du KV cache :** avec de nombreuses requêtes simultanées et des longueurs variables, sa gestion est un goulot d'étranglement majeur.

### 2.2 Métriques de performance

Avant de choisir un serveur d'inférence, savoir mesurer :

| Métrique | Définition | Unité |
|----------|-----------|-------|
| **TTFT** | Time To First Token (latence perçue) | ms |
| **TPOT** | Time Per Output Token | ms/token |
| **Throughput** | Tokens générés par seconde au total | tokens/s |
| **Latence P50/P95/P99** | Percentiles de latence end-to-end | ms |
| **MFU** | Model FLOPs Utilization (efficacité théorique du GPU) | % |
| **Concurrence** | Nombre de requêtes simultanées supportées | n |

### 2.3 Parallélisme d'inférence

Pour servir de grands modèles ou augmenter le throughput :

| Type | Description | Quand l'utiliser |
|------|-------------|-----------------|
| **Tensor Parallelism** | Découpe les tenseurs entre GPU (attention heads) | Modèle > VRAM d'un GPU |
| **Pipeline Parallelism** | Distribue les couches entre GPU | Très grands modèles (>70B) |
| **Data Parallelism** | Copies du modèle sur plusieurs GPU | Scaling horizontal du throughput |

### 2.4 MQA et GQA : optimisations architecturales du KV cache

Ces optimisations réduisent la taille du KV cache **au niveau de l'architecture du modèle**.

**Multi-Head Attention (MHA)** classique : autant de KV heads que de Q heads.

**Multi-Query Attention (MQA)** (Shazeer, 2019) : un seul KV head pour tous les Q heads.

$$\text{KV cache MQA} = \frac{1}{n\_heads} \times \text{KV cache MHA}$$

**Grouped Query Attention (GQA)** (Ainslie et al., 2023) : G KV heads pour H Q heads (compromis MHA/MQA).

$$\text{KV cache GQA} = \frac{G}{H} \times \text{KV cache MHA}$$

**Adoption actuelle :**

| Modèle | Type attention | Réduction KV cache vs MHA |
|--------|---------------|---------------------------|
| LLaMA 1, GPT-2 | MHA | ×1 (référence) |
| LLaMA 2 70B | GQA (G=8, H=64) | ×8 |
| LLaMA 3.1 8B | GQA (G=8, H=32) | ×4 |
| LLaMA 3.1 70B | GQA (G=8, H=64) | ×8 |
| Mistral 7B | GQA (G=8, H=32) | ×4 |
| Qwen 2.5 | GQA | variable |

> **Impact direct sur le serving** : un modèle GQA peut tenir plus de requêtes simultanées en VRAM qu'un modèle MHA de même taille. Toujours vérifier `num_key_value_heads` dans la config du modèle.

### 2.5 Sampling strategies

| Paramètre | Rôle | Valeur typique |
|-----------|------|----------------|
| `temperature` | Créativité (0 = déterministe, 1 = distribution native) | 0.0 - 1.0 |
| `top_p` | Nucleus sampling (cumul de probabilité) | 0.9 - 0.95 |
| `top_k` | Restriction aux k tokens les plus probables | 20 - 50 |
| `max_new_tokens` | Longueur max de la génération | 256 - 4096 |

---

## 3. vLLM et PagedAttention

### 3.1 Le problème résolu

Les serveurs d'inférence classiques allouent de la mémoire **statique et contiguë** pour le KV cache. Problèmes :
- **Fragmentation interne** : la longueur maximale est réservée dès le départ, même si la séquence est courte
- **Gaspillage** : des blocs mémoire sont partiellement utilisés
- **Faible concurrence** : peu de requêtes simultanées tiennent en VRAM

**vLLM** résout ce problème avec **PagedAttention** (Kwon et al., 2023).

### 3.2 PagedAttention

PagedAttention s'inspire de la **mémoire virtuelle** des systèmes d'exploitation (paging).

```
Gestion classique (fragmentation) :
┌─────────────────────────────────┐
│ Req 1 : [████████░░░░░░░░░░░░] │  ← max_len réservé, partiellement utilisé
│ Req 2 : [████░░░░░░░░░░░░░░░░] │
└─────────────────────────────────┘

PagedAttention (vLLM) :
┌───────┬───────┬───────┬───────┐
│ P1 R1 │ P2 R1 │ P3 R2 │ P4 R1 │  ← pages allouées dynamiquement
│ P5 R1 │ P6 R2 │ P7 R3 │ P8 R3 │     au fur et à mesure des tokens
└───────┴───────┴───────┴───────┘
```

**Principe :**
- Le KV cache est découpé en **blocs (pages) de taille fixe**
- L'allocation est **dynamique** : une page est allouée par chunk de tokens générés
- Des requêtes peuvent **partager des pages** (même préfixe système, prefix caching)
- Fragmentation quasi éliminée

### 3.3 Gains mesurés

> **Contexte des benchmarks** : chiffres issus de l'article original de vLLM (Kwon et al., 2023), mesurés sur un modèle OPT-13B/30B/175B avec des traces de conversation ShareGPT. Les gains varient selon le modèle, la longueur des séquences et la charge.

- **2-4× plus de throughput** vs Hugging Face Transformers naïf (sans batching dynamique)
- Utilisation mémoire quasi optimale vs gestion contiguë classique

### 3.4 Installation et usage

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,        # nombre de GPUs
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,   # laisser 10% pour les activations et overhead
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Explique le machine learning en 3 phrases."], sampling_params)
print(outputs[0].outputs[0].text)
```

**Serveur OpenAI-compatible :**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --port 8000 \
    --enable-chunked-prefill        # voir section 11
```

### 3.5 Fonctionnalités avancées

- **Multi-LoRA serving** : charger et switcher dynamiquement entre plusieurs adaptateurs LoRA sans recharger le modèle de base
- **Prefix caching** : partage automatique des pages KV pour les préfixes système identiques (réduction TTFT significative)
- **Speculative decoding** : support intégré (voir section 10)
- **Quantization** : AWQ, GPTQ, FP8 supportés nativement
- **Vision models** : LLaVA, Qwen-VL, Phi-3-Vision, etc.

---

## 4. Hugging Face TGI

### 4.1 Présentation

**Text Generation Inference (TGI)** est le serveur d'inférence officiel de Hugging Face. Il est utilisé en production sur les Inference Endpoints HF.

### 4.2 Fonctionnalités clés

- Continuous batching
- Token streaming (Server-Sent Events)
- Tensor parallelism
- Quantization (bitsandbytes, GPTQ, AWQ, EETQ)
- FlashAttention intégré
- API OpenAI-compatible
- Support LoRA adapters (depuis TGI v1.4+)
- Intégration directe avec le Hub HuggingFace

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

**TensorRT-LLM** est le framework NVIDIA pour optimiser l'inférence des LLMs sur GPU NVIDIA. Il opère à un niveau bas : compilation du modèle en un moteur optimisé pour un GPU cible spécifique.

### 5.2 Optimisations appliquées

- **Kernel fusion** : fusionne plusieurs opérations en un seul kernel GPU (moins d'aller-retours mémoire)
- **In-flight batching** : continuous batching natif
- **Quantization** : INT8, FP8 (H100), INT4-AWQ, GPTQ
- **Tensor/Pipeline parallelism**
- **Paged KV cache** (similaire à vLLM)
- **Speculative decoding**

### 5.3 Workflow de déploiement

TensorRT-LLM requiert une étape de **compilation** spécifique au GPU cible. Le workflow typique (ligne de commande) :

```bash
# Étape 1 : convertir les poids HuggingFace en checkpoint TRT-LLM
python convert_checkpoint.py \
    --model_dir ./llama-3.1-8b \
    --output_dir ./trtllm_ckpt \
    --dtype bfloat16

# Étape 2 : compiler le moteur pour le GPU cible
trtllm-build \
    --checkpoint_dir ./trtllm_ckpt \
    --output_dir ./trtllm_engine \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_seq_len 4096

# Étape 3 : servir avec Triton Inference Server
# (ou directement via le runtime TRT-LLM)
```

> **Note** : le moteur compilé est spécifique au GPU cible (ex. A100 80Go) et aux paramètres de build (max_batch_size, longueur max...). Changer de GPU ou de configuration nécessite une recompilation.

### 5.4 Comparaison avec vLLM

| Critère | vLLM | TensorRT-LLM |
|---------|------|--------------|
| Facilité de démarrage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Performance brute latence | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Portabilité GPU | NVIDIA + AMD (ROCm) | NVIDIA uniquement |
| Compilation nécessaire | Non | Oui (spécifique au GPU) |
| Support de modèles | Très large | Large mais parfois en retard |
| Time-to-production | Rapide | Plus long (compilation, tests) |
| Optimisations NVIDIA (H100 FP8...) | Bonnes | Maximales |

**Recommandation :** commencer avec vLLM. Migrer vers TensorRT-LLM si le gain de performance supplémentaire (typiquement 10-30% sur H100) justifie la complexité opérationnelle.

---

## 6. SGLang

### 6.1 Présentation

**SGLang** (Structured Generation Language, Zheng et al., 2023) est un framework de serving LLM développé à UCB/UCSD, conçu pour maximiser les performances sur les workloads avec réutilisation de préfixes (agents, multi-turn, RAG).

### 6.2 Innovation principale : RadixAttention

SGLang utilise un **arbre radix (trie)** pour gérer le KV cache, permettant une réutilisation maximale des calculs partagés entre requêtes.

```
Requête 1 : "[System prompt] Réponds en français. [User] Quelle est Paris ?"
Requête 2 : "[System prompt] Réponds en français. [User] Quelle est Lyon ?"
                                                           ↑
                               Le KV cache du system prompt est réutilisé
```

Avec PagedAttention (vLLM), le prefix caching est possible mais moins granulaire. Avec RadixAttention, tout préfixe commun exact est réutilisé automatiquement, token par token.

### 6.3 Quand SGLang est particulièrement efficace

- **Applications agents** : appels multi-turn avec system prompt fixe
- **RAG** : contexte documentaire souvent partagé entre requêtes
- **Batch inference** : nombreuses requêtes avec préfixes communs
- **Structured outputs** : contraintes JSON, grammaires, regex

### 6.4 Installation et usage

```bash
pip install sglang[all]
```

```bash
# Lancer le serveur
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --dtype bfloat16
```

```python
import openai  # API compatible OpenAI

client = openai.Client(base_url="http://localhost:30000/v1", api_key="None")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Bonjour !"}],
)
```

### 6.5 SGLang vs vLLM

| Critère | SGLang | vLLM |
|---------|--------|------|
| Réutilisation de préfixes | ✅ RadixAttention (très efficace) | ✅ Prefix caching (bon) |
| Workloads agents/multi-turn | ✅ Excellent | ✅ Bon |
| Latence batch homogène | ✅ Souvent meilleur | ✅ Très bon |
| Maturité/écosystème | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Support modèles | Large | Très large |
| Multi-LoRA | ✅ | ✅ |

> SGLang est souvent plus rapide que vLLM pour les workloads avec beaucoup de réutilisation de préfixes. vLLM reste le choix plus "safe" pour sa maturité et son large écosystème. Les deux évoluent rapidement : comparer les benchmarks récents pour votre cas d'usage spécifique.

---

## 7. llama.cpp et GGUF

### 7.1 Présentation

**llama.cpp** est une implémentation C++ des LLMs permettant de les exécuter localement sur CPU ou GPU, à partir de modèles **quantifiés au format GGUF**.

### 7.2 Format GGUF

GGUF (GGML Unified Format) est un format binaire contenant poids quantifiés, métadonnées et paramètres de tokenization dans un seul fichier.

**Niveaux de quantization GGUF courants :**

| Quantization | Bits/param | VRAM 7-8B | Qualité |
|-------------|-----------|-----------|---------|
| Q2_K | ~2.6 | ~3.1 Go | Médiocre (usage limité) |
| Q4_K_M | ~4.8 | ~5.0 Go | Bon compromis (recommandé) |
| Q5_K_M | ~5.7 | ~5.8 Go | Bon |
| Q6_K | ~6.6 | ~6.6 Go | Très bon |
| Q8_0 | 8.0 | ~8.5 Go | Excellent (quasi sans perte) |
| F16 | 16.0 | ~16 Go | Référence |

### 7.3 Usage

```bash
# Lancer un serveur HTTP local
./llama-server \
    -m ./models/llama3-8b-q4_k_m.gguf \
    --port 8080 \
    -ngl 35        # couches offloadées sur GPU (0 = CPU only)
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama3-8b-q4_k_m.gguf",
    n_ctx=4096,       # taille du contexte
    n_gpu_layers=35,  # couches GPU (0 = CPU only)
    verbose=False
)
output = llm("Explique les réseaux de neurones:", max_tokens=256)
print(output["choices"][0]["text"])
```

### 7.4 Cas d'usage

| Cas | llama.cpp adapté ? |
|-----|-------------------|
| Machine personnelle / laptop | ✅ Idéal |
| Application offline / edge | ✅ Idéal |
| Confidentialité maximale (données sensibles) | ✅ Idéal |
| Apple Silicon (Metal) | ✅ Excellent |
| Production haute concurrence (>10 req/s) | ❌ Préférer vLLM/TGI |
| Entraînement | ❌ Non supporté |

---

## 8. Quantization

### 8.1 Principe

La **quantization** réduit la précision numérique des poids (et parfois des activations).

**Formats courants :**

| Format | Bits | Notes |
|--------|------|-------|
| FP32 | 32 | Pleine précision, entraînement |
| BF16 | 16 | Standard LLM modernes (meilleure plage dynamique que FP16) |
| FP16 | 16 | Moins stable que BF16 pour les LLMs |
| FP8 | 8 | H100 uniquement, très efficace |
| INT8 | 8 | Entier, bonne qualité |
| INT4 / NF4 | 4 | Entier ou NormalFloat, QLoRA |

**Gain mémoire théorique :**

$$\text{VRAM}_{\text{poids}} \approx N_{params} \times \frac{\text{bits}}{8} \text{ octets}$$

| Précision | VRAM 8B |
|-----------|---------|
| FP32 | ~32 Go |
| BF16/FP16 | ~16 Go |
| INT8 | ~8 Go |
| INT4 | ~4 Go |

### 8.2 Méthodes de quantization post-training (PTQ)

#### GPTQ (Frantar et al., 2022)

Quantization par blocs avec compensation des erreurs via la matrice hessienne. Précise mais plus lente à quantifier.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(calibration_examples)
model.save_quantized("./model_gptq")
```

#### AWQ (Lin et al., 2023)

Activation-aware Weight Quantization : préserve les poids les plus saillants identifiés par l'analyse des activations. Souvent meilleure qualité que GPTQ à même nombre de bits.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./model_awq")
```

### 8.3 Impact sur la qualité

Règle générale pour des LLMs bien entraînés (les résultats varient selon le modèle et la tâche) :

| Précision | Dégradation typique sur benchmarks |
|-----------|-------------------------------------|
| FP8 / BF16 | Négligeable (<0.5%) |
| INT8 | Très faible (<1%) |
| INT4 / AWQ | Légère (1-3%) |
| INT4 / GPTQ (standard) | Modérée (2-5%) |
| INT2/INT3 | Significative (>5%) |

> Ces pourcentages sont indicatifs et varient fortement selon la tâche et le modèle.

---

## 9. FlashAttention

### 9.1 Problème de l'attention standard

L'attention classique du Transformer matérialise la **matrice d'attention complète** en mémoire :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Pour une séquence de longueur $$n$$, cette matrice est de taille $$n \times n$$.

**Deux bottlenecks distincts :**
- **Mémoire HBM** : la matrice $$n \times n$$ est stockée en HBM (High Bandwidth Memory) → $$O(n^2)$$ mémoire
- **Bande passante mémoire** : les aller-retours entre SRAM (petite, rapide) et HBM (grande, plus lente) sont coûteux → l'attention est **IO-bound**, pas compute-bound

### 9.2 Solution FlashAttention

FlashAttention (Dao et al., 2022) réorganise le calcul pour **minimiser les accès HBM** :

1. **Tiling** : Q, K, V sont découpés en blocs qui tiennent en SRAM
2. **Online softmax** : le softmax est calculé en une seule passe sans matérialiser la matrice complète
3. **Kernel fusion** : les opérations QK^T, softmax et multiplication par V sont fusionnées

**Résultats :**

| Aspect | Standard | FlashAttention |
|--------|---------|----------------|
| Mémoire HBM (matrice attention) | $$O(n^2)$$ | **$$O(n)$$** |
| Complexité de calcul (FLOPs) | $$O(n^2)$$ | **$$O(n^2)$$** (inchangée) |
| Vitesse pratique | Référence | **2-4× plus rapide** |

> **Distinction importante** : FlashAttention réduit la **mémoire HBM** de $$O(n^2)$$ à $$O(n)$$, mais **la complexité de calcul reste $$O(n^2)$$** (le même nombre de dot products est effectué). Le gain de vitesse vient du fait que l'attention standard est **memory-bandwidth-bound** : en réduisant les accès HBM, on peut exploiter bien mieux le GPU.

### 9.3 Versions

| Version | Améliorations | Matériel cible |
|---------|--------------|----------------|
| FlashAttention 1 | Base (tiling, IO-awareness) | A100, GPU Ampere |
| FlashAttention 2 | Meilleur parallélisme Q, 2× plus rapide | GPU Ampere+ |
| FlashAttention 3 | H100 optimisé, FP8 natif, pipelining asynchrone | H100 (Hopper) |

### 9.4 Intégration

```python
# Activé automatiquement dans Transformers récent (>= 4.36)
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="flash_attention_2",  # FA2 explicitement
    torch_dtype=torch.bfloat16,
)
```

FlashAttention est déjà intégré dans : vLLM, TGI, TensorRT-LLM, SGLang, PyTorch >= 2.0 (via `scaled_dot_product_attention`).

---

## 10. Speculative Decoding

### 10.1 Principe

La génération autoregressive est séquentielle. Le speculative decoding l'accélère avec **deux modèles** :

1. Un **draft model** (petit, rapide) propose $$\gamma$$ tokens en $$\gamma$$ passes
2. Le **target model** (grand) vérifie ces propositions en **un seul passage forward parallèle**

```
Draft model (LLaMA 68M, 4 passes) :
  prompt → [tok1, tok2, tok3, tok4]

Target model (LLaMA 70B, 1 seule passe) :
  prompt + draft → [✅ tok1, ✅ tok2, ❌ tok3] → resample tok3
  → 3 tokens acceptés avec 1 passe du gros modèle
```

**Condition d'acceptation** (préserve la distribution du target) :

Pour un token proposé $$x$$ avec probabilité draft $$q(x)$$ et target $$p(x)$$ :

$$P(\text{accepter}) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

Si rejeté, on resample depuis la distribution corrigée normalisée. La distribution finale est **exactement** celle du target model.

### 10.2 Gains

- **1.5× à 3× d'accélération** selon le taux d'acceptation et le ratio de taille
- Particulièrement efficace pour des outputs prévisibles (code boilerplate, JSON structuré, résumés)
- Moins efficace pour des outputs créatifs ou imprévisibles

### 10.3 Variantes modernes

| Variante | Description | Avantage |
|---------|-------------|----------|
| **Speculative Decoding** classique | Draft model séparé (ex. LLaMA 1B → LLaMA 70B) | Simple |
| **Medusa** | Têtes prédictives multiples sur le même modèle | Pas de modèle séparé |
| **EAGLE / EAGLE-2** | Draft model entraîné sur les features du target | Meilleur taux d'acceptation |
| **Lookahead Decoding** | N-grams du contexte comme draft | Sans entraînement supplémentaire |

---

## 11. Continuous Batching et Chunked Prefill

### 11.1 Le problème du batching statique

En batching classique, on attend un batch complet avant l'inférence. Problèmes :
- Séquences de longueurs très différentes → padding = gaspillage GPU
- Si une séquence finit avant les autres, son slot reste inutilisé jusqu'à la fin du batch

### 11.2 Continuous Batching (iteration-level scheduling)

Le continuous batching traite les requêtes **itération par itération** (token par token). Une nouvelle requête peut rejoindre le batch dès qu'une place se libère.

```
Batching statique :
Iter 1: [R1_t1, R2_t1, R3_t1]
Iter 2: [R1_t2, R2_t2, R3_t2]
Iter 3: [R1_t3, -------, R3_t3]  ← R2 terminée mais slot gaspillé
Iter 4: [R1_t4, -------, R3_t4]

Continuous batching :
Iter 1: [R1_t1, R2_t1, R3_t1]
Iter 2: [R1_t2, R2_t2, R3_t2]
Iter 3: [R1_t3, R4_t1, R3_t3]  ← R2 terminée, R4 démarre immédiatement
Iter 4: [R1_t4, R4_t2, R5_t1]  ← R3 terminée, R5 démarre
```

**Impact** : utilisation GPU significativement meilleure. Tous les serveurs modernes (vLLM, TGI, TensorRT-LLM, SGLang) implémentent le continuous batching.

### 11.3 Chunked Prefill

**Problème :** une requête avec un long prompt (ex. 32K tokens) peut monopoliser le GPU pendant toute la phase de prefill, retardant la génération de toutes les autres requêtes en attente (augmente leur TTFT).

**Solution :** le chunked prefill découpe le prefill des longues séquences en **petits chunks**, entrelacés avec les itérations de decode des requêtes en cours.

```
Sans chunked prefill :
Iter 1-50: [PREFILL long (32K tokens) ................] ← bloque tout
Iter 51:   [R1_decode, R2_decode, R3_decode]

Avec chunked prefill (chunk_size=512) :
Iter 1:  [PREFILL chunk 1/64, R1_decode, R2_decode]
Iter 2:  [PREFILL chunk 2/64, R1_decode, R2_decode]
...
```

**Bénéfice** : TTFT réduit pour les requêtes courtes qui attendaient derrière un long prefill, sans impact majeur sur le throughput total.

**Activation dans vLLM :** `--enable-chunked-prefill` (activé par défaut depuis vLLM >= 0.4.0).

---

## 12. Guide de Choix et Déploiement

### 12.1 Comparaison globale

| Critère | vLLM | TGI | TensorRT-LLM | SGLang | llama.cpp |
|---------|------|-----|-------------|--------|-----------|
| **Throughput** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Facilité déploiement** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU requis** | NVIDIA/AMD | NVIDIA | NVIDIA | NVIDIA/AMD | CPU ou GPU |
| **Multi-LoRA** | ✅ | ✅ (v1.4+) | ✅ | ✅ | ❌ |
| **Réutilisation préfixes** | ✅ (prefix caching) | ✅ | ✅ | ✅✅ (RadixAttention) | ❌ |
| **Local / edge** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **API OpenAI compat.** | ✅ | ✅ | Via Triton | ✅ | ✅ |

### 12.2 Arbre de décision

```
Déploiement local / laptop / edge ?
    ├── Oui → llama.cpp
    └── Non (serveur avec GPU)
         │
         ├── Workload agents / multi-turn / RAG avec préfixes partagés ?
         │       ├── Oui → SGLang
         │       └── Non
         │               │
         │               ├── Stack HuggingFace existante ?
         │               │       ├── Oui → TGI
         │               │       └── Non
         │               │               │
         │               │               ├── Optimisation NVIDIA max. (H100) ?
         │               │               │       ├── Oui → TensorRT-LLM
         │               │               │       └── Non → vLLM (choix par défaut)
```

### 12.3 Stratégie de quantization

```
VRAM suffisante pour le modèle en BF16 ?
    ├── Oui → Pas de quantization nécessaire
    └── Non
         │
         ├── Fine-tuning QLoRA nécessaire ?
         │       ├── Oui → NF4 via bitsandbytes (voir peft_lora.md)
         │       └── Non (inference seulement)
         │               │
         │               ├── H100 disponible ?
         │               │       ├── Oui → FP8 (TensorRT-LLM ou vLLM)
         │               │       └── Non
         │               │               │
         │               │               ├── Légère perte qualité acceptable ?
         │               │               │       ├── Oui → INT4/AWQ
         │               │               │       └── Non → INT8
```

### 12.4 Architecture de production recommandée

```
            ┌─────────────────────────────────────────────┐
            │       Load Balancer / API Gateway            │
            │         (nginx, traefik, kong)               │
            └──────────────────┬──────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │          vLLM / SGLang               │
            │                                      │
            │  ┌──────────────────────────────┐    │
            │  │  Model + KV Cache            │    │
            │  │  (PagedAttention / Radix)    │    │
            │  └──────────────────────────────┘    │
            │  ┌──────────────────────────────┐    │
            │  │  LoRA Adapters Pool (opt.)   │    │
            │  └──────────────────────────────┘    │
            └─────────────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │            Monitoring                │
            │     Prometheus + Grafana             │
            │  TTFT p50/p95, tokens/s,             │
            │  GPU util, KV cache usage            │
            └─────────────────────────────────────┘
```

### 12.5 Configuration vLLM production (Docker Compose)

```yaml
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
```

### 12.6 Métriques à exposer via Prometheus

```python
# Métriques natives vLLM (exposées sur /metrics)
metrics_cles = {
    # Latence
    "vllm:time_to_first_token_seconds":  "TTFT (p50, p95, p99) — latence perçue",
    "vllm:time_per_output_token_seconds": "TPOT — vitesse de génération",
    "vllm:e2e_request_latency_seconds":   "Latence totale end-to-end",

    # Throughput
    "vllm:generation_tokens_total": "Tokens générés/s",
    "vllm:prompt_tokens_total":     "Tokens prompt/s",

    # Ressources
    "vllm:gpu_cache_usage_perc":    "Utilisation KV cache (saturation si > 90%)",
    "vllm:num_requests_running":    "Requêtes en cours de génération",
    "vllm:num_requests_waiting":    "Requêtes en file d'attente (signal de saturation)",
}
```

---

## 13. ⚠️ Pièges Courants

### Serving et configuration

**Piège 1 : `gpu_memory_utilization` trop élevé avec quantization**
`0.90` peut provoquer un OOM si le modèle est quantifié et que le KV cache grandit. Avec quantization, le modèle de base tient en moins de VRAM mais le KV cache reste non-quantifié. Commencer avec `0.85`.

**Piège 2 : Utiliser `max_model_len` trop grand sans besoin**
Un `max_model_len=128000` réserve potentiellement beaucoup de pages KV, réduisant la concurrence. Si tes requêtes font en moyenne 4096 tokens, `max_model_len=8192` est plus efficace.

**Piège 3 : Confondre `max_new_tokens` et `max_model_len`**
`max_model_len` est la longueur totale (prompt + réponse) que le modèle peut traiter. `max_new_tokens` est la longueur maximale de la réponse générée. Dépasser `max_model_len - longueur_prompt` provoque une erreur.

**Piège 4 : Oublier `--enable-chunked-prefill` pour les longues séquences**
Sans chunked prefill, un prompt de 32K tokens peut bloquer le serving pendant plusieurs secondes pour toutes les autres requêtes.

### Quantization

**Piège 5 : Comparer des benchmarks de qualité sans préciser la tâche**
Un modèle INT4-AWQ peut perdre <1% sur MMLU mais 5-10% sur les benchmarks de code ou de mathématiques. Toujours évaluer sur la tâche cible.

**Piège 6 : Charger un modèle GPTQ avec vLLM sans spécifier `--quantization gptq`**
vLLM ne détecte pas toujours automatiquement la quantization. Spécifier explicitement `--quantization gptq|awq|fp8`.

### Architecture KV Cache

**Piège 7 : Utiliser la formule MHA pour un modèle GQA**
Calculer le KV cache d'un LLaMA 3.1 70B avec 64 heads au lieu de 8 KV heads surestime la VRAM KV par 8×. Toujours vérifier `num_key_value_heads` dans `config.json`.

---

## 14. 🎯 Exercices d'Auto-évaluation

### Questions de révision

1. Quelle est la différence entre TTFT et TPOT ? Lequel est le plus important pour un chatbot interactif ?
2. Pourquoi FlashAttention ne réduit-il pas la complexité de calcul O(n²) de l'attention ?
3. Calculer le KV cache de LLaMA 3.1 8B (8 KV heads, dim 128, 32 layers, BF16) pour une séquence de 4096 tokens.
4. Pourquoi le chunked prefill améliore-t-il le TTFT des requêtes courtes ?
5. Dans quel cas SGLang surpasse-t-il vLLM en throughput ?

### Exercice de calcul

Comparer la taille du KV cache pour deux modèles 7B, pour 16 requêtes simultanées de 2048 tokens chacune, en BF16 :
- Modèle A (MHA classique) : 32 layers, 32 heads, dim 128
- Modèle B (GQA) : 32 layers, 8 KV heads, dim 128

*Réponse attendue :*
- Modèle A : $$2 \times 32 \times 32 \times 128 \times 2048 \times 2 \times 16 \approx 8.6 \text{ Go}$$
- Modèle B : $$2 \times 32 \times 8 \times 128 \times 2048 \times 2 \times 16 \approx 2.1 \text{ Go}$$
- Modèle B nécessite ~4× moins de VRAM KV cache, permettant 4× plus de requêtes simultanées.

<details>
<summary>Réponses aux questions de révision</summary>

1. TTFT = temps avant le premier token (latence perçue). TPOT = temps entre chaque token (vitesse de streaming). Pour un chatbot interactif, le TTFT est critique (l'utilisateur attend une réponse). Pour un batch processing, le throughput (tokens/s total) est plus important.

2. FlashAttention recalcule les blocs en SRAM au lieu de les stocker en HBM. Il fait toujours O(n²) opérations de calcul, mais les accès HBM sont réduits de O(n²) à O(n). Le gain de vitesse vient de l'élimination du bottleneck mémoire.

3. $$2 \times 32 \times 8 \times 128 \times 4096 \times 2 \approx 0.54 \text{ Go}$$

4. Sans chunked prefill, les requêtes courtes attendent que le long prefill soit entièrement terminé avant que leur génération commence. Avec chunked prefill, leur génération peut commencer pendant les pauses entre chunks du long prefill.

5. Pour les workloads avec de nombreuses requêtes partageant des préfixes communs (système d'agents, RAG avec contexte documentaire identique, multi-turn conversations). RadixAttention réutilise le KV cache partagé plus efficacement que le prefix caching de vLLM.

</details>

---

## 15. Sources et Références

### Papers fondateurs

- **vLLM / PagedAttention** : Kwon et al., 2023 — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **FlashAttention** : Dao et al., 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention-2** : Dao, 2023 — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3** : Shah et al., 2024 — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- **Speculative Decoding** : Leviathan et al., 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- **Speculative Sampling** : Chen et al., 2023 — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- **GPTQ** : Frantar et al., 2022 — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** : Lin et al., 2023 — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **GQA** : Ainslie et al., 2023 — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **SGLang** : Zheng et al., 2023 — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- **EAGLE** : Li et al., 2024 — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

### Documentation

- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sglang.readthedocs.io/)
- [Hugging Face TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)

---

## 🔗 Liens connexes

- [Infrastructure IA →](./infra_ia.md)
- [PEFT, LoRA et Fine-tuning Efficace →](../02_deep_learning/03_llm/peft_lora.md)
- [RAG →](../02_deep_learning/03_llm/rag.md)
- [Distillation LLM →](../02_deep_learning/03_llm/distillation_llm.md)

---

*Dernière mise à jour : Juin 2026*
