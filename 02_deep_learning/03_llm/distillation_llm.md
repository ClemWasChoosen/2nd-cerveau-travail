# Distillation des Large Language Models (LLM)

## 1. Introduction & Motivation

### 1.1 Qu'est-ce que la distillation ?

La **knowledge distillation** est une technique de compression de modèles où un petit modèle (student) apprend à imiter le comportement d'un grand modèle (teacher). 

**Analogie :** Un étudiant (petit modèle) apprend d'un professeur expert (grand modèle) non pas en mémorisant des réponses exactes, mais en comprenant la façon de raisonner du professeur.

### 1.2 Pourquoi distiller un LLM ?

**Problèmes des LLM massifs :**
- Coûts d'inférence élevés (latence, GPU/TPU requis)
- Consommation mémoire importante
- Difficiles à déployer (edge devices, applications mobiles)
- Coûts financiers et environnementaux

**Bénéfices de la distillation :**
- **Réduction de taille** : 2-10x plus petit
- **Vitesse** : 2-10x plus rapide
- **Maintien des performances** : 95-99% des capacités du teacher
- **Déploiement facilité** : CPU, mobile, edge

**Exemples célèbres :**
- DistilBERT (66M params) vs BERT (110M) : -40% taille, +60% vitesse, 97% performance
- DistilGPT-2, TinyLLaMA, Phi-2, Mistral 7B (souvent distillé de modèles plus gros)

---

## 2. Fondamentaux de la Knowledge Distillation

### 2.1 Principe de base (Hinton et al., 2015)

**Idée centrale :** Les probabilités "douces" (soft probabilities) d'un modèle entraîné contiennent plus d'information que les labels durs (hard labels).

**Exemple illustratif :**

Pour l'input "Le chat dort sur le..."

**Hard label (one-hot) :**

    canapé: 1.0  
    lit: 0.0  
    tapis: 0.0  

**Soft probabilities (teacher) :**

    canapé: 0.7
    lit: 0.2
    tapis: 0.08
    chaise: 0.02

**Pourquoi les soft probabilities sont meilleures ?**
- Elles capturent les **similarités** entre classes (lit ≈ canapé pour dormir)
- Elles transmettent les **incertitudes** du modèle
- Elles contiennent de la **dark knowledge** (connaissances implicites sur les relations entre tokens)

### 2.2 Architecture de distillation

```
┌─────────────────┐
│  Teacher Model  │ (Grand, pré-entraîné, figé)
│   (frozen)      │
└────────┬────────┘
         │ soft targets (probas)
         ▼
    ┌─────────┐
    │  Input  │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  Student Model  │ (Petit, en entraînement)
│   (training)    │
└────────┬────────┘
         │
         ▼
    Loss = α·L_distill + (1-α)·L_hard
```

---

## 3. Mathématiques de la Distillation

### 3.1 Temperature Softmax

**Softmax standard :**

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Softmax avec température T :**

$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

**Rôle de la température T :**

- **T = 1** : Softmax standard (distribution "dure")
- **T > 1** : Distribution "adoucie" (soft), plus d'information dans les faibles probabilités
- **T → ∞** : Distribution uniforme

**Exemple numérique :**

```python
import numpy as np

def softmax_with_temp(logits, T=1.0):
    exp_logits = np.exp(logits / T)
    return exp_logits / exp_logits.sum()

logits = np.array([10.0, 5.0, 2.0, 0.1])

print("T=1 (standard):", softmax_with_temp(logits, T=1))
# [0.9999, 0.0001, 0.0000, 0.0000] - très "dur"

print("T=5 (soft):", softmax_with_temp(logits, T=5))
# [0.6225, 0.2314, 0.0994, 0.0467] - beaucoup plus d'information
```

**Pourquoi augmenter T ?**
- Révèle les relations entre tokens peu probables
- Facilite l'apprentissage du student (gradient plus informatif)
- Réduit l'overconfidence du teacher

### 3.2 Loss de Distillation

**Loss complète :**

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{distill}}(p_{\text{student}}^T, p_{\text{teacher}}^T) + (1-\alpha) \cdot \mathcal{L}_{\text{hard}}(p_{\text{student}}, y_{\text{true}})$$

**Composantes :**

1. **$$\mathcal{L}_{\text{distill}}$$** : Divergence entre student et teacher (soft targets)
2. **$$\mathcal{L}_{\text{hard}}$$** : Cross-entropy avec les vrais labels
3. **$$\alpha$$** : Coefficient de pondération (typiquement 0.5-0.9)

**Choix de divergence pour $$\mathcal{L}_{\text{distill}}$$ :**

**a) Kullback-Leibler (KL) Divergence** (le plus courant)

$$\mathcal{L}_{\text{KL}} = \sum_i p_{\text{teacher}}(i) \log \frac{p_{\text{teacher}}(i)}{p_{\text{student}}(i)}$$

- Mesure l'information perdue quand on utilise student au lieu de teacher
- Asymétrique : pénalise différemment les erreurs

**b) Cross-Entropy**

$$\mathcal{L}_{\text{CE}} = -\sum_i p_{\text{teacher}}(i) \log p_{\text{student}}(i)$$

- Équivalent à KL + constante (pour l'optimisation)
- Plus simple à implémenter

**c) Mean Squared Error (MSE)** (rarement pour probabilités)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_i (p_{\text{teacher}}(i) - p_{\text{student}}(i))^2$$

**Pourquoi KL est préféré ?**
- Adapté aux distributions de probabilité
- Propriétés théoriques solides (théorie de l'information)
- Gradients mieux calibrés pour l'apprentissage

### 3.3 Scaling de la Loss

**Important :** Quand on utilise une température T > 1, il faut mettre à l'échelle la loss de distillation par $$T^2$$.

$$\mathcal{L}_{\text{distill}} = T^2 \cdot KL(p_{\text{teacher}}^T \| p_{\text{student}}^T)$$

**Pourquoi $$T^2$$ ?**
Les gradients de la soft softmax sont proportionnels à $$1/T^2$$. Multiplier par $$T^2$$ compense cet effet et maintient l'échelle des gradients.

---

## 4. Distillation pour les LLM

### 4.1 Spécificités des LLM

**Différences vs distillation classique (vision, etc.) :**

1. **Tâche autoregressive** : Prédiction token par token
2. **Vocabulaire massif** : 50k-100k tokens (vs 1000 classes en vision)
3. **Séquences longues** : Contexte de 2k-100k tokens
4. **Modèles causaux** : Chaque prédiction dépend du passé

### 4.2 Types de distillation pour LLM

**a) White-box distillation**

- Accès complet au teacher (logits, couches intermédiaires)
- Peut copier les représentations internes
- Exemples : DistilBERT, TinyBERT

**b) Black-box distillation**

- Accès uniquement aux sorties (tokens générés)
- Le teacher est une API (GPT-4, Claude, etc.)
- Nécessaire quand le teacher est propriétaire

**c) Self-distillation**

- Le modèle est son propre teacher
- Utile pour l'amélioration continue

### 4.3 Distillation Next-Token Prediction

**Setup standard pour LLM causaux (GPT-like) :**

```python
import torch
import torch.nn.functional as F

def distillation_loss(
    student_logits,  # [batch, seq_len, vocab_size]
    teacher_logits,  # [batch, seq_len, vocab_size]
    labels,          # [batch, seq_len]
    temperature=2.0,
    alpha=0.7
):
    """
    Loss de distillation pour LLM autorégressifs
    """
    # 1. Soft targets avec température
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 2. KL divergence (distillation loss)
    distill_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 3. Hard loss (cross-entropy avec vrais labels)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100  # padding tokens
    )
    
    # 4. Combinaison
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    
    return total_loss, distill_loss, hard_loss
```

**Paramètres typiques :**
- Temperature : 2.0 - 4.0
- Alpha : 0.5 - 0.9 (poids de la distillation)

### 4.4 Distillation de séquences (Sequence-level)

Au lieu de distiller token par token, on peut distiller au niveau séquence complète.

**Approche :**
1. Teacher génère des complétions pour des prompts
2. Student apprend à reproduire ces complétions
3. Avantage : capture mieux la cohérence long-terme

```python
def sequence_distillation(student, teacher, prompts, max_length=512):
    """
    Distillation au niveau séquence
    """
    teacher.eval()
    
    # Teacher génère les séquences cibles
    with torch.no_grad():
        teacher_outputs = teacher.generate(
            prompts,
            max_length=max_length,
            do_sample=True,
            temperature=1.0
        )
    
    # Student apprend à prédire ces séquences
    student_logits = student(teacher_outputs[:, :-1])
    targets = teacher_outputs[:, 1:]
    
    loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1)
    )
    
    return loss
```

### 4.5 Distillation des couches intermédiaires

Pour améliorer la distillation, on peut aussi faire correspondre les représentations internes.

**Layer-wise distillation :**

$$\mathcal{L}_{\text{hidden}} = \sum_{l \in \text{layers}} MSE(W \cdot h_{\text{student}}^{(l)}, h_{\text{teacher}}^{(l)})$$

- $$W$$ : matrice de projection (si dimensions différentes)
- $$h^{(l)}$$ : hidden states de la couche $$l$$

```python
def hidden_states_loss(student_hidden, teacher_hidden, projection_layer=None):
    """
    Loss sur les états cachés intermédiaires
    """
    if projection_layer:
        student_hidden = projection_layer(student_hidden)
    
    # MSE entre representations
    loss = F.mse_loss(student_hidden, teacher_hidden)
    
    return loss

# Exemple d'utilisation
projection = nn.Linear(student_dim, teacher_dim)
hidden_loss = hidden_states_loss(
    student_outputs.hidden_states[6],  # couche 6 du student
    teacher_outputs.hidden_states[12],  # couche 12 du teacher
    projection
)
```

**Pourquoi distiller les couches intermédiaires ?**
- Force le student à apprendre les mêmes représentations
- Améliore la convergence
- Transfère mieux les capacités complexes

---

## 5. Techniques Avancées

### 5.1 Progressive Distillation

**Idée :** Distiller en plusieurs étapes avec des students de taille croissante.

```
Teacher (7B) → Student-1 (3B) → Student-2 (1B) → Student-3 (350M)
```

**Avantages :**
- Chaque étape est plus facile
- Préserve mieux les performances finales

### 5.2 On-Policy Distillation

**Problème du off-policy :** Le student apprend sur des données générées par le teacher, pas par lui-même.

**Solution :** Générer des données avec le student et les raffiner avec le teacher.

```python
# Pseudo-code
for epoch in range(num_epochs):
    # 1. Student génère des réponses
    student_generations = student.generate(prompts)
    
    # 2. Teacher évalue et corrige
    teacher_logits = teacher(student_generations)
    
    # 3. Student apprend de ces corrections
    loss = distillation_loss(student_logits, teacher_logits)
    loss.backward()
```

### 5.3 Distillation de raisonnement (Chain-of-Thought)

Pour les modèles qui font du raisonnement step-by-step :

**Approche :**
1. Teacher génère des chaînes de raisonnement
2. Student apprend à les reproduire
3. Améliore les capacités de raisonnement complexe

**Exemple :**

    Prompt: "Combien font 23 * 47 ?"
    Teacher CoT:
    "Décomposons : 23 * 47 = 23 * (40 + 7)
    = 23 * 40 + 23 * 7
    = 920 + 161
    = 1081"
    Student apprend cette chaîne complète.

---

## 6. Implémentation Pratique

### 6.1 Pipeline complet

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

class LLMDistiller:
    def __init__(
        self,
        teacher_name="gpt2-large",
        student_name="gpt2",
        temperature=2.0,
        alpha=0.7,
        device="cuda"
    ):
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # Load models
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_name).to(device)
        self.student = AutoModelForCausalLM.from_pretrained(student_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def compute_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Compute distillation loss
        teacher_logits = teacher_outputs.logits[:, :-1, :]
        student_logits = student_outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        loss, distill_loss, hard_loss = distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            self.temperature,
            self.alpha
        )
        
        return loss, distill_loss, hard_loss
    
    def train(self, train_dataloader, num_epochs=3, lr=5e-5):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                loss, distill_loss, hard_loss = self.compute_loss(batch)
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return self.student

# Utilisation
distiller = LLMDistiller(
    teacher_name="gpt2-large",
    student_name="gpt2",
    temperature=3.0,
    alpha=0.8
)

# Train
student_model = distiller.train(train_dataloader, num_epochs=3)
```

### 6.2 Distillation avec Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments
import torch.nn as nn

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Student forward
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        # Teacher forward (no grad)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            teacher_logits = outputs_teacher.logits
        
        # Distillation
        loss, _, _ = distillation_loss(
            student_logits,
            teacher_logits,
            inputs['labels'],
            self.temperature,
            self.alpha
        )
        
        return (loss, outputs_student) if return_outputs else loss

# Utilisation
training_args = TrainingArguments(
    output_dir="./distilled-model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    logging_steps=100,
)

trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    temperature=3.0,
    alpha=0.8
)

trainer.train()
```

---

## 7. Bonnes Pratiques & Conseils

### 7.1 Choix des hyperparamètres

| Hyperparamètre | Valeurs typiques | Conseil |
|----------------|------------------|---------|
| **Temperature** | 2.0 - 5.0 | Plus le vocab est grand, plus T doit être élevé |
| **Alpha** | 0.5 - 0.9 | Commencer à 0.7, augmenter si underfitting |
| **Learning rate** | 1e-5 - 5e-5 | Plus petit que pour training from scratch |
| **Batch size** | 16 - 128 | Dépend de la mémoire disponible |

### 7.2 Architecture du student

**Règles générales :**
- **Profondeur** : 1/2 ou 1/3 des couches du teacher
- **Largeur** : même hidden dimension ou 1/2
- **Attention heads** : proportionnel à la largeur
- **Vocabulaire** : identique au teacher

**Exemple :**
- Teacher : 24 layers, 1024 hidden, 16 heads
- Student : 6 layers, 512 hidden, 8 heads
- Ratio : ~1/8 des paramètres

### 7.3 Données d'entraînement

**Options :**
1. **Même dataset que le teacher** : Idéal si disponible
2. **Dataset différent** : Fonctionne si domaine similaire
3. **Données synthétiques** : Teacher génère des exemples

**Volume nécessaire :**
- Au minimum : 10-100M tokens
- Optimal : 1-10B tokens
- Dépend de la taille du student

### 7.4 Évaluation

**Métriques importantes :**
- **Perplexity** : Mesure de base de la qualité
- **Task-specific metrics** : BLEU, ROUGE, accuracy selon la tâche
- **Inference speed** : speedup vs teacher
- **Model size** : compression ratio

**Comparaisons :**
- Student vs Teacher (gap de performance)
- Student distillé vs Student entraîné from scratch
- Speed/accuracy trade-off

### 7.5 Debugging

**Problèmes courants :**

1. **Student ne converge pas**
   - Réduire alpha (plus de poids sur hard loss)
   - Réduire learning rate
   - Augmenter température

2. **Gap de performance trop important**
   - Augmenter alpha (plus de distillation)
   - Ajouter distillation des couches intermédiaires
   - Utiliser plus de données
   - Augmenter la taille du student

3. **Overfitting au teacher**
   - Réduire alpha
   - Ajouter du dropout
   - Early stopping

---

## 8. Ressources & Références

### 8.1 Papers fondamentaux

**Knowledge Distillation original :**
- [Hinton et al. (2015) - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
  - Le paper original qui introduit la distillation moderne

**Distillation pour LLM :**
- [Sanh et al. (2019) - DistilBERT](https://arxiv.org/abs/1910.01108)
  - Premier grand succès de distillation pour transformers
  
- [Jiao et al. (2020) - TinyBERT](https://arxiv.org/abs/1909.10351)
  - Distillation avec couches intermédiaires
  
- [Gu et al. (2023) - Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
  - Vue d'ensemble moderne pour les LLM

**Techniques avancées :**
- [Li et al. (2022) - On-Policy Distillation](https://arxiv.org/abs/2210.14984)
  - Distillation on-policy pour meilleure généralisation
  
- [Agarwal et al. (2024) - Black-Box Language Model Distillation](https://arxiv.org/abs/2401.00001)
  - Distillation sans accès aux logits (APIs)

### 8.2 Implémentations

**Librairies :**
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
  - Support natif pour distillation
  
- [TextBrewer](https://github.com/airaria/TextBrewer)
  - Framework dédié à la distillation de NLP
  
- [Neural Compressor](https://github.com/intel/neural-compressor)
  - Optimisation et distillation Intel

**Tutoriels pratiques :**
- [Hugging Face: Knowledge Distillation Tutorial](https://huggingface.co/docs/transformers/training#knowledge-distillation)
- [PyTorch: Model Distillation](https://pytorch.org/tutorials/intermediate/knowledge_distillation_tutorial.html)

### 8.3 Modèles distillés populaires

- **DistilBERT** : 66M params, 97% performance de BERT
- **DistilGPT-2** : 82M params, distillé de GPT-2
- **TinyLLaMA** : 1.1B params, entraîné/distillé sur style LLaMA
- **Phi-2** : 2.7B params (Microsoft), performances d'un 7B
- **MobileBERT** : Optimisé pour mobile

### 8.4 Outils de benchmarking

- [HELM](https://crfm.stanford.edu/helm/) : Évaluation holistique des LLM
- [EleutherAI LM Eval](https://github.com/EleutherAI/lm-evaluation-harness) : Suite de tests standardisés
- [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

## 9. Conclusion

### Points clés à retenir

☑ **La distillation permet de créer des modèles 2-10x plus petits avec 95-99% des performances**

☑ **La température T adoucit les distributions et révèle la "dark knowledge"**

☑ **La loss combine distillation (soft) et cross-entropy (hard) avec un coefficient α**

☑ **Pour les LLM, on peut distiller token-by-token, au niveau séquence, ou sur les couches internes**

☑ **L'implémentation nécessite : teacher figé, température 2-5, α = 0.5-0.9, scaling par T²**

### Checklist pour implémenter une distillation

 1. [ ] Charger teacher (figé) et student (trainable)
 2. [ ] Choisir température (2-5) et alpha (0.5-0.9)
 3. [ ] Implémenter loss : KL divergence + cross-entropy
 4. [ ] Scaler la distillation loss par T²
 5. [ ] Utiliser même learning rate que fine-tuning (ou plus petit)
 6. [ ] Monitorer distillation loss ET hard loss séparément
 7. [ ] Évaluer sur métriques task-specific + perplexity
 8. [ ] Mesurer speedup et compression ratio

### Prochaines étapes

Pour aller plus loin, explorer :
- Distillation de modèles multimodaux (vision + langage)
- Quantization + Distillation (combo compression)
- Distillation pour reasoning spécialisé (math, code)
- Distillation continue (teacher qui évolue)
