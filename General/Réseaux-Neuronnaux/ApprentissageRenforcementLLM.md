# Apprentissage par Renforcement pour LLM : De l'Alignement à la Mise à Jour Continue

---

## Table des Matières

## [1. Rappel : Apprentissage par Renforcement (RL) - Les Bases](#1-rappel--apprentissage-par-renforcement-rl---les-bases)
- [1.1 Concepts fondamentaux](#11-concepts-fondamentaux)
- [1.2 Équation de Bellman et fonction de valeur](#12-équation-de-bellman-et-fonction-de-valeur)
- [1.3 Policy Gradient](#13-policy-gradient)

## [2. RLHF : Reinforcement Learning from Human Feedback](#2-rlhf--reinforcement-learning-from-human-feedback)
- [2.1 Vue d'ensemble du pipeline](#21-vue-densemble-du-pipeline)
- [2.2 Étape 1 : Supervised Fine-Tuning (SFT)](#22-étape-1--supervised-fine-tuning-sft)
- [2.3 Étape 2 : Reward Modeling](#23-étape-2--reward-modeling)
- [2.4 Étape 3 : RL avec PPO](#24-étape-3--rl-avec-ppo)
- [2.5 Pourquoi RLHF fonctionne pour l'alignement](#25-pourquoi-rlhf-fonctionne-pour-lalignement)

## [3. Alternatives Modernes à RLHF](#3-alternatives-modernes-à-rlhf)
- [3.1 DPO (Direct Preference Optimization)](#31-dpo-direct-preference-optimization)
- [3.2 IPO (Identity Preference Optimization)](#32-ipo-identity-preference-optimization)
- [3.3 RLAIF (RL from AI Feedback)](#33-rlaif-rl-from-ai-feedback)
- [3.4 Comparaison pratique : Quand utiliser quoi ?](#34-comparaison-pratique--quand-utiliser-quoi-)

## [4. Maintenir un Modèle à Jour : Continual Learning](#4-maintenir-un-modèle-à-jour--continual-learning)
- [4.1 Le problème : Catastrophic Forgetting](#41-le-problème--catastrophic-forgetting)
- [4.2 Stratégies de mise à jour](#42-stratégies-de-mise-à-jour)
- [4.3 Experience Replay et méthodes de régularisation](#43-experience-replay-et-méthodes-de-régularisation)
- [4.4 Parameter-Efficient Fine-Tuning : Au-delà de LoRA](#44-parameter-efficient-fine-tuning--au-delà-de-lora)

## [5. Cas Pratiques et Recommandations](#5-cas-pratiques-et-recommandations)
- [5.1 Pipeline pour modèle en production (type ChatGPT)](#51-pipeline-pour-modèle-en-production-type-chatgpt)
- [5.2 Pipeline pour modèle interne d'entreprise](#52-pipeline-pour-modèle-interne-dentreprise)
- [5.3 Métriques et monitoring](#53-métriques-et-monitoring)

## [Récapitulatif et Recommandations Finales](#récapitulatif-et-recommandations-finales)
- [Pour ton cas spécifique (entreprise, modèle interne)](#pour-ton-cas-spécifique-entreprise-modèle-interne)
- [Ressources et Papers clés](#ressources-et-papers-clés)
---

## 1. Rappel : Apprentissage par Renforcement - Les Bases

### 1.1 Concepts fondamentaux

Le RL repose sur l'interaction **agent ↔ environnement** :

```
Agent (LLM)
    ↓ action (génération de tokens)
Environnement (humain, scoring model)
    ↓ reward (score de qualité)
Agent (mise à jour des poids)
```

**Vocabulaire clé :**
- **État (s)** : contexte actuel (prompt + tokens générés jusqu'ici)
- **Action (a)** : choix d'un token
- **Récompense (r)** : feedback sur la qualité de la génération
- **Politique π(a|s)** : probabilité de choisir l'action a dans l'état s (= le LLM lui-même)
- **Retour (G)** : somme des récompenses futures $$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**Objectif du RL :** Maximiser l'espérance des récompenses cumulées

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]$$

où $$\theta$$ = paramètres du modèle, $$\tau$$ = trajectoire (séquence état-action-récompense)

### 1.2 Équation de Bellman et fonction de valeur

La **fonction de valeur** $$V^\pi(s)$$ estime "à quel point il est bon d'être dans l'état s" :

$$V^\pi(s) = \mathbb{E}_{\pi} \left[ G_t | s_t = s \right]$$

La **fonction Q** $$Q^\pi(s, a)$$ estime "à quel point il est bon de faire l'action a dans l'état s" :

$$Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ r_t + \gamma V^\pi(s_{t+1}) | s_t = s, a_t = a \right]$$

**Pourquoi c'est important pour les LLM ?**
- Ces fonctions permettent d'évaluer si une génération est "sur la bonne voie"
- On peut entraîner un modèle de récompense qui apprend $$Q(s, a)$$ → Reward Model

### 1.3 Policy Gradient

L'idée : modifier directement la politique π pour augmenter $$J(\theta)$$.

**Théorème du gradient de politique :**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

**Intuition :** 
- Si une action mène à une bonne récompense ($$G_t$$ élevé), on augmente sa probabilité
- Si une action mène à une mauvaise récompense, on la diminue
- C'est la base de PPO (Proximal Policy Optimization)

---

## 2. RLHF : Reinforcement Learning from Human Feedback

### 2.1 Vue d'ensemble du pipeline

Le RLHF se décompose en **3 étapes** :

```
1. SFT (Supervised Fine-Tuning)
   ↓
   Modèle de base → Modèle SFT (suit les instructions)

2. Reward Modeling
   ↓
   Comparaisons humaines → Modèle de récompense

3. RL Optimization (PPO)
   ↓
   Modèle SFT + Reward Model → Modèle aligné final
```

**Pourquoi 3 étapes et pas du RL direct ?**
- Le feedback humain est rare et coûteux
- Il faut d'abord un modèle "compétent" (SFT) avant de l'affiner
- Le reward model permet de "démultiplier" le feedback humain

### 2.2 Étape 1 : Supervised Fine-Tuning (SFT)

**Objectif :** Apprendre au modèle à suivre des instructions à partir d'exemples de haute qualité.

**Données :** Paires (prompt, réponse idéale) écrites par des humains

**Loss classique :** Cross-entropy sur les tokens de la réponse

$$\mathcal{L}_{\text{SFT}} = - \sum_{i=1}^{n} \log P_\theta(y_i | y_{<i}, x)$$

où $$x$$ = prompt, $$y$$ = réponse cible

**En pratique :**
```python
# Exemple simplifié avec Hugging Face
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("base-llm")
trainer = Trainer(model=model, train_dataset=sft_dataset)
trainer.train()  # Standard supervised learning
```

**Points clés :**
- Le modèle SFT devient le "point de départ" pour le RL
- Qualité des données SFT = crucial (garbage in, garbage out)
- Typiquement 10k-100k exemples de haute qualité

### 2.3 Étape 2 : Reward Modeling

**Problème :** Comment quantifier automatiquement "c'est une bonne réponse" ?

**Solution :** Entraîner un modèle qui prédit les préférences humaines

**Collecte de données :**
1. Générer plusieurs réponses pour un même prompt avec le modèle SFT
2. Des humains classent les réponses : A > B > C > D
3. On obtient des paires de comparaison : (prompt, réponse_gagnante, réponse_perdante)

**Modèle de récompense :**

Architecture : LLM (souvent même taille que le modèle SFT) avec une tête de régression

$$r_\theta(x, y) \in \mathbb{R}$$

où $$r_\theta$$ est le score de qualité de la réponse $$y$$ au prompt $$x$$

**Loss : Bradley-Terry Model**

$$\mathcal{L}_{\text{RM}} = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

où :
- $$y_w$$ = réponse préférée (winner)
- $$y_l$$ = réponse rejetée (loser)
- $$\sigma$$ = fonction sigmoïde

**Intuition :** Le modèle apprend à donner un score plus élevé aux réponses préférées

```python
# Architecture du Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # LLM (frozen ou fine-tuné)
        self.reward_head = nn.Linear(hidden_size, 1)  # Score unique
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # Dernier token
        reward = self.reward_head(last_hidden)  # Score réel
        return reward

# Loss
def reward_loss(reward_model, prompt, response_win, response_lose):
    r_win = reward_model(prompt + response_win)
    r_lose = reward_model(prompt + response_lose)
    loss = -torch.log(torch.sigmoid(r_win - r_lose))
    return loss.mean()
```

**Points clés :**
- Le reward model capture les préférences humaines implicites
- Souvent le goulot d'étranglement : qualité = qualité du RM
- Typiquement 50k-500k comparaisons humaines

### 2.4 Étape 3 : RL avec PPO

**Objectif :** Optimiser le modèle SFT pour maximiser les récompenses du RM

**Pourquoi PPO (Proximal Policy Optimization) ?**
- Stabilité : évite les mises à jour trop agressives
- Efficacité : réutilise les données (important car générer = coûteux)

**Fonction objectif complète :**

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_{\tau} \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] - \beta \, D_{\text{KL}}(\pi_\theta || \pi_{\text{SFT}})$$

**Décomposition :**

1. **Ratio de probabilité** :
   $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}$$
   → Mesure combien la nouvelle politique diffère de l'ancienne

2. **Avantage** $$\hat{A}_t$$ :  
   $$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$  
   → "Cette action est-elle meilleure que la moyenne ?"

3. **Clipping** :
   $$\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$$ avec $$\epsilon \approx 0.2$$
   → Empêche les changements trop brusques

4. **Pénalité KL** :
   $$\beta \, D_{\text{KL}}(\pi_\theta || \pi_{\text{SFT}})$$
   → Force le modèle à rester proche du modèle SFT initial

**Pourquoi la pénalité KL est cruciale ?**
- Sans elle, le modèle peut "exploiter" le reward model (mode collapse)
- Exemple : générer toujours "Je suis désolé, je ne peux pas répondre" (score élevé mais inutile)
- La contrainte KL maintient la "santé mentale" du modèle

**Boucle d'entraînement PPO :**

```python
# Pseudo-code simplifié
for epoch in range(num_epochs):
    # 1. Générer des trajectoires avec la politique actuelle
    prompts = sample_prompts(batch_size)
    responses = policy_model.generate(prompts)
    
    # 2. Calculer les récompenses avec le Reward Model
    rewards = reward_model(prompts, responses)
    
    # 3. Calculer la pénalité KL avec le modèle SFT de référence
    kl_penalty = compute_kl(policy_model, sft_model, prompts, responses)
    
    # 4. Récompense finale
    final_rewards = rewards - beta * kl_penalty
    
    # 5. Calculer les avantages
    advantages = compute_advantages(final_rewards)
    
    # 6. Mise à jour PPO
    for _ in range(ppo_epochs):
        ratio = policy_model.prob(responses) / old_policy.prob(responses)
        clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Hyperparamètres typiques :**
- $$\beta$$ (KL penalty) : 0.01 - 0.1
- $$\epsilon$$ (clipping) : 0.2
- Learning rate : 1e-6 (beaucoup plus petit que SFT)
- Batch size : 32-128 prompts

### 2.5 Pourquoi RLHF fonctionne pour l'alignement

**Intuition profonde :**

1. **Capture des préférences complexes**
   - Les humains peuvent facilement dire "A > B" mais difficile d'écrire une règle
   - Le RM apprend ces patterns implicites (utilité, sécurité, style)

2. **Exploration guidée**
   - Le RL explore l'espace des réponses possibles
   - Le RM guide vers les zones "alignées avec les humains"

3. **Optimisation directe de l'objectif final**
   - SFT = imiter des exemples (peut être rigide)
   - RLHF = maximiser la satisfaction utilisateur (plus flexible)

**Limites :**
- **Coût** : annotation humaine très chère
- **Instabilité** : RL peut être capricieux (divergence, mode collapse)
- **Biais** : le RM hérite des biais des annotateurs
- **Complexité** : pipeline multi-étapes difficile à débugger

---

## 3. Alternatives Modernes à RLHF

### 3.1 DPO (Direct Preference Optimization)

**Problème avec RLHF :** Pipeline complexe (SFT → RM → PPO) et instable

**Idée de DPO :** "Et si on sautait l'étape du Reward Model ?"

**Insight mathématique :**

En RL optimal, on peut montrer que :

$$\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

où $$\pi_{\text{ref}}$$ = politique de référence (modèle SFT), $$\beta$$ = température

En inversant cette équation, on obtient :

$$r(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)$$

**Loss DPO :**

$$\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

**Intuition simple :**
- On augmente directement la probabilité des réponses préférées $$y_w$$
- On diminue celle des réponses rejetées $$y_l$$
- Tout ça en restant proche de $$\pi_{\text{ref}}$$ (contraint par $$\beta$$)

**Avantages DPO vs RLHF :**
- ✅ Pas de Reward Model séparé → plus simple
- ✅ Stable (c'est du supervised learning déguisé)
- ✅ Moins de mémoire (pas besoin de stocker le RM)
- ✅ Plus rapide à converger

**Inconvénients :**
- ❌ Moins flexible (on ne peut pas ajuster le reward model après)
- ❌ Nécessite toujours des paires de comparaisons
- ❌ Peut être moins performant sur des tâches très complexes

```python
# Implémentation simplifiée de la loss DPO
def dpo_loss(policy_model, ref_model, prompt, response_win, response_lose, beta=0.1):
    # Log-probs de la politique actuelle
    logp_win = policy_model.log_prob(prompt, response_win)
    logp_lose = policy_model.log_prob(prompt, response_lose)
    
    # Log-probs du modèle de référence
    with torch.no_grad():
        ref_logp_win = ref_model.log_prob(prompt, response_win)
        ref_logp_lose = ref_model.log_prob(prompt, response_lose)
    
    # Ratios log
    logits_win = beta * (logp_win - ref_logp_win)
    logits_lose = beta * (logp_lose - ref_logp_lose)
    
    # Loss
    loss = -torch.log(torch.sigmoid(logits_win - logits_lose))
    return loss.mean()
```

**Quand utiliser DPO ?**
- Ressources limitées (GPU, temps, budget annotation)
- Stabilité prioritaire
- Pas besoin d'itérer sur le reward model

### 3.2 IPO (Identity Preference Optimization)

**Problème avec DPO :** La loss peut être "trop gentille" avec les mauvaises réponses

**Idée IPO :** Pénaliser plus fortement les réponses rejetées

**Loss IPO :**

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} - \tau \right)^2 \right]$$

où $$\tau$$ = marge cible (typiquement 0.5-1.0)

**Différence avec DPO :**
- DPO : sigmoïde → asymptotiquement satisfaite
- IPO : MSE → force un gap $$\tau$$ entre winner et loser

**Intuition :**
- On veut que $$y_w$$ soit **significativement** meilleur que $$y_l$$
- Pas juste "un peu mieux"
- Utile quand les préférences sont très marquées

**Quand utiliser IPO ?**
- Données de préférence très claires (win/lose évidents)
- Besoin de réponses "sûres" (éviter les réponses limites)
- Cas d'usage critiques (médicaux, légaux)

### 3.3 RLAIF (RL from AI Feedback)

**Problème :** Annoter manuellement 500k comparaisons = $$

**Idée :** Remplacer les annotateurs humains par un LLM puissant

**Pipeline RLAIF :**

```
1. Génération de réponses candidates avec le modèle SFT
   ↓
2. Évaluation par un LLM juge (ex: GPT-4, Claude)
   Prompt: "Quelle réponse est meilleure ? A ou B ? Justifie."
   ↓
3. Création de paires (win, lose) automatiques
   ↓
4. RLHF classique OU DPO avec ces données
```

**Exemple de prompt pour le LLM juge :**

```text
Tu es un évaluateur expert. Compare les deux réponses suivantes au prompt :

Prompt: "{user_prompt}"

Réponse A: "{response_a}"
Réponse B: "{response_b}"

Critères d'évaluation :
1. Exactitude factuelle
2. Utilité pour l'utilisateur
3. Clarté et concision
4. Absence de biais ou contenu nuisible

Quelle réponse est meilleure ? Réponds par "A" ou "B" puis justifie.
```

**Avantages RLAIF :**
- ✅ **Scalabilité** : millions de comparaisons à faible coût
- ✅ **Consistance** : le LLM juge est cohérent (pas de fatigue humaine)
- ✅ **Rapidité** : itération beaucoup plus rapide

**Inconvénients :**
- ❌ Hérite des biais du LLM juge
- ❌ Peut manquer de nuances humaines
- ❌ "Alignment tax" : aligné sur un autre modèle, pas sur les humains

**Stratégies hybrides :**
1. **Bootstrap** : RLAIF initial → fine-tuning avec feedback humain
2. **Validation** : Échantillon humain pour vérifier la qualité du juge IA
3. **Ensemble** : Plusieurs LLM juges + vote majoritaire

**Quand utiliser RLAIF ?**
- Budget annotation limité
- Besoin de scalabilité
- Tâches objectives (code, maths) où le LLM juge est fiable

### 3.4 Comparaison pratique : Quand utiliser quoi ?

| Méthode | Complexité | Stabilité | Coût Annotation | Performance | Cas d'usage |
|---------|------------|-----------|-----------------|-------------|-------------|
| **RLHF (PPO)** | ⚠️⚠️⚠️ Haute | ⚠️ Moyenne | 💰💰💰 Élevé | 🏆🏆🏆 Excellente | Production haute performance (GPT-4, Claude) |
| **DPO** | ✅ Simple | ✅✅ Stable | 💰💰 Moyen | 🏆🏆 Très bonne | Modèles internes, R&D rapide |
| **IPO** | ✅ Simple | ✅✅ Stable | 💰💰 Moyen | 🏆🏆 Bonne | Cas d'usage critiques, sécurité |
| **RLAIF** | ✅✅ Moyenne | ✅✅ Stable | 💰 Faible | 🏆🏆 Bonne | Scalabilité, bootstrap |

**Arbre de décision :**

```
Budget annotation illimité + Team experte en RL ?
    └─ OUI → RLHF (PPO)
    └─ NON ↓

Données de préférence objectives (code, maths) ?
    └─ OUI → RLAIF + DPO
    └─ NON ↓

Besoin de stabilité maximale ?
    └─ OUI → DPO
    └─ NON ↓

Cas critique (médical, légal) ?
    └─ OUI → IPO ou RLHF avec validation humaine
    └─ NON → DPO (choix par défaut)
```

---

## 4. Maintenir un Modèle à Jour : Continual Learning

### 4.1 Le problème : Catastrophic Forgetting

**Scénario typique :**

```
Jour 0: Fine-tune GPT-4 sur données médicales 2024
    → Performance médecine: 95% | Performance générale: 85%

Jour 100: Fine-tune sur nouvelles données médicales 2025
    → Performance médecine: 96% | Performance générale: 60% ⚠️
```

**Qu'est-ce qui s'est passé ?**
- Le modèle a "oublié" ses capacités générales
- Les nouveaux gradients ont écrasé les anciens poids

**Pourquoi ça arrive ?**

Les réseaux de neurones apprennent en modifiant les poids :

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta \mathcal{L}_{\text{new task}}$$

Problème : $$\nabla_\theta \mathcal{L}_{\text{new}}$$ peut détruire ce qui a été appris pour les anciennes tâches

**Mesure du forgetting :**

$$\text{Forgetting} = \text{Acc}_{\text{old task before}} - \text{Acc}_{\text{old task after}}$$

Si forgetting > 10% → problème sérieux

### 4.2 Stratégies de mise à jour

**1. Naive Fine-Tuning (ce qu'il NE faut PAS faire)**

```python
# ❌ Mauvaise approche
model.load("model_v1.pth")
trainer.train(model, new_data)  # Écrase les anciens poids
model.save("model_v2.pth")
```

Résultat : catastrophic forgetting garanti

**2. Joint Training (réentraînement complet)**

```python
# ✅ Fonctionne mais coûteux
all_data = old_data + new_data
model = train_from_scratch(all_data)
```

**Avantages :**
- Pas de forgetting (le modèle voit toutes les données)
- Performance optimale

**Inconvénients :**
- Coût computationnel prohibitif (réentraîner tout depuis zéro)
- Nécessite de garder TOUTES les anciennes données (privacy, storage)

**Quand utiliser :**
- Mise à jour majeure (ex: annuelle)
- Budget GPU disponible
- Pas de contrainte de latence

**3. Sequential Fine-Tuning avec Experience Replay**

**Idée :** Mélanger anciennes et nouvelles données lors du fine-tuning

```python
# ✅ Bon compromis
model.load("model_v1.pth")

# Échantillonner des données anciennes
replay_data = sample(old_data, ratio=0.3)  # 30% anciennes données
mixed_data = new_data + replay_data

trainer.train(model, mixed_data)
model.save("model_v2.pth")
```

**Ratio typique :** 20-40% anciennes données

**Stratégies d'échantillonnage :**
- **Uniforme** : échantillon aléatoire
- **Difficile** : garder les exemples où le modèle performait mal
- **Représentatif** : stratification par domaine/tâche

**Avantages :**
- ✅ Réduit fortement le forgetting
- ✅ Coût raisonnable (pas de réentraînement complet)

**Inconvénients :**
- ❌ Nécessite de stocker des anciennes données
- ❌ Pas de garantie théorique de non-forgetting

**4. Regularization-Based Methods**

**EWC (Elastic Weight Consolidation)**

**Idée :** Certains poids sont "importants" pour les anciennes tâches → les protéger

**Loss EWC :**

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

où :
- $$\mathcal{L}_{\text{new}}$$ = loss sur nouvelles données
- $$F_i$$ = "importance" du poids $$i$$ (matrice de Fisher)
- $$\theta^*$$ = poids après apprentissage de l'ancienne tâche
- $$\lambda$$ = force de la régularisation

**Intuition :**
- $$F_i$$ élevé → poids important → grande pénalité si on le change
- $$F_i$$ faible → poids peu important → liberté de le modifier

**Calcul de la matrice de Fisher :**

$$F_i = \mathbb{E}_{x \sim \text{old data}} \left[ \left( \frac{\partial \log p(x | \theta)}{\partial \theta_i} \right)^2 \right]$$

```python
# Implémentation simplifiée EWC
def compute_fisher(model, old_data):
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()
    
    for batch in old_data:
        model.zero_grad()
        loss = model(batch).log_prob.mean()  # Log-likelihood
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher[name] += param.grad ** 2  # Accumulation
    
    # Normalisation
    for name in fisher:
        fisher[name] /= len(old_data)
    
    return fisher

# Fine-tuning avec EWC
def train_with_ewc(model, new_data, old_params, fisher, lambda_ewc=1000):
    for batch in new_data:
        loss_new = model(batch).loss
        
        # Pénalité EWC
        loss_ewc = 0
        for name, param in model.named_parameters():
            loss_ewc += (fisher[name] * (param - old_params[name]) ** 2).sum()
        
        total_loss = loss_new + (lambda_ewc / 2) * loss_ewc
        total_loss.backward()
        optimizer.step()
```

**Avantages :**
- ✅ Pas besoin de stocker les anciennes données (juste Fisher + anciens poids)
- ✅ Base théorique solide

**Inconvénients :**
- ❌ Calcul de Fisher coûteux
- ❌ Peut être trop conservateur (limite l'apprentissage de nouvelles tâches)

**5. Parameter-Efficient Fine-Tuning (PEFT)**

**Idée :** Ne modifier qu'une petite partie des poids → le reste est "protégé"

**LoRA (Low-Rank Adaptation)**

Au lieu de modifier directement $$W \in \mathbb{R}^{d \times k}$$, on apprend :

$$W' = W_0 + \Delta W = W_0 + BA$$

où :
- $$W_0$$ = poids pré-entraînés (gelés)
- $$B \in \mathbb{R}^{d \times r}$$, $$A \in \mathbb{R}^{r \times k}$$ avec $$r \ll d, k$$ (rank faible)

**Exemple numérique :**
- Modèle 7B paramètres
- Avec LoRA rank=8 : seulement 4M paramètres entraînables (0.05% !)

```python
# Implémentation LoRA pour mise à jour continue
class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base = base_layer
        self.base.requires_grad_(False)  # Geler les poids originaux
        
        d, k = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(rank, k))
        self.lora_B = nn.Parameter(torch.zeros(d, rank))
        
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out

# Mise à jour continue avec LoRA
# Version 1: LoRA initial
model_v1 = add_lora_adapters(base_model, rank=8, name="lora_v1")
train(model_v1, data_2024)

# Version 2: Nouveau LoRA (modèle de base inchangé)
model_v2 = add_lora_adapters(base_model, rank=8, name="lora_v2")
train(model_v2, data_2025)

# On peut garder les deux et switcher selon le contexte !
# Ou fusionner: model_merged = merge_loras([lora_v1, lora_v2])
```

**Avantages de LoRA pour continual learning :**
- ✅ **Pas de forgetting** : le modèle de base reste intact
- ✅ **Multi-tâches** : un adaptateur par tâche/période
- ✅ **Efficace** : 100x moins de paramètres à entraîner
- ✅ **Modularité** : on peut combiner/retirer des adaptateurs

**Stratégies multi-adaptateurs :**

```
Base Model (gelé)
    ├─ LoRA_general (toujours actif)
    ├─ LoRA_2024 (données historiques)
    ├─ LoRA_2025 (données récentes)
    └─ LoRA_domain_specific (expertise métier)

À l'inférence: output = base + α·LoRA_general + β·LoRA_2025 + γ·LoRA_domain
```

**Poids typiques :** $$\alpha=0.5$$, $$\beta=0.3$$, $$\gamma=0.2$$ (ajustables)

### 4.3 Experience Replay et méthodes de régularisation

**Résumé comparatif des stratégies :**

| Méthode | Forgetting | Coût GPU | Stockage Data | Complexité | Meilleur pour |
|---------|-----------|----------|---------------|------------|---------------|
| **Naive FT** | ❌❌❌ Très élevé | ✅ Faible | ✅ Aucun | ✅ Simple | ⚠️ À éviter |
| **Joint Training** | ✅✅✅ Aucun | ❌❌ Très élevé | ❌❌ Total | ⚠️ Moyen | Maj annuelles |
| **Replay (30%)** | ✅✅ Faible | ✅✅ Moyen | ⚠️ Partiel | ✅ Simple | Production standard |
| **EWC** | ✅✅ Faible | ✅✅ Moyen | ✅ Aucun | ⚠️⚠️ Complexe | Recherche |
| **LoRA** | ✅✅✅ Aucun | ✅✅✅ Faible | ✅ Aucun | ✅✅ Moyen | **Recommandé pour prod** |

**Recommandation générale pour ton cas (modèle interne en prod) :**

```
Configuration optimale pour mise à jour continue :

1. Modèle de base : GPT-4 / Llama 3 / Mistral (gelé)
2. LoRA adapters par version :
   - lora_baseline : training initial
   - lora_update_Q1 : données Q1 2026
   - lora_update_Q2 : données Q2 2026
   ...

3. Stratégie d'entraînement :
   - Nouveau quarter → Nouveau LoRA
   - Experience replay 20% si nécessaire
   - Validation sur ensemble de test "historique"

4. Monitoring :
   - Métriques sur données anciennes (détection forgetting)
   - Métriques sur nouvelles données (performance)
   - Si forgetting > 5% → augmenter replay ou ajuster LoRA weights
```

### 4.4 Parameter-Efficient Fine-Tuning : Au-delà de LoRA

**Autres méthodes PEFT :**

**1. Adapters**

Insérer de petits modules entre les couches du transformeur :

```
Transformer Layer
    ↓
LayerNorm
    ↓
Self-Attention (gelé)
    ↓
[Adapter Module: Down-project → ReLU → Up-project]  ← Entraînable
    ↓
LayerNorm
    ↓
Feed-Forward (gelé)
    ↓
[Adapter Module]  ← Entraînable
```

Adapters ≈ 3-5% de paramètres additionnels

**2. Prefix Tuning**

Apprendre des "tokens virtuels" à préfixer au contexte :

$$\text{Input} = [\text{Prefix}_1, \text{Prefix}_2, \ldots, \text{Prefix}_k, \text{User prompt}]$$

Les $$\text{Prefix}_i$$ sont des embeddings entraînables, le reste du modèle est gelé

**3. (IA)³ : Infused Adapter by Inhibiting and Amplifying Inner Activations**

Modifier sélectivement certaines activations internes (scaling vectors)

**Comparaison PEFT :**

| Méthode | Paramètres | Performances | Facilité | Cas d'usage |
|---------|-----------|--------------|----------|-------------|
| **LoRA** | 0.1-1% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Défaut, production |
| **Adapters** | 3-5% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Multi-tâches |
| **Prefix** | 0.01-0.1% | ⭐⭐⭐ | ⭐⭐⭐ | Mémoire limitée |
| **(IA)³** | <0.01% | ⭐⭐⭐⭐ | ⭐⭐ | Recherche |

**Conseil :** Commence par LoRA, c'est le meilleur rapport performance/simplicité

---

## 5. Cas Pratiques et Recommandations

### 5.1 Pipeline pour modèle en production (type ChatGPT)

**Architecture système :**

```
Production System (ChatGPT-like)
│
├─ Modèle Principal (v3.5)
│  ├─ Base Model (gelé, snapshot annuel)
│  ├─ LoRA_alignment_v3 (RLHF général)
│  └─ LoRA_knowledge_2026_Q1 (mise à jour connaissances)
│
├─ Reward Model (v2.3)
│  └─ Mis à jour hebdomadairement avec nouvelles annotations
│
├─ Data Collection Pipeline
│  ├─ User interactions → Logging
│  ├─ Thumbs up/down → Preference DB
│  └─ Adversarial red-teaming → Safety DB
│
└─ Continuous Improvement Loop
   ├─ Hebdomadaire: Mise à jour Reward Model
   ├─ Mensuel: Nouveau LoRA avec DPO (données 30 derniers jours)
   └─ Annuel: Réentraînement complet (joint training)
```

**Workflow mensuel détaillé :**

```python
# MOIS N : Collecte et préparation

# 1. Extraction des interactions
interactions = db.query("""
    SELECT prompt, response, feedback 
    FROM user_logs 
    WHERE timestamp BETWEEN '2026-01-01' AND '2026-01-31'
    AND feedback IS NOT NULL
""")

# 2. Filtrage qualité
high_quality = filter_quality(interactions, min_confidence=0.8)

# 3. Augmentation avec RLAIF (pour scaler)
judge_model = load_model("gpt-4")
rlaif_comparisons = []

for prompt, responses in batch_generate(high_quality, n=4):
    rankings = judge_model.rank(prompt, responses)
    pairs = create_pairs(rankings)  # (winner, loser)
    rlaif_comparisons.extend(pairs)

# 4. Combinaison humain + IA
final_dataset = human_feedback + sample(rlaif_comparisons, ratio=3.0)
# Ratio 1:3 (humain:IA) pour garder qualité tout en scalant

# 5. Entraînement nouveau LoRA avec DPO
new_lora = train_lora_dpo(
    base_model=production_model,
    preference_data=final_dataset,
    rank=16,
    learning_rate=1e-4,
    epochs=3
)

# 6. Validation A/B test
ab_test_results = run_ab_test(
    model_a=production_model,
    model_b=production_model + new_lora,
    duration_hours=48,
    traffic_split=0.05  # 5% sur nouveau modèle
)

if ab_test_results['win_rate_b'] > 0.52:  # Amélioration >2%
    deploy(new_lora, name="lora_2026_02")
    update_config(weights={'base': 0.4, 'lora_alignment': 0.3, 'lora_2026_02': 0.3})
```

**Métriques de monitoring :**

```python
# Dashboard en temps réel
metrics = {
    # Performance utilisateur
    'user_satisfaction': 4.2,  # /5 (thumbs up/down)
    'response_acceptance_rate': 0.87,  # Copier-coller, continuer conversation
    
    # Qualité technique
    'safety_violations': 0.002,  # <0.5% acceptable
    'hallucination_rate': 0.05,  # Détecté par fact-checker
    'latency_p95': 1.2,  # secondes
    
    # Continual learning
    'forgetting_on_benchmarks': 0.03,  # 3% drop sur MMLU acceptable
    'improvement_on_recent_topics': 0.12,  # 12% gain sur topics 2026
    
    # Monitoring LoRA
    'lora_weight_l2_norm': 0.15,  # Trop élevé = risque instabilité
    'kl_divergence_from_base': 0.08  # Trop élevé = drift dangereux
}

# Alertes automatiques
if metrics['forgetting_on_benchmarks'] > 0.05:
    alert("Catastrophic forgetting detected! Review last update.")
if metrics['safety_violations'] > 0.005:
    rollback_to_previous_version()
```

### 5.2 Pipeline pour modèle interne d'entreprise

**Scénario : Entreprise B2B SaaS, assistant IA pour support client**

**Contraintes spécifiques :**
- Données propriétaires sensibles (impossibilité RLAIF avec API externe)
- Budget GPU limité
- Besoin de mises à jour fréquentes (nouveaux produits, FAQ)
- Pas d'équipe RL experte

**Architecture recommandée :**

```
Internal Enterprise Assistant
│
├─ Base: Llama-3-8B (open-source, hébergé on-premise)
│
├─ Fine-tuning layers:
│  ├─ LoRA_company_knowledge (FAQ, docs produit)
│  ├─ LoRA_conversation_style (ton, guidelines)
│  └─ LoRA_recent_updates (3 derniers mois)
│
├─ Data sources:
│  ├─ Historical tickets (résolutions validées)
│  ├─ Employee feedback (rating 1-5 sur réponses)
│  └─ Product docs (mis à jour par PM)
│
└─ Update cadence:
   ├─ Quotidien: Indexation nouveaux docs (RAG, pas de fine-tuning)
   ├─ Hebdomadaire: Agrégation feedback → DPO dataset
   └─ Mensuel: Nouveau LoRA_recent_updates
```

**Pipeline hebdomadaire simplifié :**

```python
# SEMAINE N : Collecte feedback employés

# 1. Récupération interactions avec feedback
feedback_data = internal_db.get_rated_conversations(last_7_days=True)
# Format: {prompt, response, rating (1-5), employee_comment}

# 2. Conversion en paires de préférence
# Stratégie: rating ≥4 = good, rating ≤2 = bad
good_responses = [x for x in feedback_data if x.rating >= 4]
bad_responses = [x for x in feedback_data if x.rating <= 2]

preference_pairs = []
for prompt in set([x.prompt for x in feedback_data]):
    good = [x for x in good_responses if x.prompt == prompt]
    bad = [x for x in bad_responses if x.prompt == prompt]
    
    # Créer toutes les paires (good, bad) pour ce prompt
    for g in good:
        for b in bad:
            preference_pairs.append({
                'prompt': prompt,
                'chosen': g.response,
                'rejected': b.response
            })

# 3. Augmentation avec génération synthétique (pas de RLAIF externe)
# Utiliser le modèle actuel pour générer des variations
for pair in preference_pairs[:100]:  # Sous-échantillon
    # Générer réponse alternative
    alt_response = current_model.generate(pair['prompt'], temperature=1.2)
    
    # Demander à un employé expert de la noter (batch hebdomadaire)
    # Ou : heuristique simple (longueur, présence mots-clés, etc.)
    if heuristic_quality(alt_response) < heuristic_quality(pair['rejected']):
        preference_pairs.append({
            'prompt': pair['prompt'],
            'chosen': pair['chosen'],
            'rejected': alt_response
        })

# 4. DPO training (léger)
if len(preference_pairs) >= 50:  # Minimum pour éviter overfitting
    weekly_lora = train_dpo(
        base_model=llama_3_8b,
        dataset=preference_pairs,
        rank=8,  # Petit rank pour limiter overfitting
        epochs=2,
        learning_rate=5e-5
    )
    
    # 5. Validation interne
    test_cases = load_test_suite("data/internal_test_cases.json")
    scores = evaluate(weekly_lora, test_cases)
    
    if scores['accuracy'] > current_model_accuracy * 0.98:  # Max 2% régression
        merge_lora_into_recent_updates(weekly_lora)
        log_deployment(version=f"week_{week_number}")
```

**Gestion des mises à jour produit :**

```python
# Nouveau produit lancé: "Product X"

# 1. PM fournit documentation
product_x_docs = load_markdown("docs/product_x/")

# 2. Génération automatique de Q&A synthétiques
qa_pairs = generate_qa_from_docs(product_x_docs, llm=llama_3_70b)
# Exemple output:
# Q: "Comment configurer Product X pour un usage multi-tenant ?"
# A: "Product X supporte le multi-tenancy via..."

# 3. Review humain (PM valide les Q&A)
validated_qa = human_review(qa_pairs)

# 4. Fine-tuning dédié (pas DPO, juste SFT car données propres)
product_x_lora = train_sft(
    base_model=current_model,
    dataset=validated_qa,
    rank=4,  # Petit LoRA spécialisé
    epochs=5
)

# 5. Ajout à la pile de LoRAs
model_config['loras']['product_x'] = product_x_lora
model_config['weights']['product_x'] = 0.2  # 20% de poids

# Le modèle répond maintenant aux questions sur Product X !
```

**Avantages de cette approche pour une entreprise :**
- ✅ **Pas de dépendance externe** : tout on-premise
- ✅ **Coût contrôlé** : Llama-3-8B + LoRA = faisable sur 1-2 GPUs
- ✅ **Flexibilité** : ajout/retrait de LoRA selon produits actifs
- ✅ **Traçabilité** : chaque LoRA = une version/feature précise
- ✅ **Pas de forgetting** : base model jamais modifié

### 5.3 Métriques et monitoring

**Tableau de bord essentiel :**

**A. Métriques de performance**

```
┌─ Performance générale ───────────────────────────┐
│ MMLU (benchmark général)         : 82.3% → 81.9% │ ⚠️ -0.4%
│ TruthfulQA (hallucinations)      : 78.1% → 79.2% │ ✅ +1.1%
│ HumanEval (code)                 : 71.5% → 71.2% │ ⚠️ -0.3%
│                                                   │
│ Win rate vs baseline              : 54.2%        │ ✅ Significatif
│ User preference (A/B test)        : 52.1%        │ ✅ Amélioration
└───────────────────────────────────────────────────┘

┌─ Performance domaine spécifique ─────────────────┐
│ Internal FAQ accuracy            : 91.2% → 94.5% │ ✅ +3.3%
│ Product knowledge coverage       : 87.3% → 92.1% │ ✅ +4.8%
│ Legal compliance (safety)        : 99.1% → 99.3% │ ✅ +0.2%
└───────────────────────────────────────────────────┘
```

**Règle d'or :** Tolérer max 2-3% de régression sur benchmarks généraux si gain >5% sur tâche cible

**B. Métriques d'alignement**

```python
# Mesurer le drift du modèle
def compute_alignment_metrics(model_new, model_base, test_prompts):
    metrics = {}
    
    # 1. KL Divergence (mesure le "drift" probabiliste)
    kl_divs = []
    for prompt in test_prompts:
        logits_new = model_new(prompt, return_logits=True)
        logits_base = model_base(prompt, return_logits=True)
        
        kl = F.kl_div(
            F.log_softmax(logits_new, dim=-1),
            F.softmax(logits_base, dim=-1),
            reduction='batchmean'
        )
        kl_divs.append(kl.item())
    
    metrics['kl_divergence_mean'] = np.mean(kl_divs)
    
    # 2. Output similarity (combien les réponses changent)
    similarities = []
    for prompt in test_prompts:
        response_new = model_new.generate(prompt)
        response_base = model_base.generate(prompt)
        
        # BLEU, ROUGE, ou embedding similarity
        sim = sentence_similarity(response_new, response_base)
        similarities.append(sim)
    
    metrics['output_similarity'] = np.mean(similarities)
    
    # 3. Safety/Alignment violations
    red_team_prompts = load_red_team_dataset()
    violations_new = count_violations(model_new, red_team_prompts)
    violations_base = count_violations(model_base, red_team_prompts)
    
    metrics['safety_regression'] = violations_new - violations_base
    
    return metrics

# Seuils d'alerte
ALERT_THRESHOLDS = {
    'kl_divergence_mean': 0.5,  # Au-delà = trop de changement
    'output_similarity': 0.7,   # En-dessous = trop différent
    'safety_regression': 5      # >5 violations = rollback
}
```

**C. Métriques de production**

Pour un modèle en production continue :

```
Métriques temps réel (dashboard):
├─ Latency:
│  ├─ p50: 0.8s
│  ├─ p95: 1.5s
│  └─ p99: 2.3s
│
├─ Quality (rolling 24h):
│  ├─ User satisfaction: 4.3/5 (👍/👎)
│  ├─ Response acceptance: 89% (continue conversation)
│  ├─ Regeneration rate: 11% (user clique "Régénérer")
│  └─ Safety flags: 0.002% (modération)
│
├─ System health:
│  ├─ GPU utilization: 78%
│  ├─ Throughput: 1200 req/min
│  └─ Error rate: 0.05%
│
└─ Learning metrics:
   ├─ Feedbacks collectés/jour: 1,250
   ├─ Paires de préférence créées: 340
   └─ Next update in: 4 days
```

**D. Validation avant déploiement**

**Checklist obligatoire :**

```
✅ Benchmarks généraux (MMLU, TruthfulQA): < 3% régression
✅ Benchmarks domaine cible: > 3% amélioration
✅ A/B test: win rate > 51% (significatif statistiquement)
✅ Red team: pas de nouvelle vulnérabilité safety
✅ Latency: < 10% d'augmentation
✅ KL divergence: < 0.5
✅ Manual review: 50 exemples validés par humains

Si TOUS ✅ → Déploiement progressif (1% → 10% → 50% → 100%)
Sinon → Iteration ou rollback
```

---

## Récapitulatif et Recommandations Finales

### Pour ton cas spécifique (entreprise, modèle interne)

**Setup recommandé :**

```
Base: Llama-3-8B ou Mistral-7B (open-source, contrôle total)

Stratégie de mise à jour:
1. Initial: SFT sur données propriétaires → modèle baseline
2. Amélioration continue: DPO avec feedback employés (hebdo/mensuel)
3. Nouvelles features: LoRA dédiés par produit/domaine
4. Pas de forgetting: Base model gelé + stack de LoRAs

Tools:
- Hugging Face TRL (pour DPO)
- PEFT library (pour LoRA)
- vLLM (pour serving efficace)
- Weights & Biases (pour monitoring)
```

**Checklist démarrage :**

Phase 1 - Setup (Semaine 1-2):
- [ ] Choisir base model (Llama-3-8B recommandé)
- [ ] Collecter données internes (tickets, FAQ, docs)
- [ ] Créer test suite de référence (100-200 exemples validés)
- [ ] Setup infrastructure (GPU, serving, logging)
Phase 2 - Baseline (Semaine 3-4):
- [ ] SFT initial sur données propriétaires
- [ ] Validation humaine sur échantillon
- [ ] Déploiement pilote (10 utilisateurs)
- [ ] Collecte premiers feedbacks
Phase 3 - Amélioration (Mensuel):
- [ ] Agréger feedbacks utilisateurs
- [ ] Créer dataset DPO (humain + synthétique)
- [ ] Train nouveau LoRA
- [ ] A/B test interne
- [ ] Déploiement si validation OK
Phase 4 - Monitoring (Continu):
- [ ] Dashboard métriques temps réel
- [ ] Alertes sur régression
- [ ] Review mensuel par experts
- [ ] Documentation des versions

### Ressources et Papers clés

**Papers fondamentaux :**

1. **RLHF**
   - "Training language models to follow instructions with human feedback" (InstructGPT, OpenAI 2022)
   - "Learning to summarize from human feedback" (Stiennon et al., 2020)

2. **DPO**
   - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)

3. **Continual Learning**
   - "Overcoming catastrophic forgetting in neural networks" (EWC, Kirkpatrick et al., 2017)
   - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

4. **RLAIF**
   - "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)

**Libraries Python essentielles :**

```python
# Installation
pip install transformers trl peft bitsandbytes accelerate wandb

# Imports typiques
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
import wandb  # Monitoring

# Exemple complet minimal
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = get_peft_model(model, config)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Utilise model gelé automatiquement
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

**Conseils finaux :**

1. **Commence simple** : SFT → DPO → LoRA (pas tout en même temps)
2. **Mesure tout** : sans métriques, impossible de savoir si ça s'améliore
3. **Valide humainement** : les métriques auto ne suffisent jamais
4. **Itère vite** : mieux vaut 10 petites mises à jour qu'1 grosse qui échoue
5. **Protège le base model** : JAMAIS de fine-tuning direct dessus en prod
