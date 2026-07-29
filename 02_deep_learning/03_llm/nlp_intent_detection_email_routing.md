# Extraction d'Intention et Routage vers Logigrammes — Du Bag of Words aux LLMs

> **Résumé en une phrase** : Ce cours explore la chaîne algorithmique complète permettant d'identifier automatiquement l'intention d'un expéditeur dans un email et de la mapper vers un logigramme de traitement, depuis les approches classiques (BoW, TF-IDF) jusqu'aux transformers et LLMs modernes.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-07-29 |
| **Dernière mise à jour** | 2026-07-29 |
| **Domaine** | NLP / Deep Learning / LLM |
| **Niveau** | Intermédiaire → Avancé |
| **Durée de lecture** | ~45 minutes |
| **Fichier** | `nlp_intent_detection_email_routing_2026.md` |
| **Emplacement** | `/02_deep_learning/03_llm/` |
| **Tags** | `#nlp` `#intent-detection` `#embeddings` `#semantic-similarity` `#routing` `#email` `#llm` `#transformers` |

### Prérequis

- [x] Embeddings et similarité cosinus — [[rag.md]] *(déjà couvert)*
- [x] Pipeline de classification d'emails — [[infra_ia.md]] *(déjà couvert)*
- [ ] Transformers et mécanisme d'attention *(recommandé mais non bloquant)*
- [ ] Algèbre linéaire : produit scalaire, norme de vecteur

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : [[rag.md]] — Embeddings, similarité cosinus, recherche vectorielle
- **Prérequis** : [[infra_ia.md]] — Architecture pipeline email, orchestration IA
- **Complémentaires** : [[vit.md]] — Architecture Transformer (même mécanisme d'attention)
- **Suite recommandée** : *Fine-tuning de modèles d'embedding pour domaines spécialisés (à créer)*

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Ce cours couvre le problème de l'**extraction d'intention** (*intent detection*) dans des textes libres (emails), et son couplage avec un système de **routage vers logigrammes**. Nous progresserons chronologiquement des approches fondatrices (Bag of Words, TF-IDF) vers les méthodes de l'état de l'art (sentence embeddings, zero-shot NLI, LLMs), chaque génération d'algorithmes étant présentée comme une réponse aux limitations de la précédente. La partie finale synthétise ces approches dans une **architecture hybride** de production.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Expliquer pourquoi chaque famille d'algorithme a été inventée et quelles limitations elle corrige
2. **Appliquer** : Choisir l'algorithme adapté selon le volume de données, la contrainte temps-réel et la richesse sémantique requise
3. **Analyser** : Déconstruire mathématiquement la chaîne de transformation texte → vecteur → décision
4. **Évaluer** : Comparer les trade-offs entre précision, explicabilité, coût computationnel et besoin en données labelisées

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

La majorité des interactions entre humains et organisations passe par des **messages textuels non structurés** (emails, tickets, chats). Pour automatiser le traitement de ces messages, il faut résoudre un problème fondamental : **transformer une intention humaine exprimée librement en une action machine déterministe**.

Ce problème est au cœur de nombreux systèmes modernes : assistants vocaux, chatbots, systèmes de ticketing, workflows d'entreprise. Comprendre les algorithmes qui le résolvent, c'est comprendre une brique fondamentale de l'IA appliquée aux entreprises.

### Quel problème résout-il ?

**Problème** : Un email arrive avec le corps de texte :

> *"Bonjour, suite à notre échange de la semaine dernière, je souhaitais confirmer que nous n'aurons finalement pas besoin de renouveler notre abonnement pour l'année prochaine."*

Un humain comprend immédiatement : **l'intention est une résiliation**. Un système informatique classique, lui, verrait une suite de tokens sans sémantique. Le défi algorithmique est de **capturer cette intention** et de déclencher le logigramme de traitement correspondant (`process_resiliation.bpmn`).

La difficulté tient à la **variabilité linguistique** : la même intention peut s'exprimer de centaines de façons différentes ("résilier", "ne pas renouveler", "mettre fin à notre contrat", "annuler notre abonnement"...). Aucune approche par mots-clés ne peut toutes les couvrir.

### Applications dans le monde réel

1. **Service client automatisé** : Routage des emails entrants vers les bons services (facturation, support technique, résiliation) sans intervention humaine
2. **Assistants conversationnels (NLU)** : Dialogflow, Rasa, Amazon Lex — tous basés sur la détection d'intention
3. **Automatisation de workflows (RPA)** : Déclencher des processus BPMN, n8n, ou Airflow à partir de demandes textuelles
4. **Systèmes de ticketing** : Assignation automatique des tickets JIRA/ServiceNow à la bonne équipe

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous construisons les connaissances de façon cumulative. Chaque famille d'algorithme sera présentée comme une *réponse aux failles de la précédente*. Cette progression est délibérée : les approches anciennes restent pertinentes pour expliquer et intuiter les approches modernes.

---

### 1. Le problème formalisé

Posons les bases mathématiques du problème une fois pour toutes.

**Définition formelle :**

Soit $$\mathcal{D}$$ l'espace de tous les documents textuels (emails). Soit $$\mathcal{I} = \{i_1, i_2, ..., i_k\}$$ un ensemble fini d'intentions connues. Chaque intention $$i_j$$ est associée à un logigramme $$L_j$$.

On cherche une fonction :

$$f : \mathcal{D} \rightarrow \mathcal{I} \cup \{\text{fallback}\}$$

telle que pour un email $$d \in \mathcal{D}$$, $$f(d)$$ retourne l'intention la plus probable, ou `fallback` si aucune intention n'est détectable avec une confiance suffisante.

**Le défi** est que $$\mathcal{D}$$ est un espace de très haute dimension (le vocabulaire de la langue française compte ~100 000 mots), avec une structure sémantique complexe que des approches naïves ne peuvent pas capturer.

---

### 2. Famille 1 — Représentation par sac de mots (Bag of Words)

#### 2.1 Le modèle Bag of Words (BoW)

Le **Bag of Words** est le modèle de représentation de texte le plus simple. Il traite un document comme un *ensemble non ordonné de mots*, ignorant complètement la grammaire et l'ordre des termes.

**Formalisation :**

Soit $$V = \{w_1, w_2, ..., w_{|V|}\}$$ le vocabulaire (l'ensemble de tous les mots uniques du corpus). Un document $$d$$ est représenté par un vecteur $$\vec{d} \in \mathbb{R}^{|V|}$$ où :

$$\vec{d}_i = \text{nombre d'occurrences de } w_i \text{ dans } d$$

**Exemple concret :**

```
Email A : "Je veux résilier mon contrat"
Email B : "Je souhaite annuler mon abonnement"

Vocabulaire : [je, veux, résilier, mon, contrat, souhaite, annuler, abonnement]

Vecteur A : [1, 1, 1, 1, 1, 0, 0, 0]
Vecteur B : [1, 0, 0, 1, 0, 1, 1, 1]
```

**Visualisation :**

```
             je  veux résil. mon contrat souh. annuler abonn.
Email A  →  [ 1,   1,    1,   1,     1,    0,      0,     0 ]
Email B  →  [ 1,   0,    0,   1,     0,    1,      1,     1 ]
                                 ↑
               Aucune dimension commune malgré la même intention !
               Le modèle ne peut pas voir que "résilier" ≈ "annuler"
```

**Pourquoi BoW est important à comprendre ?**

BoW est la brique conceptuelle sur laquelle reposent TF-IDF, les classifieurs Naïfs Bayésiens, et même les embeddings modernes (qui peuvent être vus comme des BoW "denses et sémantiques"). Comprendre ses limites motive toutes les évolutions suivantes.

**Limites fondamentales :**

| Problème | Description | Exemple |
|---|---|---|
| **Synonymie** | Deux mots différents pour le même concept sont traités comme indépendants | "résilier" et "annuler" → dimensions différentes |
| **Polysémie** | Un mot avec plusieurs sens n'est pas désambiguïsé | "banque" (financière vs de données) → même dimension |
| **Ordre ignoré** | "Je n'ai pas de problème" et "J'ai un problème" → vecteurs proches | Négation perdue |
| **Haute dimensionnalité** | Vecteurs de 100 000 dimensions, très creux (sparse) | 99% de zéros |

---

#### 2.2 TF-IDF — Pondération intelligente des mots

TF-IDF (**Term Frequency — Inverse Document Frequency**) est une amélioration du BoW qui pondère les mots selon leur **capacité discriminante** : un mot fréquent dans un document mais rare dans les autres est plus informatif qu'un mot présent partout.

**Formulation mathématique :**

Le score TF-IDF d'un terme $$t$$ dans un document $$d$$ au sein d'un corpus de $$N$$ documents est :

$\text{TF-IDF}(t, d, D) = \underbrace{\text{TF}(t, d)}_{\text{freq. locale}} \times \underbrace{\text{IDF}(t, D)}_{\text{rarete globale}}$

**Terme Fréquence (TF) :**

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{k} f_{k,d}}$$

Où $$f_{t,d}$$ est le nombre d'occurrences de $$t$$ dans $$d$$, normalisé par la longueur totale du document. Cela évite de favoriser artificiellement les longs documents.

**Inverse Document Frequency (IDF) :**

$$\text{IDF}(t, D) = \log\frac{N}{|\{d \in D : t \in d\}|}$$

Où $$|\{d \in D : t \in d\}|$$ est le nombre de documents contenant $$t$$. Le logarithme atténue les différences extrêmes. Si un mot apparaît dans tous les documents (ex: "le", "de"), son IDF tend vers 0, le rendant sans pouvoir discriminant.

**Intuition visuelle :**

```
Corpus de 1000 emails de service client :

Mot "le"       → apparaît dans 990/1000 docs → IDF ≈ log(1000/990) ≈ 0.01  → non discriminant
Mot "résiliation" → apparaît dans 80/1000 docs → IDF ≈ log(1000/80) ≈ 2.5  → très discriminant
Mot "IBAN"     → apparaît dans 5/1000 docs  → IDF ≈ log(1000/5)  ≈ 5.3  → extrêmement discriminant
```

**Après TF-IDF + classifieur SVM :**

Un classifieur SVM (Support Vector Machine) ou Logistic Regression entraîné sur les vecteurs TF-IDF peut atteindre 85-90% de précision sur des problèmes de classification d'intention si les données labelisées sont suffisantes (~200+ exemples par intention).

**Limites persistantes de TF-IDF :**

La pondération améliore BoW mais ne résout pas le problème de **synonymie sémantique**. "résilier" et "annuler" restent des dimensions orthogonales dans l'espace vectoriel TF-IDF :

$\text{sim}(\mathbf{v}_{\text{resilier}}^{\text{TF-IDF}},\ \mathbf{v}_{\text{annuler}}^{\text{TF-IDF}}) = 0$

C'est le problème fondamental qui motive la génération suivante d'algorithmes.

---

### 3. Famille 2 — Représentations distribuées (Word Embeddings)

#### 3.1 Le tournant conceptuel : de la représentation sparse à dense

L'idée révolutionnaire de la fin des années 2000 est de représenter les mots non plus par leur identité lexicale (position dans un vocabulaire) mais par leur **contexte d'usage**. L'hypothèse distributionnelle de Harris (1954) stipule :

> *"A word is characterized by the company it keeps."* — J.R. Firth (1957)

Des mots qui apparaissent dans des contextes similaires ont des significations similaires. Donc, si on entraîne un modèle à **prédire le contexte d'un mot** (ou inversement, prédire un mot à partir de son contexte), ce modèle apprend implicitement une représentation sémantique.

**Transition représentationnelle :**

```
BoW / TF-IDF :
  "résilier"  → [0, 0, 0, ..., 1, ..., 0, 0]   (vecteur de dimension |V| ≈ 100 000, sparse)

Word Embedding :
  "résilier"  → [0.32, -0.71, 0.14, ..., 0.88] (vecteur de dimension d = 100-300, dense)
  "annuler"   → [0.29, -0.68, 0.11, ..., 0.91] (proche dans l'espace !)
  "météo"     → [-0.82, 0.43, -0.56, ..., 0.02] (loin dans l'espace)
```

#### 3.2 Word2Vec — Apprendre la sémantique par prédiction

**Word2Vec** (Mikolov et al., 2013) est le modèle fondateur des word embeddings modernes. Il entraîne un réseau de neurones superficiel sur une tâche de prédiction de mots, et les poids appris constituent les embeddings.

Deux architectures :

**CBOW (Continuous Bag of Words) :** Prédire un mot à partir de ses voisins.

$$P(w_t | w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}) = \text{softmax}(\mathbf{W}_{\text{out}} \cdot \bar{\mathbf{h}})$$

Où $$\bar{\mathbf{h}} = \frac{1}{2k}\sum_{-k \leq j \leq k, j \neq 0} \mathbf{W}_{\text{in}} \cdot \mathbf{1}_{w_{t+j}}$$ est la moyenne des vecteurs de contexte.

**Skip-gram :** Prédire les voisins à partir d'un mot (inverse de CBOW, plus performant sur les mots rares).

$$P(w_{t+j} | w_t) = \frac{\exp(\mathbf{v}'_{w_{t+j}} \cdot \mathbf{v}_{w_t})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_t})}$$

**Propriété algébrique remarquable :**

Les embeddings Word2Vec capturent des **relations sémantiques analogiques** :

$\mathbf{v}_{\text{roi}} - \mathbf{v}_{\text{homme}} + \mathbf{v}_{\text{femme}} \approx \mathbf{v}_{\text{reine}}$

C'est la preuve que l'espace vectoriel appris reflète une structure sémantique réelle.

**Limite de Word2Vec pour notre cas d'usage :**

Word2Vec produit un embedding **par mot**, mais pour détecter l'intention d'un email, on a besoin d'un embedding **de phrase entière**. La solution naïve (moyenne des embeddings de mots) est médiocre :

$\vec{d}_{\text{moy}} = \frac{1}{n}\sum_{i=1}^{n} \vec{w}_{i}$

Cette agrégation perd l'information d'ordre et de structure. "Je n'ai aucun problème" et "J'ai un problème" donneraient des vecteurs très proches.

---

#### 3.3 FastText — Extension aux sous-mots

**FastText** (Bojanowski et al., 2017, Facebook AI Research) étend Word2Vec en représentant les mots comme la somme de leurs **n-grammes de caractères** (sous-chaînes de longueur n).

Pour le mot "résiliation" avec n=3 :

$\mathbf{v}_{\text{resiliation}} = \sum_{g \,\in\, G(\text{resiliation})} \mathbf{z}_{g}$

**Avantage clé pour notre domaine :**

- Gère les **mots hors-vocabulaire** (OOV) : "résiliaton" (faute) → calculé à partir de ses trigrammes communs avec "résiliation"
- Robuste aux **variations morphologiques** : "résilier", "résiliez", "résiliation" partagent des n-grammes → embeddings proches

**Limite commune à Word2Vec et FastText :**

Ces modèles produisent des embeddings **statiques** : un mot a toujours le même vecteur quel que soit le contexte. La phrase :
- "Mon compte est à la *banque*" et
- "La *banque* de données est corrompue"

donnent le même embedding pour "banque". C'est le problème de la **polysémie contextuelle**, résolu par les Transformers.

---

### 4. Famille 3 — Embeddings de phrases et similarité sémantique (State of the Art)

#### 4.1 BERT et les représentations contextuelles

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) marque un tournant : les embeddings deviennent **contextuels**. Le même mot reçoit des représentations différentes selon son contexte, grâce au mécanisme d'**attention multi-têtes**.

Pour notre objectif, l'architecture importante est le mécanisme d'attention :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Où :
- $$Q \in \mathbb{R}^{n \times d_k}$$ : matrice des **queries** (chaque token "pose une question")
- $$K \in \mathbb{R}^{m \times d_k}$$ : matrice des **keys** (chaque token "répond aux questions")
- $$V \in \mathbb{R}^{m \times d_v}$$ : matrice des **values** (les informations à agréger)
- $$d_k$$ : dimension des keys (facteur de normalisation pour éviter les gradients trop petits)

**Intuition du mécanisme d'attention pour la polysémie :**

```
Phrase : "Je souhaite résilier mon contrat téléphonique"

Token "contrat" → sa représentation dans BERT est calculée en pondérant
tous les autres tokens :

  Attention("contrat", "résilier")  = 0.82  ← très pertinent
  Attention("contrat", "téléphonique") = 0.71  ← pertinent
  Attention("contrat", "Je")        = 0.03  ← peu pertinent
  Attention("contrat", "souhaite")  = 0.15  ← peu pertinent

→ L'embedding de "contrat" sera influencé par "résilier" et "téléphonique"
→ Disambiguïsation sémantique implicite
```

#### 4.2 Le problème de l'embedding de phrase avec BERT

BERT a été entraîné sur des tâches de classification au niveau de la phrase (classification du token [CLS]), mais **pas pour la similarité sémantique de phrases**. L'embedding du token [CLS] est une mauvaise représentation de la phrase pour calculer des similarités :

Les expériences de Reimers & Gurevych (2019) montrent que des embeddings [CLS] de BERT ont une performance inférieure à TF-IDF pour la recherche sémantique de similarité.

#### 4.3 Sentence-BERT (SBERT) — La solution pour notre cas d'usage

**SBERT** (Reimers & Gurevych, 2019) fine-tune BERT sur un réseau siamois avec une **perte de triplet** ou une perte de cosinus, pour que les embeddings de phrases sémantiquement similaires soient proches dans l'espace vectoriel.

**Architecture du réseau siamois :**

```
     Phrase A                    Phrase B
        ↓                           ↓
  [BERT Encoder]            [BERT Encoder]
  (poids partagés)          (poids partagés)
        ↓                           ↓
  [Pooling: mean]           [Pooling: mean]
        ↓                           ↓
     u ∈ ℝ^768               v ∈ ℝ^768
        ↓                           ↓
  ┌─────────────────────────────────────┐
  │  Fonction de similarité cosinus     │
  │  sim(u, v) = (u·v) / (||u||·||v||) │
  └─────────────────────────────────────┘
              ↓
        Score de similarité ∈ [-1, 1]
```

**Fonction de perte d'entraînement (multiple negative ranking loss) :**

$\mathcal{L} = -\log\frac{\exp(\text{sim}(u, v^+) / \tau)}{\exp(\text{sim}(u, v^+) / \tau) + \sum_{j=1}^{N}\exp(\text{sim}(u, v^{-}_{j}) / \tau)}$

Où :
- $$v^+$$ : phrase positive (sémantiquement similaire à $$u$$)
- $v^{-}_{j}$ : phrases négatives (sémantiquement différentes)
- $$\tau$$ : température (hyperparamètre contrôlant la netteté des distributions)

**Application directe à l'extraction d'intention :**

```
LISTE D'INTENTIONS (encodée une seule fois, mise en cache) :
  i₁ = "Le client veut annuler son contrat ou son abonnement"
  i₂ = "Le client a un problème avec sa facture ou un paiement"
  i₃ = "Le client rencontre un problème technique"

EMAIL ENTRANT :
  d = "Je ne souhaite pas renouveler notre engagement pour l'année prochaine"

CALCUL :
  sim(embed(d), embed(i₁)) = 0.81  ← PLUS HAUTE SIMILARITÉ
  sim(embed(d), embed(i₂)) = 0.23
  sim(embed(d), embed(i₃)) = 0.18

DÉCISION :
  Intention détectée : i₁ (résiliation)
  Logigramme déclenché : process_resiliation.bpmn
```

**Pourquoi cette approche est fondamentale pour notre cas d'usage :**

Elle permet du **zero-shot intent detection** : on n'a besoin d'aucun exemple d'entraînement. Il suffit de décrire chaque intention en language naturel. Ajouter une nouvelle intention ne requiert pas de réentraîner un modèle — juste d'ajouter une entrée dans la liste.

#### 4.4 La similarité cosinus — Approfondissement mathématique

La **similarité cosinus** est la métrique centrale de ce système. Elle mesure l'angle entre deux vecteurs, indépendamment de leur magnitude.

$\text{sim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}$

**Propriétés :**
- Valeurs dans $$[-1, 1]$$
- $$\text{sim} = 1$$ : vecteurs colinéaires (même direction sémantique)
- $$\text{sim} = 0$$ : vecteurs orthogonaux (aucune relation sémantique)
- $$\text{sim} = -1$$ : vecteurs antiparallèles (sens opposés, rare en NLP)

**Pourquoi le cosinus et pas la distance euclidienne ?**

La distance euclidienne $d_{\text{euc}}(\vec{u}, \vec{v}) = \|\vec{u} - \vec{v}\| = \sqrt{\sum_i (u_i - v_i)^2}$ est sensible à la **magnitude** des vecteurs. Or, deux emails identiques en contenu mais l'un plus long que l'autre produiraient des embeddings de magnitude différente. La similarité cosinus normalise cette magnitude :

$\text{sim}(\vec{u}, \vec{v}) = \frac{\vec{u}}{\|\vec{u}\|} \cdot \frac{\vec{v}}{\|\vec{v}\|}$

On compare les **directions** dans l'espace sémantique, pas les tailles.

**Dérivation de la relation distance euclidienne → cosinus :**

$\begin{aligned}
\|\vec{u} - \vec{v}\|^2 &= (\vec{u} - \vec{v}) \cdot (\vec{u} - \vec{v}) \\
&= \|\vec{u}\|^2 - 2(\vec{u} \cdot \vec{v}) + \|\vec{v}\|^2
\end{aligned}$

Si $$\vec{u}$$ et $$\vec{v}$$ sont normalisés ($\|\vec{u}\| = \|\vec{v}\| = 1$) :

$\|\vec{u} - \vec{v}\|^2 = 2 - 2(\vec{u} \cdot \vec{v}) = 2(1 - \text{sim}(\vec{u}, \vec{v}))$

**Conclusion :** Pour des vecteurs normalisés, distance euclidienne et similarité cosinus sont **équivalentes** (ordre de classement identique). Les modèles SBERT produisent des embeddings normalisés, donc les deux métriques peuvent être utilisées indifféremment.

#### 4.5 Le seuillage de confiance (Threshold)

Un système de routage robuste ne doit pas forcer une décision quand aucune intention n'est claire. On introduit un **seuil de confiance** $$\tau$$ :

$$f(d) = \begin{cases} \arg\max_j \text{ sim}(\vec{d}, \vec{i}_j) & \text{si } \max_j \text{ sim}(\vec{d}, \vec{i}_j) \geq \tau \\ \text{fallback} & \text{sinon} \end{cases}$$

**Calibration du seuil :**

Le choix de $$\tau$$ relève d'un **trade-off précision/rappel** :
- $$\tau$$ élevé (ex: 0.75) → peu de faux positifs, mais beaucoup de `fallback` → charge humaine importante
- $$\tau$$ bas (ex: 0.35) → peu de `fallback`, mais risque de mauvais routages

En pratique, on calibre $$\tau$$ sur un jeu de validation en maximisant le F1-score ou en respectant une contrainte métier (ex: "le taux de faux positifs ne doit pas dépasser 5%").

---

### 5. Famille 4 — Classification zero-shot par inférence de langage naturel (NLI)

#### 5.1 L'approche NLI pour l'extraction d'intention

Une approche alternative aux embeddings de phrases utilise les modèles entraînés sur la **Natural Language Inference (NLI)** : étant donné une prémisse et une hypothèse, le modèle prédit si l'hypothèse est `entailment` (impliquée), `contradiction`, ou `neutral`.

Cette capacité peut être **réutilisée pour la classification zero-shot** :

```
Prémisse  = l'email : "Je ne souhaite pas renouveler notre contrat"
Hypothèse = l'intention : "Cette demande concerne une résiliation"
→ Le modèle NLI prédit : ENTAILMENT (confiance = 0.92)

Hypothèse = l'intention : "Cette demande concerne une facture"
→ Le modèle NLI prédit : CONTRADICTION (confiance = 0.87)
```

**Avantage :** Les modèles NLI (BART-MNLI, DeBERTa-MNLI) sont plus robustes que la similarité cosinus pure pour des intentions formulées de manière complexe ou ambiguë.

**Inconvénient :** La latence est proportionnelle au nombre d'intentions (un appel modèle par intention vs. un seul pour les embeddings + comparaison vectorielle).

---

### 6. Famille 5 — LLMs et Prompt Engineering

#### 6.1 LLMs comme classifieurs universels

Les **Large Language Models** (GPT-4, Mistral, LLaMA 3...) apprennent une représentation du langage si riche qu'ils peuvent classifier des intentions directement via le prompt, sans aucun entraînement supplémentaire.

**Mécanisme sous-jacent :**

Un LLM approxime la distribution conditionnelle $$P(y | x, \text{prompt})$$ où $$x$$ est l'email, $$\text{prompt}$$ encode la tâche, et $$y$$ est la sortie souhaitée (le nom de l'intention).

La qualité du résultat dépend entièrement de la **qualité du prompt** : c'est le **Prompt Engineering**.

#### 6.2 Anatomie d'un prompt de classification d'intention

Un bon prompt pour notre cas d'usage suit la structure :

```
[ROLE] Tu es un assistant spécialisé dans le routage de demandes clients.

[CONTRAINTES] Réponds UNIQUEMENT avec un objet JSON valide.

[DEFINITIONS] Les intentions possibles sont :
- "RESILIATION" : le client souhaite mettre fin à un contrat ou abonnement
- "FACTURATION" : question ou litige sur une facture ou un paiement
- "SUPPORT_TECH" : problème technique avec un produit ou service
- "RENSEIGNEMENT" : demande d'information sans action requise
- "AUTRE" : aucune des catégories ci-dessus

[INPUT] Email à classifier :
{email_text}

[OUTPUT FORMAT]
{
  "intention": "<nom_intention>",
  "confidence": <0.0-1.0>,
  "justification": "<explication en 1 phrase>"
}
```

**Propriétés mathématiques du LLM pour la classification :**

Le LLM génère la sortie token par token selon :

$$P(y_t | y_{<t}, x) = \text{softmax}(\mathbf{W}_{\text{vocab}} \cdot \mathbf{h}_t)$$

Où $$\mathbf{h}_t$$ est la représentation cachée à l'étape $$t$$, issue de l'empilement de couches d'attention. La **température** $$T$$ contrôle la certitude :

$$P_T(y_t | ...) = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

- $$T \to 0$$ : comportement déterministe (prend le max des logits)
- $$T = 1$$ : distribution originale du modèle
- $$T > 1$$ : distribution plus "plate", réponses plus créatives/aléatoires

**Pour la classification d'intention, on utilise toujours $$T$$ proche de 0** (entre 0.0 et 0.1) pour maximiser la reproductibilité.

#### 6.3 Few-Shot Learning dans le prompt

On peut améliorer significativement la précision en ajoutant des **exemples** directement dans le prompt :

```
Exemples de classification :

Email : "Veuillez procéder à l'annulation de mon abonnement annuel"
→ {"intention": "RESILIATION", "confidence": 0.97, ...}

Email : "Ma dernière facture comporte une erreur de 45€"
→ {"intention": "FACTURATION", "confidence": 0.94, ...}

Email à classifier :
{email_text}
```

**Pourquoi le Few-Shot fonctionne ?**

Par le mécanisme d'**in-context learning** : le LLM utilise les exemples pour inférer le format et la logique de la tâche sans mettre à jour ses poids. C'est une forme de meta-apprentissage implicite, qui reste un sujet de recherche actif.

---

### 7. Le Mapping Intention → Logigramme

L'extraction d'intention n'est que la première moitié du problème. La deuxième est le **routage vers le logigramme** approprié.

#### 7.1 Table de routage statique

La solution la plus simple est un dictionnaire clé-valeur :

$\text{ROUTING-TABLE} : \mathcal{I} \rightarrow \mathcal{L}$

Où $$\mathcal{L}$$ est l'ensemble des logigrammes disponibles. Cela fonctionne bien si les intentions sont stables et les logigrammes bien définis.

**Enrichissement par entités :**

On peut affiner le routage en combinant l'intention détectée avec des **entités extraites** (NER — Named Entity Recognition) :

$\text{Logigramme} = \text{ROUTING-TABLE}[\text{intention}][\text{entites}]$

Exemple :

```
Intention : FACTURATION
  + entité montant > 1000€  → workflow_litige_premium (SLA 24h)
  + entité type = "avoir"   → workflow_avoir_comptable
  + entité type = "double facturation" → workflow_doublon_facture
```

#### 7.2 Routage par similarité sémantique sur les descriptions de logigrammes

Si les logigrammes sont nombreux ou évoluent fréquemment, on peut **embedder les descriptions de logigrammes** et faire de la recherche de similarité directement dessus — sans table de mapping explicite.

$$L^* = \arg\max_{L_j \in \mathcal{L}} \text{sim}\left(\text{embed}(d_{\text{email}}), \text{embed}(\text{description}(L_j))\right)$$

C'est une extension directe du mécanisme RAG que tu connais déjà ([[rag.md]]).

---

## 💡 Compréhension Intuitive

### Analogie du monde réel

Imaginez un **standardiste expert** dans une grande entreprise. Au fil des années, il a développé une intuition remarquable : dès les premières phrases d'un appel, il sait exactement quel service doit prendre en charge la demande.

Son "modèle mental" fonctionne exactement comme nos algorithmes :
- **BoW/TF-IDF** : Il écoute les mots-clés ("facture", "résilier") et route mécaniquement
- **Word Embeddings** : Il comprend que "résilier" et "annuler" signifient la même chose
- **SBERT** : Il comprend le sens global de la phrase, pas juste les mots
- **LLM** : Il raisonne sur le contexte complet, y compris les sous-entendus

La "table de routage" est son **annuaire mental** : intention détectée → service à appeler.

### Questions pour vérifier la compréhension

1. **Q1** : Pourquoi TF-IDF est-il meilleur que BoW pour distinguer les emails importants des emails triviaux ?
   - *Réponse attendue* : IDF pénalise les mots ubiquitaires ("le", "de") et valorise les termes rares spécifiques à un email, augmentant le pouvoir discriminant du vecteur.

2. **Q2** : Pourquoi la similarité cosinus est-elle préférée à la distance euclidienne pour comparer des embeddings ?
   - *Réponse attendue* : Elle est invariante à la magnitude des vecteurs, ne favorisant pas les documents plus longs qui ont naturellement des embeddings de plus grande norme.

3. **Q3** : Quel est l'avantage principal de SBERT sur une simple moyenne des embeddings Word2Vec ?
   - *Réponse attendue* : SBERT a été fine-tuné spécifiquement pour que la similarité cosinus entre phrases soit sémantiquement significative, ce que la moyenne Word2Vec ne garantit pas.

4. **Q4** : Dans quel cas utilise-t-on le LLM en fallback plutôt qu'en premier recours ?
   - *Réponse attendue* : Pour des raisons de latence et de coût — les embeddings + cosinus sont ~100x plus rapides et moins coûteux pour les cas simples.

---

## ⚖️ Comparaisons et Choix de Design

### Synthèse comparative des approches

| Critère | TF-IDF + SVM | Word2Vec (moyenne) | SBERT + Cosinus | Zero-Shot NLI | LLM (GPT/Mistral) |
|---|---|---|---|---|---|
| **Données requis** | 200+ exemples/intention | Corpus non labelisé | Aucune | Aucune | Aucune |
| **Précision** | Bonne (85-90%) | Moyenne (70-80%) | Très bonne (88-93%) | Bonne (82-88%) | Excellente (92-96%) |
| **Latence** | ~1ms | ~2ms | ~10-50ms | ~200-500ms | ~500ms-2s |
| **Coût computation** | Très faible | Faible | Modéré | Élevé | Très élevé |
| **Gestion synonymes** | ❌ | ✅ | ✅✅ | ✅✅ | ✅✅✅ |
| **Ajout intention** | Réentraîner le modèle | Réentraîner | Juste ajouter description | Juste ajouter description | Modifier prompt |
| **Explicabilité** | ✅✅ (features = mots) | ⚠️ | ⚠️ | ✅ (NLI score) | ✅ (justification) |
| **Hors-vocabulaire** | ❌ | ⚠️ (FastText ✅) | ✅ | ✅ | ✅ |

### Recommandations de choix

- ✅ **Utiliser TF-IDF + SVM** quand : données labelisées disponibles (>500 emails), besoin d'explicabilité forte, contrainte de latence <5ms
- ✅ **Utiliser SBERT + Cosinus** quand : peu ou pas de données labelisées, nouvelle application, intentions décrites en langage naturel (zero-shot)
- ✅ **Utiliser LLM comme fallback** quand : la similarité cosinus est < seuil $$\tau$$, email ambigu ou multi-intentionnel, besoin d'explication de la décision
- ✅ **Utiliser NLI** quand : les intentions sont formulées sous forme d'hypothèses complexes, que la robustesse prime sur la latence

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreurs fréquentes

#### Erreur 1 : Utiliser l'embedding [CLS] de BERT brut pour la similarité

**Description** : BERT non fine-tuné produit des embeddings [CLS] qui ne sont pas calibrés pour la similarité sémantique. Deux phrases sémantiquement proches peuvent avoir un cosinus < 0.5.

**Solution** : Utiliser exclusivement des modèles spécifiquement entraînés pour la similarité de phrases (SBERT, E5, BGE).

**Sources** : Reimers & Gurevych (2019) — *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*

#### Erreur 2 : Ne pas seuiller la confiance

**Description** : Si le système route tout email vers la meilleure intention même avec un score 0.30, des emails hors-sujet (spam, conversation interne) seront mal routés.

**Solution** : Toujours implémenter un fallback avec seuil calibré sur des données réelles.

#### Erreur 3 : Descriptions d'intentions trop courtes

**Description** : "résiliation" comme description d'intention est pauvre sémantiquement. L'embedding d'un seul mot est moins riche qu'une phrase descriptive.

**Solution** : Écrire des descriptions enrichies : *"Le client souhaite mettre fin à son contrat, annuler son abonnement ou ne pas le renouveler"*.

#### Erreur 4 : Confondre intent detection et NER

**Description** : L'intention répond à "Que veut le client ?" et les entités à "Sur quoi / avec quels paramètres ?". Ces deux problèmes sont distincts mais complémentaires.

**Solution** : Pipeline séquentiel ou parallèle : intent detection d'abord, puis extraction d'entités conditionnée à l'intention détectée.

### ✅ Bonnes pratiques

#### Pratique 1 : Caching des embeddings d'intentions

Les embeddings des descriptions d'intentions ne changent pas entre les appels. Les pré-calculer une fois et les stocker en mémoire réduit la latence de moitié.

#### Pratique 2 : Monitoring du score de confiance

Logger systématiquement le score de similarité maximal pour chaque décision. Une dérive vers le bas indique que de nouvelles intentions émergent dans les emails et ne sont pas couvertes par la liste existante.

#### Pratique 3 : Stratégie multi-intentionnelle

Certains emails contiennent plusieurs intentions ("Je veux résilier ET j'ai une question sur ma dernière facture"). Prévoir un **top-k** au lieu d'un top-1 : extraire les k meilleures intentions si leurs scores sont au-dessus de $$\tau$$.

### 📋 Checklist de validation du système

- [ ] Les descriptions d'intentions sont-elles suffisamment riches et variées ?
- [ ] Le seuil de confiance $$\tau$$ est-il calibré sur des données réelles ?
- [ ] Un fallback humain est-il prévu pour les emails non routables ?
- [ ] Les embeddings d'intentions sont-ils précalculés et mis en cache ?
- [ ] Le système gère-t-il les emails multi-intentionnels (top-k) ?
- [ ] Les décisions sont-elles loguées avec les scores de confiance ?
- [ ] Un mécanisme de détection de "nouvelles intentions" émergentes est-il prévu ?

---

## 🧪 Exercices et Validation des Connaissances

### Exercice 1 : Débutant — Comprendre la progression algorithmique

**Énoncé** :

Pour l'email suivant, décrivez comment chacune des 4 familles d'algorithmes le traiterait, et identifiez à quel stade la bonne intention serait détectée :

*Email : "Suite à nos difficultés financières récentes, nous serions contraints de ne pas poursuivre notre collaboration."*

<details>
<summary>✅ Réponse attendue</summary>

- **BoW/TF-IDF** : Les mots-clés "résiliation" et "annuler" sont absents. Un classifieur TF-IDF entraîné sur des libellés directs échouerait probablement. Intention non détectée ou mal classifiée.

- **Word2Vec (moyenne)** : "collaboration", "poursuivre" (négation ignorée !), "financières" → l'embedding moyen serait bruité. La négation "ne pas poursuivre" est sémantiquement opposée à "poursuivre" mais le BoW de Word2Vec ne capture pas la négation. Résultat incertain.

- **SBERT** : Le modèle encode la phrase entière en tenant compte du contexte. "ne pas poursuivre notre collaboration" dans le contexte de "difficultés financières" → embedding proche des intentions de résiliation/fin de contrat. Bonne détection.

- **LLM** : Comprend le sous-entendu social, la formulation indirecte et polie, et identifie l'intention de résiliation avec haute confiance. Explication : "L'expéditeur indique indirectement vouloir mettre fin à un partenariat commercial pour des raisons financières."

**Leçon** : Les formulations polies, indirectes ou euphémistiques nécessitent les approches modernes (SBERT ou LLM).
</details>

### Exercice 2 : Intermédiaire — Raisonner sur la similarité cosinus

**Énoncé** :

Soit trois vecteurs d'embedding (simplifiés à 3 dimensions pour l'exercice) :

$$\vec{e}_{\text{email}} = [0.8, 0.5, 0.2]$$
$$\vec{i}_{\text{résiliation}} = [0.9, 0.4, 0.1]$$
$$\vec{i}_{\text{facturation}} = [0.2, 0.9, 0.7]$$

Calculez les deux similarités cosinus et déterminez l'intention routée. Concluez sur le choix de seuil $$\tau$$.

<details>
<summary>✅ Solution</summary>

**Norme de chaque vecteur :**

$$||\vec{e}|| = \sqrt{0.8^2 + 0.5^2 + 0.2^2} = \sqrt{0.64 + 0.25 + 0.04} = \sqrt{0.93} \approx 0.964$$

$$||\vec{i}_R|| = \sqrt{0.9^2 + 0.4^2 + 0.1^2} = \sqrt{0.81 + 0.16 + 0.01} = \sqrt{0.98} \approx 0.990$$

$$||\vec{i}_F|| = \sqrt{0.2^2 + 0.9^2 + 0.7^2} = \sqrt{0.04 + 0.81 + 0.49} = \sqrt{1.34} \approx 1.158$$

**Produits scalaires :**

$$\vec{e} \cdot \vec{i}_R = 0.8 \times 0.9 + 0.5 \times 0.4 + 0.2 \times 0.1 = 0.72 + 0.20 + 0.02 = 0.94$$

$$\vec{e} \cdot \vec{i}_F = 0.8 \times 0.2 + 0.5 \times 0.9 + 0.2 \times 0.7 = 0.16 + 0.45 + 0.14 = 0.75$$

**Similarités cosinus :**

$$\text{sim}(\vec{e}, \vec{i}_R) = \frac{0.94}{0.964 \times 0.990} \approx \frac{0.94}{0.954} \approx 0.985$$

$$\text{sim}(\vec{e}, \vec{i}_F) = \frac{0.75}{0.964 \times 1.158} \approx \frac{0.75}{1.116} \approx 0.672$$

**Décision :** Intention = RÉSILIATION (score 0.985 >> seuil $$\tau$$ typique de 0.5).

**Conclusion sur $$\tau$$ :** Un seuil $$\tau = 0.5$$ serait ici suffisant. Avec des embeddings réels en haute dimension, les scores sont généralement moins polarisés, justifiant des seuils autour de 0.4-0.6.
</details>

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Académiques Fondamentaux

#### 1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

- **Auteurs** : Nils Reimers, Iryna Gurevych (2019)
- **Publication** : EMNLP 2019
- **URL** : [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- **Contribution clé** : Démontre l'inadéquation des embeddings BERT bruts pour la similarité et propose un réseau siamois avec perte de triplet qui multiplie par 10 les performances sur les benchmarks de similarité sémantique
- **Pertinence** : Paper fondateur de l'approche SBERT, directement applicable à notre système
- **Niveau** : Technique, accessible avec les bases Transformer

#### 2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- **Auteurs** : Devlin, Chang, Lee, Toutanova (2019)
- **Publication** : NAACL 2019
- **URL** : [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- **Contribution clé** : Entraînement bidirectionnel par masked language modeling et next sentence prediction, permettant des représentations contextuelles profondes
- **Pertinence** : Comprendre BERT est fondamental pour comprendre SBERT
- **Niveau** : Mathématique (maîtrise des Transformers requise)

#### 3. Efficient Natural Language Response Suggestion for Smart Reply

- **Auteurs** : Henderson et al. (2017)
- **Publication** : Google Research
- **URL** : [https://arxiv.org/abs/1705.00652](https://arxiv.org/abs/1705.00652)
- **Contribution clé** : Application concrète de l'intent detection dans un système de réponse automatique à des emails, architecture déployée en production
- **Pertinence** : Cas d'usage directement comparable au nôtre, vision engineering
- **Niveau** : Accessible, orienté applications

#### 4. Benchmarking Zero-shot Text Classification

- **Auteurs** : Yin, Hay, Roth (2019)
- **Publication** : EMNLP 2019
- **URL** : [https://arxiv.org/abs/1909.00161](https://arxiv.org/abs/1909.00161)
- **Contribution clé** : Formalise et benchmarke l'approche NLI pour la classification zero-shot, établit les métriques de référence
- **Pertinence** : Comprendre les limites et performances des approches zero-shot
- **Niveau** : Accessible

### 📚 Ressources Complémentaires

#### Documentation officielle

- **Sentence-Transformers** — [https://www.sbert.net/](https://www.sbert.net/)
  - 📌 **Section recommandée** : "Pre-trained Models" et "Usage > Semantic Textual Similarity"

- **HuggingFace — Zero-Shot Classification Pipeline** — [https://huggingface.co/tasks/zero-shot-classification](https://huggingface.co/tasks/zero-shot-classification)
  - 📌 **Section recommandée** : Documentation du pipeline `zero-shot-classification` avec modèles NLI

### 🛠️ Modèles recommandés (HuggingFace)

| Modèle | Langues | Dimension | Cas d'usage recommandé |
|---|---|---|---|
| `paraphrase-multilingual-mpnet-base-v2` | 50+ (dont FR) | 768 | **Standard : emails FR/EN mixtes** |
| `camembert-base` (fine-tuné SBERT) | FR uniquement | 768 | Textes purement français |
| `intfloat/multilingual-e5-large` | 100+ | 1024 | Haute précision, latence acceptable |
| `facebook/bart-large-mnli` | EN (principal) | NLI | Zero-shot NLI anglais |
| `cross-encoder/nli-MiniLM2-L6-H768` | EN | NLI | Zero-shot NLI rapide |

---

## 📝 Résumé Rapide (Quick Reference)

### Pipeline de décision

```
EMAIL ENTRANT
      │
      ▼
[Règles rapides]  ─ confiance haute ──► Intention → Logigramme
      │
  incertain
      │
      ▼
[embed(email) → SBERT]
[embed(intentions) → SBERT] (précalculé, mis en cache)
      │
      ▼
[argmax cosine_similarity] ─ score ≥ τ ──► Intention → Logigramme
      │
  score < τ
      │
      ▼
[LLM fallback + justification] ──► Intention → Logigramme
                                       │
                                   toujours incertain
                                       │
                                       ▼
                                [Escalade humaine]
```

### Formules clés

| Concept | Formule |
|---|---|
| TF-IDF | $\text{TF-IDF}(t,d) = \text{TF}(t,d) \cdot \log\!\left(\frac{N}{n_t}\right)$ |
| Similarité cosinus | $\text{sim}(\vec{u},\vec{v}) = \frac{\vec{u}\cdot\vec{v}}{\lVert\vec{u}\rVert \cdot \lVert\vec{v}\rVert}$ |
| Décision seuillée | $f(d) = \arg\max_{j}\,\text{sim}(\vec{d},\vec{i}_{j}) \text{ si } \max_{j} \geq \tau$ |
| Attention (Transformer) | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$ |

### Décisions clés

```
Besoin zero-shot (pas de données labelisées) ?
  OUI → SBERT + cosinus (ou NLI si intents complexes)
  NON → TF-IDF + SVM si latence critique, sinon SBERT fine-tuné

Contrainte de latence < 10ms ?
  OUI → TF-IDF + SVM ou règles
  NON → SBERT + cosinus

Email ambigu ou multi-intentionnel ?
  OUI → LLM fallback + top-k intentions
  NON → argmax simple sur les scores SBERT

Intentions évoluent fréquemment ?
  OUI → SBERT (juste modifier la liste de descriptions)
  NON → SVM fine-tuné (plus précis sur intentions stables)
```

### Pièges à éviter

1. ⚠️ **Embedding [CLS] BERT brut** → Utiliser SBERT ou E5
2. ⚠️ **Pas de seuil de confiance** → Toujours implémenter un fallback
3. ⚠️ **Descriptions d'intentions trop courtes** → Phrases descriptives enrichies
4. ⚠️ **Confondre intent et entité** → Pipeline séquentiel : intent → NER → routing

---

## 🔗 Intégration Repository GitHub

### Fichiers à mettre à jour

1. **`02_deep_learning/03_llm/rag.md`** — Ajouter dans "Voir aussi" :
   ```
   - [[nlp_intent_detection_email_routing_2026]] — Extension du RAG à la détection d'intention zero-shot
   ```

2. **`03_infrastructure/infra_ia.md`** — Ajouter dans la section 9 (Classification d'emails) :
   ```
   - Voir [[nlp_intent_detection_email_routing_2026]] pour l'approfondissement algorithmique de analyze_intent_ml()
   ```

3. **`README.md`** ou index principal — Ajouter :
   ```
   ### 02_deep_learning/03_llm
   - [[nlp_intent_detection_email_routing_2026]] — Intent detection : BoW → SBERT → LLM + routage logigrammes
   ```

### Proposition de nommage

- **Nom de fichier** : `nlp_intent_detection_email_routing_2026.md`
- **Emplacement** : `/02_deep_learning/03_llm/`
