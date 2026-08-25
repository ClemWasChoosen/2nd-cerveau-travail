# 🧠 Second Cerveau - Base de Connaissances Technique

> **Repository personnel de documentation en Statistiques, Machine Learning, Deep Learning, Infrastructure, Bases de données et Méthodologie**  
> Développeur R&D - Informatique & Intelligence Artificielle

[![Last Update](https://img.shields.io/badge/Last%20Update-August%202026-blue)]()
[![Cours](https://img.shields.io/badge/Cours-34-green)]()
[![Structure](https://img.shields.io/badge/Structure-Thématique-success)]()

---

## 📋 Table des Matières

- [🎯 À Propos](#-à-propos)
- [🗂️ Structure du Repository](#️-structure-du-repository)
- [📚 Catalogue des Cours](#-catalogue-des-cours)
  - [00. Statistics Foundations](#00-statistics-foundations)
  - [01. Machine Learning](#01-machine-learning)
  - [02. Deep Learning](#02-deep-learning)
  - [03. Infrastructure](#03-infrastructure)
  - [04. Databases](#04-databases)
  - [05. Méthodologie](#05-méthodologie)
- [🚀 Parcours Recommandés](#-parcours-recommandés)
- [📖 Comment Utiliser ce Repository](#-comment-utiliser-ce-repository)
- [🔗 Liens Entre les Cours](#-liens-entre-les-cours)
- [✨ Contribuer](#-contribuer)

---

## 🎯 À Propos

Ce repository constitue mon **"Second Cerveau"** personnel : une base de connaissances structurée, évolutive et relisible à long terme sur les statistiques, le Machine Learning, le Deep Learning, l'infrastructure IA, les bases de données et la méthodologie R&D.

### Objectifs

- 📝 **Documenter** les concepts techniques sous forme de cours réutilisables
- 🔗 **Relier** les notions entre elles selon une logique de progression
- 🎓 **Progresser** des fondations mathématiques vers les architectures et applications avancées
- 🛠️ **Pratiquer** avec des exemples, des implémentations et des cas d'usage
- 🔬 **Comprendre** les papiers scientifiques et leurs hypothèses, résultats et limites

### Principes pédagogiques

La base suit plusieurs principes complémentaires :

- **Progression du simple au complexe** : les prérequis sont indiqués pour chaque cours
- **Compréhension par le pourquoi** : chaque méthode est reliée au problème qu'elle résout
- **Articulation théorie-pratique** : mathématiques, intuition, code et limites sont présentés ensemble
- **Interconnexion** : les cours comportent des liens vers les notions connexes
- **Reproductibilité** : les sources, hypothèses et choix importants sont explicités
- **Relisibilité** : les documents sont conçus pour rester compréhensibles plusieurs années plus tard

---

## 🗂️ Structure du Repository

```text
2nd-cerveau-travail/
│
├── 00_statistics_foundations/       # Fondations statistiques et probabilistes
│
├── 01_machine_learning/             # Machine Learning classique
│   ├── 01_fundamentals/             # Prétraitement et post-traitement
│   ├── 02_supervised/               # Apprentissage supervisé
│   ├── 03_unsupervised/             # Apprentissage non supervisé
│   └── 04_interpretability/         # Interprétabilité ML
│
├── 02_deep_learning/                # Réseaux de neurones profonds
│   ├── 01_architectures/            # CNN, ViT, YOLO, BERT
│   ├── 02_fundamentals/             # Composants fondamentaux
│   ├── 03_llm/                      # NLP, LLM, embeddings et génération
│   └── 04_interpretabilite/         # Interprétabilité Deep Learning
│
├── 03_infrastructure/               # Déploiement et serving de modèles IA
├── 04_databases/                    # Bases de données et SQL
└── 05_methodologie/                 # Méthodes de travail et communication R&D
```

### Conventions de nommage

- **Dossiers principaux numérotés** : progression globale des connaissances
- **Sous-dossiers thématiques** : spécialisation à l'intérieur d'un domaine
- **Fichiers en `snake_case`** : noms portables et cohérents
- **Formats Markdown** : documents lisibles dans GitHub et les éditeurs de texte
- **Formules en LaTeX** : blocs `$$ ... $$` pour le rendu mathématique GitHub

---

## 📚 Catalogue des Cours

### 00. Statistics Foundations

Fondations mathématiques nécessaires à l'analyse de données et au Machine Learning.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [00_INDEX_OUTILS_STATISTIQUES.md](00_statistics_foundations/00_INDEX_OUTILS_STATISTIQUES.md) | Index des outils statistiques | ⭐ Débutant → ⭐⭐ Intermédiaire | `#statistics` `#index` |
| [probability_foundations.md](00_statistics_foundations/probability_foundations.md) | Fondations de probabilités | ⭐⭐ Intermédiaire | `#probability` `#statistics` |
| [random_variables.md](00_statistics_foundations/random_variables.md) | Variables aléatoires | ⭐⭐ Intermédiaire | `#random-variables` `#probability` |
| [common_distributions.md](00_statistics_foundations/common_distributions.md) | Lois de probabilité usuelles | ⭐⭐ Intermédiaire | `#distributions` `#probability` |
| [measures_central_tendency.md](00_statistics_foundations/measures_central_tendency.md) | Mesures de tendance centrale | ⭐ Débutant | `#mean` `#median` `#statistics` |
| [measures_dispersion.md](00_statistics_foundations/measures_dispersion.md) | Mesures de dispersion | ⭐ Débutant | `#variance` `#standard-deviation` |
| [data_visualization_principles.md](00_statistics_foundations/data_visualization_principles.md) | Principes de visualisation des données | ⭐ Débutant → ⭐⭐ Intermédiaire | `#visualization` `#data-analysis` |

**Parcours recommandé** : probabilités → variables aléatoires → lois usuelles → mesures → visualisation.

---

### 01. Machine Learning

Algorithmes classiques de préparation, prédiction, réduction de dimension et interprétabilité.

#### 🔧 01_fundamentals - Fondamentaux

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [preprocessing_data.md](01_machine_learning/01_fundamentals/preprocessing_data.md) | Prétraitement des données | ⭐ Débutant | `#preprocessing` `#data-cleaning` |
| [postprocessing_data.md](01_machine_learning/01_fundamentals/postprocessing_data.md) | Post-traitement et interprétation des résultats | ⭐ Débutant | `#postprocessing` `#evaluation` |

#### 🎯 02_supervised - Apprentissage Supervisé

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [logistic_regression.md](01_machine_learning/02_supervised/logistic_regression.md) | Régression logistique | ⭐⭐ Intermédiaire | `#classification` `#logistic-regression` |
| [svm.md](01_machine_learning/02_supervised/svm.md) | Support Vector Machines | ⭐⭐ Intermédiaire | `#svm` `#classification` `#kernel` |
| [xgboost.md](01_machine_learning/02_supervised/xgboost.md) | Extreme Gradient Boosting | ⭐⭐ Intermédiaire | `#boosting` `#ensemble` `#xgboost` |
| [lightgbm.md](01_machine_learning/02_supervised/lightgbm.md) | Light Gradient Boosting Machine | ⭐⭐ Intermédiaire | `#boosting` `#lightgbm` `#performance` |

#### 🔍 03_unsupervised - Apprentissage Non-Supervisé

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [dimension_reduction/tsne.md](01_machine_learning/03_unsupervised/dimension_reduction/tsne.md) | t-SNE pour la visualisation | ⭐⭐ Intermédiaire | `#tsne` `#visualization` `#dimensionality` |
| [dimension_reduction/umap.md](01_machine_learning/03_unsupervised/dimension_reduction/umap.md) | UMAP pour la réduction de dimension | ⭐⭐ Intermédiaire | `#umap` `#visualization` `#dimensionality` |

#### 🔬 04_interpretability - Interprétabilité

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [shap_shapley_values.md](01_machine_learning/04_interpretability/shap_shapley_values.md) | SHAP et valeurs de Shapley | ⭐⭐⭐ Avancé | `#shap` `#explainability` `#shapley` |

---

### 02. Deep Learning

Architectures, composants et applications avancées des réseaux de neurones.

#### 🏗️ 01_architectures - Architectures de Réseaux

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [cnn.md](02_deep_learning/01_architectures/cnn.md) | Réseaux de neurones convolutifs | ⭐⭐ Intermédiaire | `#cnn` `#computer-vision` `#convolution` |
| [vit.md](02_deep_learning/01_architectures/vit.md) | Vision Transformers | ⭐⭐⭐ Avancé | `#transformer` `#vision` `#attention` |
| [yolo.md](02_deep_learning/01_architectures/yolo.md) | YOLO pour la détection d'objets | ⭐⭐⭐ Avancé | `#yolo` `#detection` `#real-time` |
| [bert_architecture_variantes.md](02_deep_learning/01_architectures/bert_architecture_variantes.md) | Architecture BERT et variantes françaises | ⭐⭐⭐ Avancé | `#bert` `#transformer` `#nlp` `#camembert` |

#### ⚙️ 02_fundamentals - Fondamentaux du Deep Learning

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [fonction_activation.md](02_deep_learning/02_fundamentals/fonction_activation.md) | Fonctions d'activation | ⭐ Débutant | `#activation` `#neural-networks` `#non-linearity` |

#### 🤖 03_llm - NLP, LLM et représentations sémantiques

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [apprentissage_renforcement_llm.md](02_deep_learning/03_llm/apprentissage_renforcement_llm.md) | Apprentissage par renforcement pour les LLM | ⭐⭐⭐ Avancé | `#rlhf` `#llm` `#alignment` |
| [distillation_llm.md](02_deep_learning/03_llm/distillation_llm.md) | Distillation des modèles de langage | ⭐⭐⭐ Avancé | `#distillation` `#compression` `#llm` |
| [peft_lora.md](02_deep_learning/03_llm/peft_lora.md) | PEFT et fine-tuning LoRA | ⭐⭐⭐ Avancé | `#peft` `#lora` `#fine-tuning` |
| [rag.md](02_deep_learning/03_llm/rag.md) | Retrieval-Augmented Generation | ⭐⭐⭐ Avancé | `#rag` `#retrieval` `#generation` `#llm` |
| [sbert_sentence_embeddings.md](02_deep_learning/03_llm/sbert_sentence_embeddings.md) | SBERT et modèles de sentence embeddings | ⭐⭐ → ⭐⭐⭐ Intermédiaire à avancé | `#sbert` `#embeddings` `#semantic-similarity` `#nlp` |
| [nlp_intent_detection_email_routing.md](02_deep_learning/03_llm/nlp_intent_detection_email_routing.md) | Extraction d'intention et routage sémantique | ⭐⭐ → ⭐⭐⭐ Intermédiaire à avancé | `#nlp` `#intent-detection` `#routing` `#embeddings` |

#### 🔬 04_interpretabilite - Interprétabilité

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [integrated_gradients.md](02_deep_learning/04_interpretabilite/integrated_gradients.md) | Integrated Gradients | ⭐⭐⭐ Avancé | `#explainability` `#interpretability` `#gradients` |

**Parcours recommandé** : fundamentals → architectures → BERT → SBERT/RAG → applications NLP → interprétabilité.

---

### 03. Infrastructure

Déploiement, exécution et mise à disposition des modèles IA.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [infra_ia.md](03_infrastructure/infra_ia.md) | Infrastructure et déploiement de systèmes IA | ⭐⭐ Intermédiaire | `#infrastructure` `#deployment` `#mlops` |
| [serving_llm.md](03_infrastructure/serving_llm.md) | Serving et mise en production des LLM | ⭐⭐⭐ Avancé | `#serving` `#llm` `#inference` |

---

### 04. Databases

Gestion et manipulation des bases de données relationnelles.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [sql/creation_tables.md](04_databases/sql/creation_tables.md) | Création et gestion de tables SQL | ⭐ Débutant | `#sql` `#database` `#tables` |

---

### 05. Méthodologie

Compétences transverses pour apprendre, documenter et présenter un travail de R&D.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [comprendre_papier_scientifique.md](05_methodologie/comprendre_papier_scientifique.md) | Lire et analyser un papier scientifique | ⭐⭐ → ⭐⭐⭐ Intermédiaire à avancé | `#papers` `#research` `#scientific-method` |
| [presenter_resultats_ia.md](05_methodologie/presenter_resultats_ia.md) | Présenter des résultats IA | ⭐⭐ Intermédiaire | `#presentation` `#communication` `#metrics` |
| [creer_document_presentation.md](05_methodologie/creer_document_presentation.md) | Créer un document de présentation | ⭐⭐ Intermédiaire | `#documentation` `#presentation` `#communication` |

---

## 🚀 Parcours Recommandés

### Parcours Machine Learning

```text
Statistics Foundations
        ↓
Preprocessing / Postprocessing
        ↓
Logistic Regression → SVM → XGBoost / LightGBM
        ↓
t-SNE / UMAP → SHAP
```

### Parcours Deep Learning et NLP

```text
Fonctions d'activation
        ↓
Architectures CNN / ViT / BERT
        ↓
Sentence Embeddings et SBERT
        ↓
RAG / PEFT-LoRA / LLM
        ↓
Serving et interprétabilité
```

### Parcours R&D et présentation

```text
Fondations techniques
        ↓
Comprendre un papier scientifique
        ↓
Évaluer une méthode
        ↓
Présenter les résultats
        ↓
Documenter et transmettre
```

---

## 📖 Comment Utiliser ce Repository

### Pour apprendre un sujet

1. Identifier les prérequis indiqués dans le cours.
2. Lire l'intuition et le problème avant les détails mathématiques.
3. Refaire les calculs sur les exemples numériques.
4. Exécuter et modifier les exemples de code.
5. Répondre aux questions de vérification.
6. Consulter les sources et les cours connexes.

### Pour retrouver une notion

- Utiliser la recherche GitHub sur les mots-clés.
- Consulter les index de section.
- Suivre les liens relatifs entre les cours.
- Commencer par le README lorsque l'on ne connaît pas encore le classement du sujet.

### Pour maintenir une note

Chaque cours doit conserver une séparation claire entre :

- les faits et définitions établis ;
- les hypothèses ou intuitions ;
- les résultats expérimentaux ;
- les limites ;
- les sources ;
- les pistes de travail futures.

---

## 🔗 Liens Entre les Cours

Le repository suit une logique de graphe de connaissances :

- les statistiques fondent l'analyse et l'évaluation des modèles ;
- le Machine Learning introduit les premières familles d'algorithmes ;
- le Deep Learning généralise les représentations apprises ;
- BERT introduit l'encodeur Transformer contextualisé ;
- SBERT spécialise cet encodeur pour produire des vecteurs de phrases comparables ;
- RAG exploite les embeddings pour rechercher du contexte ;
- l'infrastructure rend les modèles utilisables ;
- la méthodologie permet de vérifier, expliquer et transmettre les résultats.

---

## ✨ Contribuer

### Standards de documentation

Chaque cours doit comporter, autant que pertinent :

- **Métadonnées** : titre, date, domaine, niveau, durée et tags
- **Prérequis** : connaissances nécessaires ou recommandées
- **Objectifs d'apprentissage** : compétences visées
- **Progression pédagogique** : du problème vers la solution
- **Fondements théoriques** : définitions, hypothèses et formules
- **Exemples pratiques** : code ou calculs reproductibles
- **Limites** : situations où la méthode échoue ou doit être complétée
- **Évaluation** : métriques, protocoles et pièges
- **Ressources** : papiers, documentation et liens vérifiables

### Conventions

- **Format** : Markdown (`.md`)
- **Nomenclature** : `snake_case` pour les fichiers
- **Code** : blocs de code avec langage déclaré
- **Mathématiques** : blocs LaTeX `$$ ... $$`, sans macros non supportées par GitHub
- **Sources** : liens directs vers les références utilisées
- **Liens internes** : chemins relatifs vers les cours connexes

### Proposer une contribution

- Ouvrir une **issue** pour discuter d'une amélioration ou d'un nouveau cours.
- Créer une **Pull Request** pour ajouter ou corriger du contenu.
- Vérifier les liens, les formules et le rendu GitHub avant soumission.

---

## 📊 Statistiques

- **Total de cours** : 34
- **Dernière mise à jour** : 25 août 2026
- **Catégories principales** : 6
- **Cours statistiques** : 7
- **Cours Machine Learning** : 9
- **Cours Deep Learning** : 12
- **Cours Infrastructure** : 2
- **Cours Databases** : 1
- **Cours Méthodologie** : 3

---

## 🎓 Ressources Externes Recommandées

### Livres

- **Deep Learning** — Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **Pattern Recognition and Machine Learning** — Christopher Bishop
- **Hands-On Machine Learning** — Aurélien Géron

### Cours en ligne

- [Fast.ai](https://www.fast.ai/) — Deep Learning pratique
- [Stanford CS229](https://cs229.stanford.edu/) — Machine Learning
- [DeepLearning.AI](https://www.deeplearning.ai/) — Spécialisations IA

### Papiers fondateurs

Les références scientifiques propres à chaque sujet sont regroupées dans la section « Sources » de chaque cours.

---

## 📝 Licence

Ce repository est un projet personnel à usage éducatif.  
Les cours sont basés sur des connaissances publiques et des sources académiques citées.

---

## 📬 Contact

**ClemWasChoosen** — Développeur R&D  
🔗 [GitHub](https://github.com/ClemWasChoosen/2nd-cerveau-travail)
