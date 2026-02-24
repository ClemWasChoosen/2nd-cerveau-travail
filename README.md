# 🧠 Second Cerveau - Base de Connaissances Technique

> **Repository personnel de documentation en Machine Learning, Deep Learning, Infrastructure et Méthodologie**  
> Développeur R&D - Informatique & Intelligence Artificielle

[![Last Update](https://img.shields.io/badge/Last%20Update-February%202026-blue)]()
[![Cours](https://img.shields.io/badge/Cours-26-green)]()
[![Structure](https://img.shields.io/badge/Structure-Optimis%C3%A9e-success)]()

---

## 📋 Table des Matières

- [🎯 À Propos](#-à-propos)
- [🗂️ Structure du Repository](#️-structure-du-repository)
- [📚 Catalogue des Cours](#-catalogue-des-cours)
  - [01. Machine Learning](#01-machine-learning)
  - [02. Deep Learning](#02-deep-learning)
  - [03. Infrastructure](#03-infrastructure)
  - [04. Databases](#04-databases)
  - [05. Méthodologie](#05-méthodologie)
- [🚀 Navigation Rapide](#-navigation-rapide)
- [📖 Comment Utiliser ce Repository](#-comment-utiliser-ce-repository)
- [🔗 Liens Entre les Cours](#-liens-entre-les-cours)
- [✨ Contribuer](#-contribuer)

---

## 🎯 À Propos

Ce repository constitue mon **"Second Cerveau"** personnel - une base de connaissances structurée et évolutive sur l'informatique, le Machine Learning et l'Intelligence Artificielle.

### Objectifs

- 📝 **Documentation structurée** : Notes techniques relisibles et compréhensibles à long terme
- 🔗 **Interconnexion** : Liens bidirectionnels entre concepts (méthode Zettelkasten)
- 🎓 **Apprentissage continu** : Progression du fondamental vers l'avancé
- 🛠️ **Praticité** : Exemples concrets et applications réelles

### Principes Pédagogiques

Cette base documentaire suit les principes scientifiques de l'apprentissage :
- **Cognitive Load Theory** (Sweller) : Progression du simple au complexe
- **Multimédia Learning** (Mayer) : Combinaison texte + visualisations
- **Zettelkasten** : Notes atomiques et liens bidirectionnels
- **Framework Diátaxis** : Documentation technique structurée

---

## 🗂️ Structure du Repository

```
2nd-cerveau-travail/
│
├── 📂 01_machine_learning/          # Apprentissage automatique classique
│   ├── 01_fundamentals/             # Prétraitement et postprocessing
│   ├── 02_supervised/               # Algorithmes supervisés (SVM, XGBoost, etc.)
│   └── 03_unsupervised/             # Réduction de dimensionnalité, clustering
│
├── 📂 02_deep_learning/             # Réseaux de neurones profonds
│   ├── 01_architectures/            # CNN, ViT, YOLO
│   ├── 02_fundamentals/             # Fonctions d'activation, optimisation
│   ├── 03_llm/                      # Large Language Models
│   └── 04_interpretabilite/         # Explainability & Interpretability
│
├── 📂 03_infrastructure/            # Déploiement et infrastructure IA
│
├── 📂 04_databases/                 # Gestion de données
│   └── sql/                         # SQL et bases relationnelles
│
└── 📂 05_methodologie/              # Compétences transverses
    ├── Présentation des résultats
    └── Lecture de papers scientifiques
```

### Nomenclature

- **Numérotation** : Les dossiers principaux sont numérotés pour indiquer la progression logique
- **snake_case** : Convention de nommage cohérente pour tous les fichiers
- **Organisation thématique** : Regroupement par domaine puis spécialisation

---

## 📚 Catalogue des Cours

### 01. Machine Learning

#### 🔧 01_fundamentals - Fondamentaux
Concepts essentiels pour préparer et traiter les données.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [preprocessing_data.md](01_machine_learning/01_fundamentals/preprocessing_data.md) | Techniques de prétraitement des données | ⭐ Débutant | `#preprocessing` `#data-cleaning` |
| [postprocessing_data.md](01_machine_learning/01_fundamentals/postprocessing_data.md) | Post-traitement et interprétation des résultats | ⭐ Débutant | `#postprocessing` `#evaluation` |

#### 🎯 02_supervised - Apprentissage Supervisé
Algorithmes d'apprentissage supervisé classiques et modernes.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [svm.md](01_machine_learning/02_supervised/svm.md) | Support Vector Machines | ⭐⭐ Intermédiaire | `#svm` `#classification` `#kernel` |
| [xgboost.md](01_machine_learning/02_supervised/xgboost.md) | Extreme Gradient Boosting | ⭐⭐ Intermédiaire | `#boosting` `#ensemble` `#xgboost` |
| [lightgbm.md](01_machine_learning/02_supervised/lightgbm.md) | Light Gradient Boosting Machine | ⭐⭐ Intermédiaire | `#boosting` `#lightgbm` `#performance` |

#### 🔍 03_unsupervised - Apprentissage Non-Supervisé
Techniques de réduction de dimensionnalité et clustering.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [dimension_reduction/tsne.md](01_machine_learning/03_unsupervised/dimension_reduction/tsne.md) | t-Distributed Stochastic Neighbor Embedding | ⭐⭐ Intermédiaire | `#tsne` `#visualization` `#dimensionality` |

**📌 Parcours recommandé** : fundamentals → supervised → unsupervised

---

### 02. Deep Learning

#### 🏗️ 01_architectures - Architectures de Réseaux
Architectures fondamentales et modernes de deep learning.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [cnn.md](02_deep_learning/01_architectures/cnn.md) | Convolutional Neural Networks | ⭐⭐ Intermédiaire | `#cnn` `#computer-vision` `#convolution` |
| [vit.md](02_deep_learning/01_architectures/vit.md) | Vision Transformers | ⭐⭐⭐ Avancé | `#transformer` `#vision` `#attention` |
| [yolo.md](02_deep_learning/01_architectures/yolo.md) | You Only Look Once (Object Detection) | ⭐⭐⭐ Avancé | `#yolo` `#detection` `#real-time` |

#### ⚙️ 02_fundamentals - Fondamentaux du Deep Learning
Composants essentiels des réseaux de neurones.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [fonction_activation.md](02_deep_learning/02_fundamentals/fonction_activation.md) | Fonctions d'activation (ReLU, Sigmoid, etc.) | ⭐ Débutant | `#activation` `#neural-networks` `#non-linearity` |

#### 🤖 03_llm - Large Language Models
Techniques avancées pour les modèles de langage.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [apprentissage_renforcement_llm.md](02_deep_learning/03_llm/apprentissage_renforcement_llm.md) | RLHF - Reinforcement Learning from Human Feedback | ⭐⭐⭐ Avancé | `#rlhf` `#llm` `#alignment` |
| [distillation_llm.md](02_deep_learning/03_llm/distillation_llm.md) | Distillation de modèles de langage | ⭐⭐⭐ Avancé | `#distillation` `#compression` `#llm` |
| [rag.md](02_deep_learning/03_llm/rag.md) | Retrieval-Augmented Generation | ⭐⭐⭐ Avancé | `#rag` `#retrieval` `#generation` `#llm` |

#### 🔬 04_interpretabilite - Interprétabilité
Techniques pour comprendre et expliquer les décisions des modèles.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [integrated_gradients.md](02_deep_learning/04_interpretabilite/integrated_gradients.md) | Integrated Gradients pour l'explainability | ⭐⭐⭐ Avancé | `#explainability` `#interpretability` `#gradients` |

**📌 Parcours recommandé** : fundamentals → architectures → llm → interpretabilite

---

### 03. Infrastructure

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [infra_ia.md](03_infrastructure/infra_ia.md) | Infrastructure et déploiement de systèmes IA | ⭐⭐ Intermédiaire | `#infrastructure` `#deployment` `#mlops` |

---

### 04. Databases

#### 💾 SQL
Gestion et manipulation de bases de données relationnelles.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [sql/creation_tables.md](04_databases/sql/creation_tables.md) | Création et gestion de tables SQL | ⭐ Débutant | `#sql` `#database` `#tables` |

---

### 05. Méthodologie

Compétences transverses essentielles pour un développeur R&D.

| Cours | Description | Niveau | Tags |
|-------|-------------|--------|------|
| [presenter_resultats_ia.md](05_methodologie/presenter_resultats_ia.md) | Présenter efficacement des résultats IA | ⭐⭐ Intermédiaire | `#presentation` `#communication` `#results` |
| [comprendre_papier_scientifique.md](05_methodologie/comprendre_papier_scientifique.md) | Méthodologie de lecture de papers scientifiques | ⭐⭐ Intermédiaire | `#research` `#papers` `#reading` |

---

## 🚀 Navigation Rapide

### Par Niveau

#### ⭐ Débutant
- [Preprocessing Data](01_machine_learning/01_fundamentals/preprocessing_data.md)
- [Postprocessing Data](01_machine_learning/01_fundamentals/postprocessing_data.md)
- [Fonctions d'Activation](02_deep_learning/02_fundamentals/fonction_activation.md)
- [SQL - Création de Tables](04_databases/sql/creation_tables.md)

#### ⭐⭐ Intermédiaire
- [SVM](01_machine_learning/02_supervised/svm.md)
- [XGBoost](01_machine_learning/02_supervised/xgboost.md)
- [LightGBM](01_machine_learning/02_supervised/lightgbm.md)
- [t-SNE](01_machine_learning/03_unsupervised/dimension_reduction/tsne.md)
- [CNN](02_deep_learning/01_architectures/cnn.md)
- [Infrastructure IA](03_infrastructure/infra_ia.md)
- [Présenter Résultats IA](05_methodologie/presenter_resultats_ia.md)
- [Comprendre Papers Scientifiques](05_methodologie/comprendre_papier_scientifique.md)

#### ⭐⭐⭐ Avancé
- [Vision Transformers (ViT)](02_deep_learning/01_architectures/vit.md)
- [YOLO](02_deep_learning/01_architectures/yolo.md)
- [RLHF pour LLM](02_deep_learning/03_llm/apprentissage_renforcement_llm.md)
- [Distillation LLM](02_deep_learning/03_llm/distillation_llm.md)
- [RAG](02_deep_learning/03_llm/rag.md)
- [Integrated Gradients](02_deep_learning/04_interpretabilite/integrated_gradients.md)

### Par Thème

#### 🖼️ Computer Vision
- [CNN](02_deep_learning/01_architectures/cnn.md)
- [Vision Transformers](02_deep_learning/01_architectures/vit.md)
- [YOLO](02_deep_learning/01_architectures/yolo.md)

#### 📝 Natural Language Processing
- [RLHF](02_deep_learning/03_llm/apprentissage_renforcement_llm.md)
- [Distillation LLM](02_deep_learning/03_llm/distillation_llm.md)
- [RAG](02_deep_learning/03_llm/rag.md)

#### 🎯 Algorithmes Classiques
- [SVM](01_machine_learning/02_supervised/svm.md)
- [XGBoost](01_machine_learning/02_supervised/xgboost.md)
- [LightGBM](01_machine_learning/02_supervised/lightgbm.md)

#### 🔍 Explainability & Interpretability
- [Integrated Gradients](02_deep_learning/04_interpretabilite/integrated_gradients.md)

---

## 📖 Comment Utiliser ce Repository

### Pour Apprendre

1. **Commencez par les fondamentaux** : 
   - `01_machine_learning/01_fundamentals/`
   - `02_deep_learning/02_fundamentals/`

2. **Suivez la progression numérique** dans chaque catégorie

3. **Explorez les liens bidirectionnels** entre cours connexes

4. **Pratiquez avec les exemples** fournis dans chaque cours

### Pour Référence Rapide

- Utilisez la **recherche GitHub** (touche `/`) pour trouver des mots-clés
- Consultez les **tags** dans le catalogue pour identifier les cours pertinents
- Utilisez **Ctrl+F** dans ce README pour chercher un sujet spécifique

### Pour Contribuer à Vos Propres Notes

```bash
# Créer une nouvelle branche pour vos ajouts
git checkout -b ajout-nouveau-cours

# Respecter la nomenclature
# Format: categorie/sous-categorie/nom_du_cours.md
# Exemple: 02_deep_learning/03_llm/nouveau_concept.md

# Commit avec message descriptif
git commit -m "✨ Ajout cours sur [SUJET]"

# Push et créer une PR
git push origin ajout-nouveau-cours
```

---

## 🔗 Liens Entre les Cours

Les cours sont interconnectés selon la méthode **Zettelkasten**. Voici les principales relations :

### Parcours Machine Learning Complet

```
preprocessing_data.md
    ↓
svm.md / xgboost.md / lightgbm.md
    ↓
postprocessing_data.md
    ↓
presenter_resultats_ia.md
```

### Parcours Deep Learning Vision

```
fonction_activation.md
    ↓
cnn.md
    ↓
vit.md / yolo.md
    ↓
integrated_gradients.md
```

### Parcours LLM

```
fonction_activation.md
    ↓
apprentissage_renforcement_llm.md
    ↓
distillation_llm.md / rag.md
```

### Compétences Transverses

```
comprendre_papier_scientifique.md
    ↓
[Tout cours avancé]
    ↓
presenter_resultats_ia.md
    ↓
infra_ia.md
```

---

## ✨ Contribuer

### Standards de Documentation

Chaque cours suit un format structuré :

- **Métadonnées** : Titre, date, niveau, tags
- **Prérequis** : Connaissances nécessaires
- **Objectifs d'apprentissage** : Ce que vous allez maîtriser
- **Contenu théorique** : Concepts avec sources académiques
- **Exemples pratiques** : Code et cas d'usage réels
- **Ressources complémentaires** : Papers, articles, vidéos

### Conventions

- **Format** : Markdown (`.md`)
- **Nomenclature** : `snake_case` pour les fichiers
- **Code** : Blocs de code avec syntaxe appropriée
- **Math** : Formules en LaTeX (`$$formula$$`)
- **Sources** : Toujours citer les références

### Suggestions Bienvenues

Si vous utilisez ce repository comme inspiration et avez des suggestions :
- Ouvrez une **issue** pour proposer des améliorations
- Créez une **PR** pour ajouter du contenu
- Partagez vos retours d'expérience

---

## 📊 Statistiques

- **Total de cours** : 26
- **Dernière mise à jour** : Février 2026
- **Catégories principales** : 5
- **Niveaux** : Débutant (4) | Intermédiaire (8) | Avancé (6)

---

## 🎓 Ressources Externes Recommandées

### Livres
- **Deep Learning** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **Pattern Recognition and Machine Learning** - Christopher Bishop
- **Hands-On Machine Learning** - Aurélien Géron

### Cours en Ligne
- [Fast.ai](https://www.fast.ai/) - Deep Learning pratique
- [Stanford CS229](http://cs229.stanford.edu/) - Machine Learning
- [DeepLearning.AI](https://www.deeplearning.ai/) - Spécialisations IA

### Papers Fondateurs
- Consultez la section "Ressources" de chaque cours pour les papers spécifiques

---

## 📝 Licence

Ce repository est un projet personnel à usage éducatif.  
Les cours sont basés sur des connaissances publiques et des sources académiques citées.

---

## 📬 Contact

**Vito Corleone** - Développeur R&D  
🔗 [GitHub](https://github.com/ClemWasChoosen/2nd-cerveau-travail)

---

<div align="center">

**⭐ Si ce repository vous est utile, n'hésitez pas à le mettre en favoris ! ⭐**

*"Le savoir est la seule richesse qui s'accroît quand on la partage."*

</div>
