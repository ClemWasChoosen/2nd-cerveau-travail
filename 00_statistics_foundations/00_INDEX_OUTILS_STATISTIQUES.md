# 📊 INDEX COMPLET DES OUTILS STATISTIQUES POUR DATA SCIENCE & IA

> **Votre guide de référence** : Ce document est votre carte pour naviguer dans l'arsenal statistique du Data Scientist. Utilisez-le pour trouver rapidement quel outil utiliser selon votre problème.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-18 |
| **Dernière mise à jour** | 2026-03-18 |
| **Domaine** | Statistiques / Data Science / IA |
| **Niveau** | Tous niveaux (Index de navigation) |
| **Durée de lecture** | ~15 minutes |
| **Fichier** | `00_INDEX_STATISTICS.md` |
| **Emplacement** | `/00_statistics_foundations/` |
| **Tags** | `#statistics` `#index` `#reference` `#decision-tree` |

---

## 🎯 Comment Utiliser Cet Index

Cet index est conçu comme un **système de navigation en 3 dimensions** :

1. **📂 Navigation par Domaine** : Parcourir les grandes catégories statistiques
2. **❓ Navigation par Question** : "J'ai ce problème, quel outil utiliser ?"
3. **🔧 Navigation par Outil** : Référence alphabétique de tous les outils

**Légende des symboles** :
- ⭐ **Essentiel** : À maîtriser absolument
- 🔥 **Très utilisé** : Usage quotidien en Data Science
- 🎓 **Avancé** : Nécessite bases solides
- 💡 **Pratique** : Application directe métier
- 📐 **Mathématique** : Forte composante théorique

---

## 📂 PARTIE 1 : Navigation par Domaine Statistique

### 🎯 Module 1 : Statistiques Descriptives

> **Objectif** : Résumer et décrire vos données

| Outil | Utilité | Cours associé | Priorité |
|-------|---------|---------------|----------|
| **Moyenne, Médiane, Mode** | Mesurer la tendance centrale | [[measures_central_tendency]] | ⭐🔥 |
| **Variance, Écart-type** | Mesurer la dispersion | [[measures_dispersion]] | ⭐🔥 |
| **Quantiles, Percentiles** | Comprendre la distribution | [[measures_dispersion]] | ⭐🔥 |
| **Coefficient de variation** | Comparer dispersions d'échelles différentes | [[measures_dispersion]] | 🔥 |
| **Skewness (Asymétrie)** | Détecter biais dans distribution | [[distribution_analysis]] | 🔥 |
| **Kurtosis (Aplatissement)** | Détecter queues lourdes/outliers | [[distribution_analysis]] | 🔥 |
| **Histogrammes** | Visualiser distribution | [[data_visualization_principles]] | ⭐🔥 |
| **Box plots** | Détecter outliers et quartiles | [[data_visualization_principles]] | ⭐🔥 |
| **Scatter plots** | Visualiser relations bivariées | [[data_visualization_principles]] | ⭐🔥 |

**Quand utiliser ce module ?**
- Première exploration d'un nouveau jeu de données
- Rapport de qualité de données
- Communication de résultats à non-experts

---

### 🎲 Module 2 : Théorie des Probabilités

> **Objectif** : Modéliser l'incertitude et les processus aléatoires

#### 2.1 Fondamentaux

| Concept | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Espace probabilisé** | Base formelle des probabilités | [[probability_foundations]] | 📐⭐ |
| **Probabilité conditionnelle** | P(A\|B) - Modéliser dépendances | [[probability_foundations]] | ⭐🔥 |
| **Théorème de Bayes** | Inférence inverse (cause → effet) | [[bayesian_foundations]] | ⭐🔥💡 |
| **Indépendance** | Simplifier modèles complexes | [[probability_foundations]] | ⭐🔥 |
| **Variables aléatoires** | Modéliser quantités incertaines | [[random_variables]] | ⭐📐 |
| **Espérance** | Valeur moyenne attendue | [[random_variables]] | ⭐🔥 |
| **Variance** | Incertitude autour de l'espérance | [[random_variables]] | ⭐🔥 |
| **Covariance** | Mesurer dépendance linéaire | [[correlation_covariance]] | ⭐🔥 |
| **Corrélation de Pearson** | Dépendance linéaire normalisée [-1,1] | [[correlation_covariance]] | ⭐🔥💡 |

#### 2.2 Distributions de Probabilité Discrètes

| Distribution | Modélise | Cours associé | Priorité |
|--------------|----------|---------------|----------|
| **Bernoulli** | Succès/échec unique (ex: erreur oui/non) | [[common_distributions]] | ⭐🔥💡 |
| **Binomiale** | Nombre de succès sur n essais | [[common_distributions]] | ⭐🔥💡 |
| **Multinomiale** | Généralisation binomiale (>2 catégories) | [[common_distributions]] | 🔥💡 |
| **Géométrique** | Temps d'attente avant 1er succès | [[common_distributions]] | 🔥 |
| **Poisson** | Nombre d'événements rares (ex: clics/jour) | [[common_distributions]] | ⭐🔥💡 |
| **Binomiale négative** | Surdispersion (variance > moyenne) | [[common_distributions]] | 🎓 |
| **Hypergéométrique** | Échantillonnage sans remise | [[common_distributions]] | 🔥 |

**Cas d'usage typiques** :
- **Bernoulli/Binomiale** : Validation qualité données (votre exemple !)
- **Poisson** : Trafic web, nombre de pannes, événements rares
- **Multinomiale** : Classification multiclasse, NLP (sac de mots)

#### 2.3 Distributions de Probabilité Continues

| Distribution | Modélise | Cours associé | Priorité |
|--------------|----------|---------------|----------|
| **Uniforme** | Incertitude totale sur intervalle | [[common_distributions]] | ⭐🔥 |
| **Normale (Gaussienne)** | Phénomènes naturels, erreurs | [[gaussian_distribution]] | ⭐⭐🔥💡 |
| **Log-Normale** | Quantités multiplicatives (prix, tailles) | [[common_distributions]] | 🔥💡 |
| **Exponentielle** | Temps entre événements | [[common_distributions]] | 🔥💡 |
| **Gamma** | Somme de variables exponentielles | [[common_distributions]] | 🎓 |
| **Bêta** | Probabilités (valeurs dans [0,1]) | [[common_distributions]] | 🎓💡 |
| **Student (t)** | Petits échantillons, variance inconnue | [[student_distribution]] | ⭐🔥💡 |
| **Chi-deux (χ²)** | Tests de variance, goodness-of-fit | [[chi_square_distribution]] | ⭐🔥💡 |
| **Fisher (F)** | Comparaison de variances, ANOVA | [[fisher_distribution]] | 🔥💡 |
| **Weibull** | Fiabilité, durée de vie | [[common_distributions]] | 🎓💡 |

**Cas d'usage typiques** :
- **Normale** : Modèle par défaut (TCL), Machine Learning
- **Log-Normale** : Prix immobiliers, revenus, tailles de fichiers
- **Exponentielle** : Temps d'attente, durée de vie composants
- **Student** : Tests statistiques sur petits échantillons
- **Chi-deux** : Tests d'adéquation, indépendance

#### 2.4 Théorèmes Fondamentaux

| Théorème | Utilité | Cours associé | Priorité |
|----------|---------|---------------|----------|
| **Loi des Grands Nombres** | Moyenne empirique → espérance théorique | [[limit_theorems]] | ⭐📐 |
| **Théorème Central Limite** | Somme de VA → Normale (justifie tout !) | [[limit_theorems]] | ⭐⭐🔥📐 |
| **Inégalité de Markov** | Borne supérieure de probabilité | [[inequalities]] | 🎓📐 |
| **Inégalité de Tchebychev** | Borne via variance | [[inequalities]] | 🎓📐 |
| **Inégalité de Hoeffding** | Concentration (ML theory) | [[concentration_inequalities]] | 🎓📐 |

---

### 📊 Module 3 : Statistiques Inférentielles

> **Objectif** : Faire des conclusions sur une population à partir d'un échantillon

#### 3.1 Théorie de l'Échantillonnage

| Concept | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Échantillon aléatoire simple** | Base de l'inférence | [[sampling_theory]] | ⭐🔥💡 |
| **Échantillonnage stratifié** | Garantir représentativité | [[sampling_methods]] | 🔥💡 |
| **Échantillonnage par grappes** | Réduire coûts de collecte | [[sampling_methods]] | 🔥💡 |
| **Échantillonnage systématique** | Simplicité pratique | [[sampling_methods]] | 🔥💡 |
| **Bootstrap** | Estimer distribution via rééchantillonnage | [[bootstrap_methods]] | ⭐🔥💡 |
| **Distribution d'échantillonnage** | Variabilité de l'estimateur | [[sampling_theory]] | ⭐📐 |
| **Erreur standard** | Précision de l'estimateur | [[sampling_theory]] | ⭐🔥💡 |
| **Biais d'échantillonnage** | Identifier et corriger | [[sampling_bias]] | 🔥💡 |

**Votre cas d'usage** : Validation qualité de données
→ Cours : [[sample_size_quality_control]] ⭐💡

#### 3.2 Estimation Ponctuelle

| Méthode | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Estimateur sans biais** | Espérance = paramètre vrai | [[point_estimation]] | ⭐📐 |
| **Estimateur convergent** | Converge vers vraie valeur | [[point_estimation]] | ⭐📐 |
| **Efficacité** | Variance minimale | [[point_estimation]] | 🎓📐 |
| **Maximum de Vraisemblance** | Méthode universelle d'estimation | [[maximum_likelihood]] | ⭐⭐🔥📐 |
| **Méthode des Moments** | Alternative simple au MLE | [[method_of_moments]] | 🔥📐 |
| **Estimation bayésienne** | Intégrer connaissances a priori | [[bayesian_estimation]] | ⭐🎓💡 |
| **Maximum a Posteriori (MAP)** | MLE bayésien | [[bayesian_estimation]] | 🔥🎓 |

**Lien ML** : Le MLE est la base de la plupart des algorithmes ML !

#### 3.3 Intervalles de Confiance

| Type | Utilité | Cours associé | Priorité |
|------|---------|---------------|----------|
| **IC pour moyenne (variance connue)** | Cas théorique | [[confidence_intervals]] | ⭐🔥 |
| **IC pour moyenne (variance inconnue)** | Cas réel (Student) | [[confidence_intervals]] | ⭐🔥💡 |
| **IC pour proportion** | Taux d'erreur, taux de conversion | [[confidence_intervals]] | ⭐🔥💡 |
| **IC pour variance** | Chi-deux | [[confidence_intervals]] | 🔥 |
| **IC pour différence de moyennes** | Comparer 2 groupes | [[confidence_intervals]] | ⭐🔥💡 |
| **IC bootstrap** | Méthode non paramétrique | [[bootstrap_methods]] | 🔥💡 |
| **IC bayésien (crédibilité)** | Interprétation probabiliste | [[bayesian_inference]] | 🎓💡 |

**Cas d'usage typiques** :
- **Proportion** : "Le taux d'erreur est entre 1% et 3% (95% confiance)"
- **Différence de moyennes** : A/B testing

#### 3.4 Tests d'Hypothèses

| Test | Hypothèse testée | Cours associé | Priorité |
|------|------------------|---------------|----------|
| **Test Z (1 échantillon)** | Moyenne = μ₀ (var connue) | [[hypothesis_testing]] | ⭐🔥 |
| **Test t de Student (1 échantillon)** | Moyenne = μ₀ (var inconnue) | [[hypothesis_testing]] | ⭐🔥💡 |
| **Test t de Student (2 échantillons)** | μ₁ = μ₂ (moyennes égales) | [[two_sample_tests]] | ⭐🔥💡 |
| **Test t apparié** | Avant/après sur mêmes sujets | [[paired_tests]] | 🔥💡 |
| **Test de proportion (1 échantillon)** | Proportion = p₀ | [[proportion_tests]] | ⭐🔥💡 |
| **Test de proportions (2 échantillons)** | p₁ = p₂ | [[proportion_tests]] | 🔥💡 |
| **Test du Chi-deux d'indépendance** | Variables catégorielles indépendantes | [[chi_square_tests]] | ⭐🔥💡 |
| **Test du Chi-deux d'adéquation** | Distribution suit modèle théorique | [[chi_square_tests]] | 🔥💡 |
| **Test de Fisher (exact)** | Alternative Chi-deux (petits effectifs) | [[fisher_exact_test]] | 🔥💡 |
| **Test F (Fisher)** | Égalité de variances | [[variance_tests]] | 🔥💡 |
| **Test de Kolmogorov-Smirnov** | Comparer distributions | [[nonparametric_tests]] | 🔥💡 |
| **Test de Shapiro-Wilk** | Test de normalité | [[normality_tests]] | ⭐🔥💡 |
| **Test de Levene** | Égalité de variances (robuste) | [[variance_tests]] | 🔥💡 |

**Concepts transverses** :
- **p-value** : Probabilité d'observer ces données sous H₀
- **Erreur de Type I (α)** : Faux positif (rejeter H₀ à tort)
- **Erreur de Type II (β)** | Faux négatif (ne pas rejeter H₀ à tort)
- **Puissance (1-β)** : Probabilité de détecter effet réel
→ Cours : [[hypothesis_testing_foundations]] ⭐⭐🔥💡

#### 3.5 Tests Non Paramétriques

| Test | Équivalent paramétrique | Cours associé | Priorité |
|------|------------------------|---------------|----------|
| **Test des signes** | Test t 1 échantillon | [[nonparametric_tests]] | 🔥 |
| **Test de Wilcoxon (signed-rank)** | Test t apparié | [[nonparametric_tests]] | 🔥💡 |
| **Test de Mann-Whitney (U)** | Test t 2 échantillons | [[nonparametric_tests]] | ⭐🔥💡 |
| **Test de Kruskal-Wallis** | ANOVA | [[nonparametric_tests]] | 🔥💡 |
| **Test de Friedman** | ANOVA répétées | [[nonparametric_tests]] | 🔥 |
| **Coefficient de Spearman** | Corrélation de Pearson | [[correlation_tests]] | 🔥💡 |
| **Tau de Kendall** | Corrélation alternative | [[correlation_tests]] | 🔥 |

**Quand utiliser ?**
- Données non normales et transformation impossible
- Petits échantillons
- Données ordinales (rangs)
- Robustesse aux outliers

---

### 🧪 Module 4 : Design Expérimental et A/B Testing

> **Objectif** : Planifier expériences rigoureuses et interpréter résultats

| Concept/Outil | Utilité | Cours associé | Priorité |
|---------------|---------|---------------|----------|
| **Calcul de taille d'échantillon** | Garantir puissance statistique | [[sample_size_calculation]] | ⭐🔥💡 |
| **Analyse de puissance** | Évaluer capacité détection effet | [[power_analysis]] | ⭐🔥💡 |
| **A/B Testing** | Comparer 2 versions (web, produit) | [[ab_testing]] | ⭐🔥💡 |
| **Tests multivariés** | Comparer >2 versions | [[multivariate_testing]] | 🔥💡 |
| **Correction de Bonferroni** | Taux erreur famille (tests multiples) | [[multiple_testing]] | ⭐🔥💡 |
| **False Discovery Rate (FDR)** | Benjamini-Hochberg (moins conservateur) | [[multiple_testing]] | 🔥💡 |
| **Randomisation** | Éliminer biais de sélection | [[experimental_design]] | ⭐🔥💡 |
| **Blocking** | Contrôler variables confondantes | [[experimental_design]] | 🔥💡 |
| **Plan factoriel** | Tester plusieurs facteurs simultanément | [[factorial_design]] | 🎓💡 |
| **Sequential testing** | Arrêt anticipé (économies) | [[sequential_analysis]] | 🎓💡 |

**Cas d'usage typiques** :
- **A/B Testing** : Landing pages, features produit, algorithmes ML
- **Sample size** : Planifier études avant collecte de données
- **Multiple testing** : Feature selection en ML, analyses exploratoires

---

### 📈 Module 5 : Régression et Modélisation

> **Objectif** : Modéliser relations entre variables

#### 5.1 Régression Linéaire (Vue Statistique)

| Concept | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Moindres Carrés Ordinaires (OLS)** | Estimer coefficients de régression | [[linear_regression_statistical]] | ⭐🔥💡 |
| **Hypothèses de la régression** | Validité du modèle | [[linear_regression_statistical]] | ⭐🔥💡 |
| **R² (coefficient de détermination)** | Qualité d'ajustement | [[regression_diagnostics]] | ⭐🔥💡 |
| **R² ajusté** | Pénalise nombre de variables | [[regression_diagnostics]] | 🔥💡 |
| **Test F global** | Significativité du modèle | [[linear_regression_statistical]] | ⭐🔥 |
| **Tests t sur coefficients** | Significativité de chaque variable | [[linear_regression_statistical]] | ⭐🔥💡 |
| **Intervalles de confiance (coefficients)** | Précision estimation | [[linear_regression_statistical]] | 🔥💡 |
| **Intervalles de prédiction** | Incertitude prédictions nouvelles | [[linear_regression_statistical]] | 🔥💡 |
| **Résidus** | Écarts modèle/données | [[regression_diagnostics]] | ⭐🔥💡 |
| **QQ-plot** | Vérifier normalité résidus | [[regression_diagnostics]] | 🔥💡 |
| **Hétéroscédasticité** | Variance non constante | [[regression_diagnostics]] | 🔥💡 |
| **Multicolinéarité** | Variables prédictives corrélées | [[regression_diagnostics]] | 🔥💡 |
| **VIF (Variance Inflation Factor)** | Mesurer multicolinéarité | [[regression_diagnostics]] | 🔥💡 |
| **Leverage / Influence** | Identifier observations influentes | [[regression_diagnostics]] | 🔥💡 |
| **Distance de Cook** | Quantifier influence | [[regression_diagnostics]] | 🔥💡 |

#### 5.2 Régression Généralisée

| Modèle | Cas d'usage | Cours associé | Priorité |
|--------|-------------|---------------|----------|
| **Régression Logistique** | Prédiction binaire (0/1) | [[logistic_regression]] | ⭐🔥💡 |
| **Régression de Poisson** | Comptages (événements rares) | [[glm_models]] | 🔥💡 |
| **Régression multinomiale** | Classification multiclasse | [[multinomial_regression]] | 🔥💡 |
| **Régression ordinale** | Variables ordinales (ex: satisfaction) | [[ordinal_regression]] | 🔥💡 |

#### 5.3 Sélection de Modèles

| Méthode | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **AIC (Akaike)** | Comparer modèles (pénalise complexité) | [[model_selection]] | ⭐🔥💡 |
| **BIC (Bayesian)** | Alternative AIC (pénalité plus forte) | [[model_selection]] | 🔥💡 |
| **Sélection pas à pas (stepwise)** | Automatiser sélection variables | [[variable_selection]] | 🔥💡 |
| **Validation croisée** | Évaluer performance out-of-sample | [[cross_validation]] | ⭐🔥💡 |
| **Bootstrap pour régression** | Stabilité coefficients | [[bootstrap_regression]] | 🔥 |

---

### 🔀 Module 6 : Analyse Multivariée

> **Objectif** : Analyser relations complexes entre multiples variables

| Méthode | Objectif | Cours associé | Priorité |
|---------|----------|---------------|----------|
| **Matrice de corrélation** | Vue d'ensemble des relations | [[correlation_covariance]] | ⭐🔥💡 |
| **Matrice de covariance** | Corrélations non normalisées | [[correlation_covariance]] | ⭐🔥 |
| **PCA (vue statistique)** | Réduction dimensionnalité | [[pca_statistical_perspective]] | ⭐🔥💡 |
| **Analyse Factorielle** | Identifier variables latentes | [[factor_analysis]] | 🎓💡 |
| **Analyse en Composantes Indépendantes (ICA)** | Séparation de sources | [[ica_analysis]] | 🎓💡 |
| **Analyse Discriminante Linéaire (LDA)** | Séparation de classes | [[lda_analysis]] | 🔥💡 |
| **Analyse Canonique** | Corrélation entre 2 ensembles | [[canonical_correlation]] | 🎓 |
| **MANOVA** | ANOVA multivariée | [[manova]] | 🎓💡 |
| **Clustering Hiérarchique** | Dendrogrammes | [[hierarchical_clustering]] | 🔥💡 |
| **K-means (vue statistique)** | Partitionnement optimal | [[kmeans_statistical]] | ⭐🔥💡 |
| **Gaussian Mixture Models (GMM)** | Clustering probabiliste | [[gaussian_mixture_models]] | ⭐🔥💡 |
| **Modèles graphiques** | Dépendances conditionnelles | [[graphical_models]] | 🎓📐 |

---

### 📊 Module 7 : ANOVA et Comparaisons Multiples

> **Objectif** : Comparer plus de 2 groupes

| Test/Méthode | Utilité | Cours associé | Priorité |
|--------------|---------|---------------|----------|
| **ANOVA à 1 facteur** | Comparer k groupes (1 variable) | [[anova_one_way]] | ⭐🔥💡 |
| **ANOVA à 2 facteurs** | 2 variables indépendantes | [[anova_two_way]] | 🔥💡 |
| **ANOVA à mesures répétées** | Même sujet, conditions multiples | [[anova_repeated_measures]] | 🔥💡 |
| **ANCOVA** | ANOVA avec covariables | [[ancova]] | 🎓💡 |
| **Tests post-hoc (Tukey, Scheffé)** | Comparer groupes après ANOVA | [[posthoc_tests]] | 🔥💡 |
| **Contrastes planifiés** | Hypothèses a priori | [[contrasts]] | 🎓 |

---

### ⏱️ Module 8 : Séries Temporelles

> **Objectif** : Analyser données dépendantes du temps

| Concept/Modèle | Utilité | Cours associé | Priorité |
|----------------|---------|---------------|----------|
| **Autocorrélation (ACF)** | Corrélation avec lags | [[time_series_basics]] | ⭐🔥💡 |
| **Autocorrélation partielle (PACF)** | Corrélation directe | [[time_series_basics]] | 🔥💡 |
| **Stationnarité** | Propriétés constantes dans le temps | [[stationarity]] | ⭐🔥💡 |
| **Test de Dickey-Fuller** | Tester stationnarité | [[stationarity_tests]] | 🔥💡 |
| **Décomposition (trend, saisonnalité)** | Séparer composantes | [[time_series_decomposition]] | 🔥💡 |
| **ARIMA** | Modèle autorégressif intégré | [[arima_models]] | ⭐🔥💡 |
| **SARIMA** | ARIMA avec saisonnalité | [[sarima_models]] | 🔥💡 |
| **Modèles ARCH/GARCH** | Modéliser volatilité | [[volatility_models]] | 🎓💡 |
| **Test de causalité de Granger** | X cause-t-elle Y ? | [[granger_causality]] | 🎓💡 |

---

### 🔍 Module 9 : Analyse de Survie

> **Objectif** : Modéliser temps jusqu'à événement

| Méthode | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Fonction de survie** | Probabilité survie au-delà de t | [[survival_analysis]] | 🔥💡 |
| **Courbe de Kaplan-Meier** | Estimation non paramétrique | [[kaplan_meier]] | 🔥💡 |
| **Modèle de Cox (risques proportionnels)** | Régression de survie | [[cox_regression]] | 🔥💡 |
| **Censure** | Données incomplètes | [[survival_analysis]] | 🔥💡 |
| **Log-rank test** | Comparer courbes de survie | [[logrank_test]] | 🔥💡 |

**Applications** : Churn prediction, fiabilité, durée de vie client

---

### 🎲 Module 10 : Statistiques Bayésiennes

> **Objectif** : Intégrer connaissances a priori et quantifier incertitude

| Concept | Utilité | Cours associé | Priorité |
|---------|---------|---------------|----------|
| **Théorème de Bayes** | Mise à jour croyances | [[bayesian_foundations]] | ⭐🔥💡 |
| **Prior (a priori)** | Connaissances initiales | [[bayesian_priors]] | ⭐🎓 |
| **Likelihood (vraisemblance)** | Information des données | [[bayesian_inference]] | ⭐🎓 |
| **Posterior (a posteriori)** | Croyances mises à jour | [[bayesian_inference]] | ⭐🎓 |
| **Prior conjugué** | Simplification calculs | [[conjugate_priors]] | 🎓📐 |
| **MAP (Maximum a Posteriori)** | Estimation bayésienne | [[bayesian_estimation]] | 🔥🎓 |
| **Intervalle de crédibilité** | IC bayésien | [[bayesian_inference]] | 🎓💡 |
| **MCMC (Monte Carlo)** | Échantillonnage posterior complexe | [[mcmc_methods]] | 🎓📐💡 |
| **Metropolis-Hastings** | Algorithme MCMC | [[mcmc_methods]] | 🎓📐 |
| **Gibbs Sampling** | MCMC pour modèles graphiques | [[gibbs_sampling]] | 🎓📐 |
| **Modèles hiérarchiques bayésiens** | Pooling partiel d'information | [[hierarchical_bayes]] | 🎓💡 |
| **A/B testing bayésien** | Alternative tests fréquentistes | [[bayesian_ab_testing]] | 🔥💡 |

---

### 🤖 Module 11 : Statistiques pour Machine Learning

> **Objectif** : Fondements statistiques des algorithmes ML

| Concept | Lien avec ML | Cours associé | Priorité |
|---------|--------------|---------------|----------|
| **Biais-Variance Tradeoff** | Overfitting vs underfitting | [[bias_variance_tradeoff]] | ⭐⭐🔥💡 |
| **Validation croisée** | Évaluer généralisation | [[cross_validation]] | ⭐⭐🔥💡 |
| **Bootstrap** | Bagging, Random Forests | [[bootstrap_methods]] | ⭐🔥💡 |
| **Maximum de Vraisemblance** | Loss functions en ML | [[maximum_likelihood]] | ⭐🔥📐 |
| **Régularisation (vue statistique)** | Ridge (L2), Lasso (L1) | [[regularization_statistical]] | ⭐🔥💡 |
| **Inférence causale** | Distinguer corrélation/causalité | [[causal_inference]] | 🎓💡 |
| **Propensity Score Matching** | Biais de sélection | [[propensity_score]] | 🎓💡 |
| **A/B testing vs causalité** | Inférence rigoureuse | [[causal_inference_ab]] | 🔥💡 |
| **PAC Learning** | Théorie de l'apprentissage | [[pac_learning]] | 🎓📐 |
| **VC Dimension** | Capacité de généralisation | [[vc_dimension]] | 🎓📐 |

---

## ❓ PARTIE 2 : Navigation par Question Pratique

### 🔍 "J'explore un nouveau jeu de données"

**Workflow recommandé** :

1. **Statistiques descriptives** → [[measures_central_tendency]], [[measures_dispersion]]
2. **Visualisations** → [[data_visualization_principles]]
3. **Détection outliers** → Box plots, Z-scores
4. **Test de normalité** → [[shapiro_wilk_test]]
5. **Analyse de corrélations** → [[correlation_covariance]]
6. **Valeurs manquantes** → [[missing_data_analysis]]

---

### ✅ "Je veux valider la qualité de mes données"

**Votre cas d'usage : "Quelle taille d'échantillon pour détecter des erreurs ?"**

**Outils nécessaires** :
1. **Modélisation** : Loi de Bernoulli (présence/absence erreur) → [[common_distributions]]
2. **Estimation** : Proportion d'erreurs avec IC → [[confidence_intervals]]
3. **Test** : H₀ = "Taux d'erreur ≤ seuil acceptable" → [[proportion_tests]]
4. **Calcul taille échantillon** : Garantir puissance → [[sample_size_calculation]]
5. **Compromis** : Coût échantillonnage vs risque → [[power_analysis]]

**Cours dédié recommandé** : [[sample_size_quality_control]] ⭐💡

**Formule rapide** (pour proportion) :

$$n = \frac{Z_{\alpha/2}^2 \cdot p(1-p)}{E^2}$$

Où :
- $$n$$ = taille échantillon
- $$Z_{\alpha/2}$$ = quantile normale (1.96 pour 95% confiance)
- $$p$$ = proportion estimée d'erreurs
- $$E$$ = marge d'erreur acceptable

---

### 🆚 "Je veux comparer 2 groupes"

**Arbre de décision** :

1. **Variable numérique** (ex: temps de chargement A vs B)
   - Normalité OK + variances égales → **Test t de Student** [[two_sample_tests]]
   - Normalité OK + variances inégales → **Test t de Welch** [[two_sample_tests]]
   - Normalité NON → **Test de Mann-Whitney** [[nonparametric_tests]]
   
2. **Variable catégorielle** (ex: taux de clic A vs B)
   - Effectifs ≥ 30 → **Test Z de proportions** [[proportion_tests]]
   - Effectifs < 30 → **Test exact de Fisher** [[fisher_exact_test]]
   
3. **Mesures appariées** (avant/après même individu)
   - Normalité OK → **Test t apparié** [[paired_tests]]
   - Normalité NON → **Test de Wilcoxon** [[nonparametric_tests]]

---

### 🔢 "Je veux comparer plus de 2 groupes"

**Arbre de décision** :

1. **Variable numérique + 1 facteur** (ex: méthodes A, B, C, D)
   - Normalité OK + variances égales → **ANOVA 1 facteur** [[anova_one_way]]
   - Normalité NON → **Test de Kruskal-Wallis** [[nonparametric_tests]]
   - Puis tests post-hoc → [[posthoc_tests]]

2. **Variable numérique + 2 facteurs** (ex: méthode × pays)
   - → **ANOVA 2 facteurs** [[anova_two_way]]
   - Tester interactions

3. **Mesures répétées** (même sujet, plusieurs conditions)
   - → **ANOVA à mesures répétées** [[anova_repeated_measures]]

---

### 📈 "Je veux prédire une variable continue"

**Workflow** :

1. **Régression linéaire simple** (1 prédicteur) → [[linear_regression_statistical]]
2. **Régression linéaire multiple** (plusieurs prédicteurs) → [[linear_regression_statistical]]
3. **Vérifier hypothèses** :
   - Linéarité : Scatter plots
   - Normalité résidus : QQ-plot → [[regression_diagnostics]]
   - Homoscédasticité : Résidus vs fitted → [[regression_diagnostics]]
   - Indépendance : Durbin-Watson → [[regression_diagnostics]]
4. **Détecter problèmes** :
   - Multicolinéarité : VIF → [[regression_diagnostics]]
   - Observations influentes : Cook's distance → [[regression_diagnostics]]
5. **Sélection de variables** : AIC, BIC, stepwise → [[model_selection]]
6. **Validation** : Cross-validation → [[cross_validation]]

**Alternatives si hypothèses non respectées** :
- Transformation variables (log, Box-Cox)
- Régression robuste
- Régression non paramétrique (LOESS, GAM)

---

### 🎯 "Je veux prédire une variable binaire"

**Outils** :

1. **Régression logistique** → [[logistic_regression]] ⭐
2. **Interprétation** :
   - Odds Ratios
   - Effets marginaux
3. **Évaluation** :
   - Matrice de confusion
   - ROC, AUC → [[classification_metrics]]
   - Test de Hosmer-Lemeshow (goodness-of-fit)
4. **Sélection variables** : Tests de Wald, AIC, BIC
5. **Validation** : Cross-validation

---

### 🧪 "Je fais un A/B test"

**Checklist complète** :

1. **Avant le test** :
   - Calculer taille échantillon → [[sample_size_calculation]]
   - Définir métrique primaire (1 seule !)
   - Fixer durée du test
   - Randomisation correcte

2. **Pendant le test** :
   - Ne PAS regarder résultats avant la fin (peeking problem)
   - Ou utiliser sequential testing → [[sequential_analysis]]

3. **Après le test** :
   - **Métrique numérique** (ex: temps sur page) :
     - Test t de Student → [[two_sample_tests]]
   - **Métrique binaire** (ex: conversion) :
     - Test de proportions → [[proportion_tests]]
   - **Approche bayésienne** (alternative) → [[bayesian_ab_testing]]

4. **Métriques multiples** :
   - Correction de Bonferroni → [[multiple_testing]]
   - Ou FDR → [[multiple_testing]]

5. **Interprétation** :
   - Intervalles de confiance (pas que p-value !)
   - Significativité pratique vs statistique

**Cours dédié** : [[ab_testing]] ⭐⭐🔥💡

---

### 📊 "Je veux segmenter mes clients/données"

**Méthodes de clustering** :

1. **K-means** (rapide, sphérique) → [[kmeans_statistical]]
2. **Clustering hiérarchique** (dendrogramme) → [[hierarchical_clustering]]
3. **GMM** (probabiliste, ellipsoïdal) → [[gaussian_mixture_models]]
4. **DBSCAN** (formes arbitraires, détecte outliers)

**Choisir nombre de clusters** :
- Elbow method (inertie)
- Silhouette score
- BIC pour GMM → [[model_selection]]

**Validation** :
- Silhouette
- Davies-Bouldin
- Calinski-Harabasz

---

### ⏱️ "J'analyse des données temporelles"

**Workflow** :

1. **Visualisation** : Line plot, décomposition
2. **Test de stationnarité** : Dickey-Fuller → [[stationarity_tests]]
3. **Si non stationnaire** : Différenciation, transformation
4. **Identifier modèle** :
   - ACF, PACF → [[time_series_basics]]
   - Choisir ordres AR, MA
5. **Modéliser** :
   - ARIMA → [[arima_models]]
   - SARIMA si saisonnalité → [[sarima_models]]
6. **Diagnostics** :
   - Résidus = bruit blanc ?
   - Ljung-Box test
7. **Prédiction** : Intervalles de confiance

---

### 🔍 "Je cherche des relations de causalité"

**Attention** : Corrélation ≠ Causalité !

**Méthodes** :

1. **Expérimentation** : A/B test randomisé (gold standard) → [[experimental_design]]
2. **Observationnel** :
   - Régression avec covariables
   - Propensity Score Matching → [[propensity_score]]
   - Instrumental Variables
   - Difference-in-Differences
   - Regression Discontinuity Design
3. **Granger Causality** (séries temporelles) → [[granger_causality]]
4. **Structural Equation Modeling (SEM)**

**Cours dédié** : [[causal_inference]] 🎓💡

---

### 📉 "J'ai des valeurs aberrantes (outliers)"

**Stratégies** :

1. **Détection** :
   - Box plot (1.5 × IQR)
   - Z-score (> 3 ou < -3)
   - Isolation Forest
   - Local Outlier Factor (LOF)

2. **Décision** :
   - **Erreur de mesure** → Corriger ou supprimer
   - **Vrai outlier** → Garder mais analyser séparément
   - **Cas rare intéressant** → Étude de cas

3. **Méthodes robustes** :
   - Tests non paramétriques → [[nonparametric_tests]]
   - Régression robuste (Huber, LAD)
   - Médiane au lieu de moyenne
   - MAD (Median Absolute Deviation) au lieu d'écart-type

---

### 📊 "Je dois choisir entre plusieurs modèles"

**Critères** :

1. **Performance prédictive** :
   - Cross-validation → [[cross_validation]]
   - Holdout set
   - MSE, RMSE, MAE (régression)
   - Accuracy, F1, AUC (classification)

2. **Complexité** :
   - AIC : $$2k - 2\ln(L)$$ → [[model_selection]]
   - BIC : $$k\ln(n) - 2\ln(L)$$ → [[model_selection]]
   - BIC pénalise plus la complexité

3. **Interprétabilité** :
   - Modèles simples préférés si performance similaire

4. **Biais-variance** :
   - Underfitting vs overfitting → [[bias_variance_tradeoff]]

---

### 🎲 "Je veux quantifier l'incertitude de mes prédictions"

**Approches** :

1. **Intervalles de confiance** :
   - Régression linéaire : IC sur moyenne ou prédiction → [[linear_regression_statistical]]
   - Bootstrap : IC non paramétrique → [[bootstrap_methods]]

2. **Intervalles de prédiction** :
   - Plus larges que IC (incluent incertitude individuelle)

3. **Approche bayésienne** :
   - Posterior predictive distribution → [[bayesian_inference]]
   - Quantifie incertitude naturellement

4. **Ensembles** :
   - Variance entre modèles (Random Forest, Bootstrap)

---

## 🔧 PARTIE 3 : Référence Alphabétique des Outils

### A

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| A/B Testing | Expérimentation | [[ab_testing]] | ⭐🔥💡 |
| ACF (Autocorrélation) | Séries temporelles | [[time_series_basics]] | 🔥💡 |
| AIC (Akaike) | Sélection modèles | [[model_selection]] | ⭐🔥💡 |
| ANCOVA | ANOVA | [[ancova]] | 🎓💡 |
| ANOVA | Comparaisons multiples | [[anova_one_way]] | ⭐🔥💡 |
| ARIMA | Séries temporelles | [[arima_models]] | ⭐🔥💡 |

### B

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Bayes (Théorème) | Probabilité | [[bayesian_foundations]] | ⭐🔥💡 |
| Bernoulli (Loi) | Distributions | [[common_distributions]] | ⭐🔥💡 |
| Biais-Variance Tradeoff | ML | [[bias_variance_tradeoff]] | ⭐⭐🔥💡 |
| BIC (Bayesian) | Sélection modèles | [[model_selection]] | 🔥💡 |
| Binomiale (Loi) | Distributions | [[common_distributions]] | ⭐🔥💡 |
| Bonferroni (Correction) | Tests multiples | [[multiple_testing]] | ⭐🔥💡 |
| Bootstrap | Rééchantillonnage | [[bootstrap_methods]] | ⭐🔥💡 |
| Box plot | Visualisation | [[data_visualization_principles]] | ⭐🔥💡 |

### C

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Chi-deux (Test) | Tests d'hypothèses | [[chi_square_tests]] | ⭐🔥💡 |
| Chi-deux (Distribution) | Distributions | [[chi_square_distribution]] | ⭐🔥💡 |
| Coefficient de variation | Descriptives | [[measures_dispersion]] | 🔥 |
| Confiance (Intervalle) | Inférence | [[confidence_intervals]] | ⭐🔥💡 |
| Cook (Distance) | Régression diagnostics | [[regression_diagnostics]] | 🔥💡 |
| Corrélation de Pearson | Multivariée | [[correlation_covariance]] | ⭐🔥💡 |
| Corrélation de Spearman | Non paramétrique | [[correlation_tests]] | 🔥💡 |
| Covariance | Multivariée | [[correlation_covariance]] | ⭐🔥 |
| Cox (Régression) | Survie | [[cox_regression]] | 🔥💡 |
| Cross-validation | Validation | [[cross_validation]] | ⭐🔥💡 |

### D-E

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Dickey-Fuller (Test) | Séries temporelles | [[stationarity_tests]] | 🔥💡 |
| Écart-type | Descriptives | [[measures_dispersion]] | ⭐🔥 |
| Espérance | Probabilité | [[random_variables]] | ⭐🔥 |
| Exponentielle (Loi) | Distributions | [[common_distributions]] | 🔥💡 |

### F

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Fisher (Test exact) | Tests d'hypothèses | [[fisher_exact_test]] | 🔥💡 |
| Fisher (Test F) | Tests de variance | [[variance_tests]] | 🔥💡 |
| Fisher (Distribution F) | Distributions | [[fisher_distribution]] | 🔥💡 |

### G

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Gamma (Loi) | Distributions | [[common_distributions]] | 🎓 |
| Gaussienne (Loi) | Distributions | [[gaussian_distribution]] | ⭐⭐🔥💡 |
| Gaussian Mixture Models | Clustering | [[gaussian_mixture_models]] | ⭐🔥💡 |
| Granger (Causalité) | Séries temporelles | [[granger_causality]] | 🎓💡 |

### H-K

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Histogramme | Visualisation | [[data_visualization_principles]] | ⭐🔥💡 |
| Kaplan-Meier | Survie | [[kaplan_meier]] | 🔥💡 |
| K-means | Clustering | [[kmeans_statistical]] | ⭐🔥💡 |
| Kolmogorov-Smirnov (Test) | Non paramétrique | [[nonparametric_tests]] | 🔥💡 |
| Kruskal-Wallis (Test) | Non paramétrique | [[nonparametric_tests]] | 🔥💡 |
| Kurtosis | Descriptives | [[distribution_analysis]] | 🔥 |

### L

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Lasso (L1) | Régularisation | [[regularization_statistical]] | ⭐🔥💡 |
| Levene (Test) | Tests de variance | [[variance_tests]] | 🔥💡 |
| Loi des Grands Nombres | Théorèmes limites | [[limit_theorems]] | ⭐📐 |
| Log-Normale (Loi) | Distributions | [[common_distributions]] | 🔥💡 |
| Logistique (Régression) | Régression | [[logistic_regression]] | ⭐🔥💡 |

### M

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Mann-Whitney (Test U) | Non paramétrique | [[nonparametric_tests]] | ⭐🔥💡 |
| MANOVA | ANOVA multivariée | [[manova]] | 🎓💡 |
| Maximum de Vraisemblance | Estimation | [[maximum_likelihood]] | ⭐⭐🔥📐 |
| MCMC | Bayésien | [[mcmc_methods]] | 🎓📐💡 |
| Médiane | Descriptives | [[measures_central_tendency]] | ⭐🔥 |
| Moindres Carrés (OLS) | Régression | [[linear_regression_statistical]] | ⭐🔥💡 |
| Moyenne | Descriptives | [[measures_central_tendency]] | ⭐🔥 |

### N-O

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Normale (Loi) | Distributions | [[gaussian_distribution]] | ⭐⭐🔥💡 |
| Odds Ratio | Régression logistique | [[logistic_regression]] | 🔥💡 |

### P

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| p-value | Tests d'hypothèses | [[hypothesis_testing]] | ⭐⭐🔥💡 |
| PACF | Séries temporelles | [[time_series_basics]] | 🔥💡 |
| PCA | Multivariée | [[pca_statistical_perspective]] | ⭐🔥💡 |
| Poisson (Loi) | Distributions | [[common_distributions]] | ⭐🔥💡 |
| Poisson (Régression) | Régression | [[glm_models]] | 🔥💡 |
| Propensity Score | Causalité | [[propensity_score]] | 🎓💡 |
| Puissance (Analyse) | Expérimentation | [[power_analysis]] | ⭐🔥💡 |

### Q-R

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| QQ-plot | Diagnostics | [[regression_diagnostics]] | 🔥💡 |
| Quantiles | Descriptives | [[measures_dispersion]] | ⭐🔥 |
| R² (Coefficient détermination) | Régression | [[regression_diagnostics]] | ⭐🔥💡 |
| Régression Linéaire | Régression | [[linear_regression_statistical]] | ⭐🔥💡 |
| Régression Logistique | Régression | [[logistic_regression]] | ⭐🔥💡 |
| Ridge (L2) | Régularisation | [[regularization_statistical]] | ⭐🔥💡 |

### S

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Sample Size (Calcul) | Expérimentation | [[sample_size_calculation]] | ⭐🔥💡 |
| SARIMA | Séries temporelles | [[sarima_models]] | 🔥💡 |
| Shapiro-Wilk (Test) | Normalité | [[normality_tests]] | ⭐🔥💡 |
| Skewness | Descriptives | [[distribution_analysis]] | 🔥 |
| Spearman (Corrélation) | Non paramétrique | [[correlation_tests]] | 🔥💡 |
| Student (Distribution t) | Distributions | [[student_distribution]] | ⭐🔥💡 |
| Student (Test t) | Tests d'hypothèses | [[hypothesis_testing]] | ⭐🔥💡 |

### T-U-V

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Tau de Kendall | Corrélation | [[correlation_tests]] | 🔥 |
| Théorème Central Limite | Théorèmes limites | [[limit_theorems]] | ⭐⭐🔥📐 |
| Uniforme (Loi) | Distributions | [[common_distributions]] | ⭐🔥 |
| Variance | Descriptives | [[measures_dispersion]] | ⭐🔥 |
| VIF (Variance Inflation Factor) | Régression diagnostics | [[regression_diagnostics]] | 🔥💡 |

### W-Z

| Outil | Catégorie | Cours | Priorité |
|-------|-----------|-------|----------|
| Weibull (Loi) | Distributions | [[common_distributions]] | 🎓💡 |
| Wilcoxon (Test) | Non paramétrique | [[nonparametric_tests]] | 🔥💡 |
| Z-score | Normalisation | [[standardization]] | ⭐🔥 |
| Z-test | Tests d'hypothèses | [[hypothesis_testing]] | ⭐🔥 |

---

## 🗺️ PARTIE 4 : Parcours d'Apprentissage Recommandés

### 🎯 Parcours 1 : Foundations Essentielles (4-6 semaines)

**Pour qui** : Débuter ou consolider bases

1. [[measures_central_tendency]] - Statistiques descriptives 1
2. [[measures_dispersion]] - Statistiques descriptives 2
3. [[data_visualization_principles]] - Visualiser données
4. [[probability_foundations]] - Bases probabilités
5. [[common_distributions]] - Distributions essentielles (Bernoulli, Binomiale, Normale)
6. [[gaussian_distribution]] - Loi Normale en profondeur
7. [[limit_theorems]] - TCL et Loi des Grands Nombres
8. [[sampling_theory]] - Théorie échantillonnage
9. [[confidence_intervals]] - Intervalles de confiance
10. [[hypothesis_testing]] - Tests d'hypothèses (fondations)

---

### 🔥 Parcours 2 : Data Scientist Pratique (6-8 semaines)

**Pour qui** : Appliquer statistiques en entreprise

1. **Bases** : Parcours 1 (accéléré si déjà vu)
2. [[sample_size_calculation]] - Planifier études
3. [[ab_testing]] - A/B testing rigoureux
4. [[two_sample_tests]] - Comparer groupes
5. [[proportion_tests]] - Tests de proportions
6. [[anova_one_way]] - Comparer >2 groupes
7. [[chi_square_tests]] - Variables catégorielles
8. [[linear_regression_statistical]] - Régression linéaire
9. [[logistic_regression]] - Classification
10. [[regression_diagnostics]] - Valider modèles
11. [[correlation_covariance]] - Relations entre variables
12. [[nonparametric_tests]] - Alternatives robustes
13. [[bootstrap_methods]] - Rééchantillonnage
14. [[cross_validation]] - Validation modèles

---

### 🎓 Parcours 3 : Ingénieur ML Avancé (8-12 semaines)

**Pour qui** : Fondements théoriques du ML

1. **Bases** : Parcours 2
2. [[maximum_likelihood]] - MLE (fondamental)
3. [[bias_variance_tradeoff]] - Cœur du ML
4. [[regularization_statistical]] - Ridge, Lasso
5. [[model_selection]] - AIC, BIC
6. [[bayesian_foundations]] - Statistiques bayésiennes
7. [[bayesian_estimation]] - MAP, posterior
8. [[pca_statistical_perspective]] - PCA théorique
9. [[gaussian_mixture_models]] - GMM
10. [[kmeans_statistical]] - K-means rigoureux
11. [[causal_inference]] - Corrélation ≠ causalité
12. [[concentration_inequalities]] - Théorie généralisation
13. [[pac_learning]] - Théorie apprentissage

---

### 💼 Parcours 4 : Data Analyst Business (4-6 semaines)

**Pour qui** : Analyser données et communiquer résultats

1. [[measures_central_tendency]] - Statistiques descriptives
2. [[measures_dispersion]] - Variabilité
3. [[data_visualization_principles]] - Graphiques efficaces
4. [[confidence_intervals]] - Communiquer incertitude
5. [[hypothesis_testing]] - Prendre décisions
6. [[two_sample_tests]] - Comparer segments
7. [[proportion_tests]] - Taux de conversion
8. [[chi_square_tests]] - Relations catégorielles
9. [[ab_testing]] - Tests produit
10. [[linear_regression_statistical]] - Prédictions simples
11. [[sample_size_calculation]] - Planifier collecte
12. [[multiple_testing]] - Éviter faux positifs

---

### ⏱️ Parcours 5 : Séries Temporelles (3-4 semaines)

**Pour qui** : Données temporelles, prévisions

1. [[time_series_basics]] - ACF, PACF
2. [[stationarity]] - Stationnarité
3. [[stationarity_tests]] - Dickey-Fuller
4. [[time_series_decomposition]] - Trend, saisonnalité
5. [[arima_models]] - ARIMA
6. [[sarima_models]] - Saisonnalité
7. [[granger_causality]] - Causalité temporelle
8. [[volatility_models]] - ARCH/GARCH (finance)

---

## 🚀 Comment Utiliser Cet Index au Quotidien

### Scénario 1 : Nouveau Problème

1. **Identifiez votre question** dans la Partie 2 (Navigation par Question)
2. **Suivez l'arbre de décision** pour trouver l'outil approprié
3. **Consultez le cours dédié** (liens [[course_name]])
4. **Appliquez** avec code et exemples du cours

### Scénario 2 : Approfondir un Sujet

1. **Trouvez le cours** dans la Partie 1 (Navigation par Domaine)
2. **Vérifiez prérequis** listés dans le cours
3. **Suivez liens connexes** (principe Zettelkasten)
4. **Pratiquez** avec exercices et projets

### Scénario 3 : Révision Rapide

1. **Partie 3** (Référence Alphabétique) pour trouver rapidement
2. **Section "Résumé Rapide"** de chaque cours
3. **Code minimal** pour usage immédiat

---

## 📚 Ressources Complémentaires Transversales

### Livres Fondamentaux

1. **"All of Statistics"** - Larry Wasserman
   - [URL](http://www.stat.cmu.edu/~larry/all-of-statistics/)
   - 📌 **Pourquoi** : Complet, mathématique, lien ML
   - 🎯 **Niveau** : Intermédiaire-Avancé

2. **"Statistical Inference"** - Casella & Berger
   - 📌 **Pourquoi** : Bible de l'inférence statistique
   - 🎯 **Niveau** : Avancé

3. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - [URL](https://hastie.su.domains/ElemStatLearn/)
   - 📌 **Pourquoi** : Pont statistiques ↔ ML
   - 🎯 **Niveau** : Avancé

4. **"Practical Statistics for Data Scientists"** - Bruce & Bruce
   - 📌 **Pourquoi** : Approche pratique, code R/Python
   - 🎯 **Niveau** : Débutant-Intermédiaire

### Cours en Ligne

1. **StatQuest (YouTube)** - Josh Starmer
   - [URL](https://www.youtube.com/c/joshstarmer)
   - 📌 **Pourquoi** : Visualisations exceptionnelles
   - ⏱️ **Format** : Vidéos 5-15 min

2. **Seeing Theory** - Brown University
   - [URL](https://seeing-theory.brown.edu/)
   - 📌 **Pourquoi** : Visualisations interactives probabilités
   - 🎯 **Niveau** : Tous niveaux

### Outils et Bibliothèques

**Python** :
- `scipy.stats` : Distributions, tests statistiques
- `statsmodels` : Régression, séries temporelles
- `pingouin` : Tests statistiques faciles
- `pymc` : Statistiques bayésiennes

**R** (référence en statistiques) :
- Base R : Tests statistiques intégrés
- `tidyverse` : Manipulation et visualisation
- `lme4` : Modèles mixtes
- `survival` : Analyse de survie

---

## 🔄 Maintenance de Cet Index

Cet index sera mis à jour à chaque ajout de cours avec :

- ✅ **Liens actifs** vers nouveaux cours créés
- 📅 **Date de dernière mise à jour** en haut du document
- 🆕 **Nouveaux outils** intégrés dans les 3 parties
- 🔗 **Liens bidirectionnels** (principe Zettelkasten)

---

## 📝 Notes de Révision Rapide

### Top 10 des Outils Absolument Essentiels

1. **Théorème Central Limite** - Justifie presque tout en statistiques
2. **Tests t de Student** - Comparer moyennes (usage quotidien)
3. **Loi Normale** - Distribution par défaut
4. **Intervalles de Confiance** - Quantifier incertitude
5. **p-value** - Prise de décision (avec précautions)
6. **Régression Linéaire** - Modélisation de base
7. **Maximum de Vraisemblance** - Estimation universelle
8. **A/B Testing** - Validation rigoureuse
9. **Biais-Variance Tradeoff** - Cœur du ML
10. **Bootstrap** - Méthode non paramétrique puissante

### Erreurs Statistiques les Plus Fréquentes à Éviter

1. ❌ **p-hacking** : Tester jusqu'à trouver p < 0.05
2. ❌ **Peeking** : Regarder résultats A/B test avant la fin
3. ❌ **Pas de correction tests multiples** : Taux erreur familial explosé
4. ❌ **Confondre corrélation et causalité**
5. ❌ **Ignorer hypothèses des tests** : Normalité, indépendance, etc.
6. ❌ **Interpréter p-value comme P(H₀|données)** : C'est l'inverse !
7. ❌ **Oublier puissance statistique** : Risque de faux négatifs
8. ❌ **Biais de sélection** : Échantillon non représentatif
9. ❌ **Surinterprétation p-value** : Regarder aussi IC et effet size
10. ❌ **Oublier coût d'échantillonnage** : Balance précision vs coût

---

## 🎯 Prochains Cours à Créer (Feuille de Route)

### Priorité 1 (Fondations) ⭐⭐

- [ ] [[probability_foundations]]
- [ ] [[common_distributions]]
- [ ] [[gaussian_distribution]]
- [ ] [[sampling_theory]]
- [ ] [[confidence_intervals]]
- [ ] [[hypothesis_testing]]

### Priorité 2 (Pratique Data Science) ⭐

- [ ] [[sample_size_calculation]]
- [ ] [[ab_testing]]
- [ ] [[two_sample_tests]]
- [ ] [[linear_regression_statistical]]
- [ ] [[logistic_regression]]
- [ ] [[regression_diagnostics]]

### Priorité 3 (Avancé ML) 🎓

- [ ] [[maximum_likelihood]]
- [ ] [[bias_variance_tradeoff]]
- [ ] [[bayesian_foundations]]
- [ ] [[gaussian_mixture_models]]
- [ ] [[causal_inference]]

---

## 🏁 Conclusion

Cet index est votre **boussole statistique**. Marquez cette page dans vos favoris et revenez-y chaque fois que vous :

- 🤔 Êtes bloqué sur un problème statistique
- 🔍 Cherchez quel outil utiliser
- 📚 Voulez approfondir un sujet
- 🔗 Naviguez entre cours connexes

**Principe Zettelkasten** : Chaque cours créé aura :
- ✅ Liens vers cet index
- ✅ Liens vers cours prérequis
- ✅ Liens vers cours suivants
- ✅ Mise à jour de cet index avec lien actif

Bon apprentissage ! 🚀

---

**Dernière mise à jour** : 2026-03-18  
**Nombre de cours planifiés** : ~80  
**Nombre de cours créés** : 0 (nous commençons !)  
**Prochaine étape** : Premier cours des fondations
