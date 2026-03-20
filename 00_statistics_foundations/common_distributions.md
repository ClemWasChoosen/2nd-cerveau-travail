# 📊 Distributions de Probabilité Classiques

> **Résumé en une phrase** : Les distributions de probabilité classiques (Bernoulli, Binomiale, Poisson, Normale, Exponentielle) sont des modèles mathématiques universels qui capturent les patterns statistiques récurrents dans la nature, les sciences et les données.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-03-20 |
| **Dernière mise à jour** | 2026-03-20 |
| **Domaine** | Théorie des Probabilités |
| **Niveau** | Intermédiaire |
| **Durée de lecture** | ~60 minutes |
| **Fichier** | `common_distributions.md` |
| **Emplacement** | `/00_statistics_foundations/02_probability_theory/` |
| **Tags** | `#probability` `#distributions` `#bernoulli` `#binomial` `#poisson` `#normal` `#exponential` `#uniform` |

### Prérequis

- [x] [[probability_foundations]] - Axiomes, probabilités (ESSENTIEL)
- [x] [[random_variables]] - VA, PMF, PDF, espérance, variance (ESSENTIEL)
- [ ] Calcul différentiel et intégral (pour dérivations)
- [ ] Combinatoire de base (arrangements, combinaisons)

### Cours connexes (Liens Zettelkasten)

- **Prérequis** : 
  - [[probability_foundations]] - Fondements probabilités
  - [[random_variables]] - Variables aléatoires
- **Complémentaires** : 
  - [[moment_generating_functions]] - Fonctions génératrices
  - [[data_visualization_principles]] - Visualiser distributions
- **Suite recommandée** : 
  - [[central_limit_theorem]] - Théorème central limite
  - [[hypothesis_testing]] - Tests statistiques
  - [[maximum_likelihood]] - Estimation paramètres
  - [[bayesian_inference]] - Inférence bayésienne

---

## 🎯 Vue d'ensemble

### Qu'allez-vous apprendre ?

Plutôt que de modéliser chaque phénomène aléatoire from scratch, les statisticiens ont identifié des **familles universelles de distributions** qui apparaissent encore et encore dans la nature. Ce cours vous enseigne les **distributions fondamentales** : leurs **dérivations mathématiques**, leurs **propriétés théoriques**, leurs **relations** entre elles, et surtout **pourquoi** elles émergent naturellement dans différents contextes. Vous comprendrez les fondements mathématiques profonds qui font de la Normale la "reine des distributions" et du Processus de Poisson le modèle universel des événements rares.

### Objectifs d'apprentissage (Taxonomie de Bloom)

À la fin de ce cours, vous serez capable de :

1. **Comprendre** : Dériver mathématiquement PMF/PDF des distributions classiques
2. **Appliquer** : Calculer probabilités, espérances, variances pour chaque distribution
3. **Analyser** : Identifier quelle distribution modélise un phénomène donné
4. **Évaluer** : Comparer distributions (Binomiale vs Poisson, Normale vs Log-Normale)
5. **Créer** : Construire modèles probabilistes complexes par composition
6. **Synthétiser** : Comprendre liens entre distributions (approximations, cas limites)

---

## 🔍 Contexte et Motivation

### Pourquoi ce sujet est-il important ?

**Les distributions de probabilité sont les "briques LEGO" de la modélisation statistique.**

Sans elles, impossible de :
- **Tester des hypothèses** : Tous les tests (t-test, chi², ANOVA) reposent sur des distributions
- **Estimer l'incertitude** : Intervalles de confiance nécessitent distribution échantillonnage
- **Machine Learning probabiliste** : Naive Bayes, Régression logistique, VAE, Diffusion models
- **Modéliser processus réels** : Files d'attente (Poisson), durées de vie (Exponentielle), erreurs de mesure (Normale)

**Exemple fondamental** : Pourquoi la distribution Normale apparaît-elle PARTOUT ?

**Réponse** : Le **Théorème Central Limite** (TCL) garantit que la **somme de nombreuses VA indépendantes** converge vers une Normale, quelle que soit leur distribution individuelle.

**Conséquences pratiques** :
- **Tailles humaines** : Somme de milliers de facteurs génétiques → Normale
- **Erreurs de mesure** : Somme de nombreuses petites perturbations → Normale
- **Rendements financiers** : Agrégation de multiples chocs → Approximativement Normale
- **Scores de tests** : Somme de réponses à multiples questions → Normale

**C'est pourquoi 70% des méthodes statistiques supposent normalité !**

### Quel problème résout-il ?

**Problème** : Vous gérez un call center. Combien de lignes téléphoniques prévoir pour que moins de 1% des appels soient en attente ?

**Sans modèle probabiliste** : Impossible de répondre rationnellement.

**Avec Processus de Poisson** :
- **Hypothèse** : Appels arrivent aléatoirement, indépendamment, à taux constant λ (ex: 20 appels/heure)
- **Modèle** : Nombre d'appels en 1h suit $$X \sim \text{Poisson}(\lambda = 20)$$
- **Calcul** : $$P(X \leq k) = \sum_{i=0}^{k} \frac{e^{-\lambda} \lambda^i}{i!}$$
- **Réponse** : Trouver $$k$$ tel que $$P(X \leq k) \geq 0.99$$ → Dimensionner $$k+1$$ lignes

**La distribution de Poisson transforme un problème opérationnel complexe en calcul mathématique élémentaire.**

### Applications dans le monde réel

1. **Bernoulli / Binomiale** :
   - A/B testing (conversion oui/non)
   - Contrôle qualité (pièce défectueuse oui/non)
   - Médecine (patient guéri oui/non)

2. **Poisson** :
   - Télécommunications (nombre de paquets réseau)
   - Assurance (nombre de sinistres)
   - Radioactivité (désintégrations atomiques)

3. **Normale (Gaussienne)** :
   - Erreurs de mesure
   - Machine Learning (prior bayésien, régularisation)
   - Finance (rendements, Black-Scholes)

4. **Exponentielle** :
   - Durées de vie composants électroniques
   - Temps d'attente (file d'attente)
   - Délai entre événements Poisson

---

## 📚 Fondamentaux Théoriques

> **Navigation cognitive** : Nous procédons du **simple au complexe** : Bernoulli (1 épreuve) → Binomiale (n épreuves) → Poisson (limite) → Normale (somme infinie) → Exponentielle (temps continu).

---

## PARTIE I : DISTRIBUTIONS DISCRÈTES

### 1. Distribution de Bernoulli

#### 1.1 Définition et Motivation

**Contexte** : Expérience avec **2 résultats possibles** (succès/échec).

**Exemples** :
- Lancer pièce : Pile (1) ou Face (0)
- Test médical : Positif (1) ou Négatif (0)
- Clic sur pub : Oui (1) ou Non (0)

**Variable aléatoire de Bernoulli** : $$X \sim \text{Bernoulli}(p)$$

$$X = \begin{cases} 1 & \text{avec probabilité } p \\ 0 & \text{avec probabilité } 1-p \end{cases}$$

où $$p \in [0, 1]$$ est le **paramètre de succès**.

#### 1.2 PMF (Fonction de Masse)

**Formulation classique** :

$$P(X = k) = \begin{cases} p & \text{si } k=1 \\ 1-p & \text{si } k=0 \\ 0 & \text{sinon} \end{cases}$$

**Formulation compacte** (élégante) :

$$P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}$$

**Vérification** : 
- $$k=1$$ : $$P(X=1) = p^1 (1-p)^{1-1} = p^1 (1-p)^0 = p$$
- $$k=0$$ : $$P(X=0) = p^0 (1-p)^{1-0} = (1-p)$$
- Somme : $$p + (1-p) = 1$$ ✓

#### 1.3 Espérance et Variance

**Espérance** :

$$E[X] = \sum_{k=0}^{1} k \cdot P(X=k) = 0 \cdot (1-p) + 1 \cdot p = p$$

**Interprétation** : En moyenne, on obtient $$p$$ succès. Si $$p=0.3$$, on "espère" 0.3 succès (= 30% de chance).

**Variance** :

Méthode 1 (définition) :
$$\text{Var}(X) = E[(X - E[X])^2] = E[(X - p)^2]$$

$$= (0-p)^2 \cdot (1-p) + (1-p)^2 \cdot p$$

$$= p^2(1-p) + (1-p)^2 p = p(1-p)[p + (1-p)] = p(1-p)$$

Méthode 2 (formule computationnelle) :
$$E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p$$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$$

**Propriété remarquable** : Variance maximale quand $$p = 0.5$$ (incertitude maximale).

**Démonstration** : 
$$\frac{d}{dp}[p(1-p)] = 1 - 2p = 0 \Rightarrow p = \frac{1}{2}$$

$$\text{Var}(X)|_{p=0.5} = 0.5 \times 0.5 = 0.25$$

---

### 2. Distribution Binomiale

#### 2.1 Dérivation Mathématique

**Contexte** : Répéter $$n$$ fois **indépendamment** une épreuve de Bernoulli de paramètre $$p$$.

**Variable aléatoire** : $$X$$ = nombre de succès sur $$n$$ épreuves.

$$X \sim \text{Binomiale}(n, p)$$

**Question** : Quelle est $$P(X = k)$$ ?

**Raisonnement combinatoire** :

1. **Choisir** quelles $$k$$ épreuves parmi $$n$$ sont des succès : $$\binom{n}{k}$$ façons

2. **Probabilité d'une séquence spécifique** avec $$k$$ succès et $$n-k$$ échecs :
   - Succès (×k) : probabilité $$p$$ chacun → $$p^k$$
   - Échecs (×(n-k)) : probabilité $$(1-p)$$ chacun → $$(1-p)^{n-k}$$
   - Probabilité totale d'une séquence : $$p^k (1-p)^{n-k}$$

3. **Somme sur toutes les séquences** :

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k \in \{0, 1, \ldots, n\}$$

où $$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$ est le coefficient binomial.

#### 2.2 Vérification de la Normalisation

**Propriété** : $$\sum_{k=0}^{n} P(X=k) = 1$$

**Démonstration** :

$$\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

**Formule du binôme de Newton** : $$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}$$

En posant $$a = p$$ et $$b = 1-p$$ :

$$\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1^n = 1$$ ✓

**Élégance mathématique** : La PMF binomiale découle directement du binôme de Newton !

#### 2.3 Espérance et Variance

**Espérance** :

**Méthode directe** (laborieuse) :

$$E[X] = \sum_{k=0}^{n} k \binom{n}{k} p^k (1-p)^{n-k}$$

**Méthode élégante** (linéarité) :

Décomposer $$X$$ en somme de $$n$$ Bernoulli indépendantes :

$$X = X_1 + X_2 + \cdots + X_n$$

où $$X_i \sim \text{Bernoulli}(p)$$ (1 si i-ème épreuve = succès, 0 sinon).

Par **linéarité de l'espérance** :

$$E[X] = E[X_1 + \cdots + X_n] = E[X_1] + \cdots + E[X_n] = p + p + \cdots + p = np$$

**Résultat** : $$E[X] = np$$

**Interprétation** : Sur $$n$$ épreuves avec probabilité $$p$$, on "espère" $$np$$ succès en moyenne.

**Variance** :

$$X = X_1 + \cdots + X_n$$ où $$X_i$$ indépendantes.

Propriété : $$\text{Var}(X_1 + \cdots + X_n) = \text{Var}(X_1) + \cdots + \text{Var}(X_n)$$ (si indépendantes)

$$\text{Var}(X) = n \times \text{Var}(X_1) = n \times p(1-p) = np(1-p)$$

**Écart-type** : $$\sigma(X) = \sqrt{np(1-p)}$$

#### 2.4 Cas Particuliers et Propriétés

**Cas** $$n=1$$ : $$\text{Binomiale}(1, p) = \text{Bernoulli}(p)$$

**Cas** $$p=0.5$$ : Distribution symétrique (maximum à $$k = n/2$$)

**Somme de Binomiales** : Si $$X \sim \text{Bin}(n_1, p)$$ et $$Y \sim \text{Bin}(n_2, p)$$ indépendantes avec **même** $$p$$ :

$$X + Y \sim \text{Binomiale}(n_1 + n_2, p)$$

**Pourquoi ?** Additionner succès de $$n_1$$ épreuves et $$n_2$$ épreuves = $$n_1+n_2$$ épreuves totales.

---

### 3. Distribution de Poisson

#### 3.1 Motivation et Contexte Historique

**Siméon Denis Poisson (1837)** introduit cette distribution pour modéliser **événements rares** sur intervalle de temps/espace.

**Exemples canoniques** :
- Nombre d'appels téléphoniques par heure
- Nombre de fautes de frappe par page
- Nombre de particules radioactives détectées par seconde
- Nombre de buts marqués dans un match de football
- Nombre de clients arrivant dans un magasin par minute

**Caractéristiques communes** :
1. Événements se produisent **aléatoirement** dans le temps/espace
2. **Indépendance** : Un événement n'influence pas les autres
3. **Taux constant** : Probabilité d'événement dans petit intervalle ≈ constante
4. **Événements rares** : Probabilité dans intervalle infinitésimal → 0

#### 3.2 Dérivation à partir de la Binomiale (Loi des Petits Nombres)

**Scénario** : Diviser un intervalle de temps $$t$$ en $$n$$ sous-intervalles très petits. Dans chaque sous-intervalle, événement se produit avec probabilité $$p$$.

**Taux moyen** : $$\lambda = np$$ (nombre moyen d'événements)

**Nombre d'événements** : $$X \sim \text{Binomiale}(n, p)$$

**Limite de Poisson** : Quand $$n \to \infty$$ et $$p \to 0$$ tels que $$np = \lambda$$ reste constant :

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Substituer $$p = \frac{\lambda}{n}$$ :

$$P(X = k) = \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

$$= \frac{n!}{k!(n-k)!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

$$= \frac{n(n-1)\cdots(n-k+1)}{n^k} \cdot \frac{\lambda^k}{k!} \cdot \left(1 - \frac{\lambda}{n}\right)^{n} \cdot \left(1 - \frac{\lambda}{n}\right)^{-k}$$

**Passage à la limite** $$n \to \infty$$ :

1. $$\frac{n(n-1)\cdots(n-k+1)}{n^k} = \frac{n}{n} \cdot \frac{n-1}{n} \cdots \frac{n-k+1}{n} \to 1 \times 1 \times \cdots \times 1 = 1$$

2. $$\left(1 - \frac{\lambda}{n}\right)^{n} \to e^{-\lambda}$$ (définition de $$e$$)

3. $$\left(1 - \frac{\lambda}{n}\right)^{-k} \to 1^{-k} = 1$$

**Résultat** : $$P(X = k) \to \frac{\lambda^k}{k!} e^{-\lambda}$$

#### 3.3 PMF de Poisson

**Variable aléatoire** : $$X \sim \text{Poisson}(\lambda)$$ où $$\lambda > 0$$

**PMF** :

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \ldots\}$$

**Paramètre** : $$\lambda$$ = taux moyen (espérance ET variance)

**Vérification normalisation** :

$$\sum_{k=0}^{\infty} P(X=k) = \sum_{k=0}^{\infty} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{\lambda^k}{k!}$$

**Série exponentielle** : $$e^{\lambda} = \sum_{k=0}^{\infty} \frac{\lambda^k}{k!}$$

Donc : $$e^{-\lambda} \cdot e^{\lambda} = 1$$ ✓

#### 3.4 Espérance et Variance

**Espérance** :

$$E[X] = \sum_{k=0}^{\infty} k \cdot \frac{\lambda^k e^{-\lambda}}{k!}$$

$$= e^{-\lambda} \sum_{k=1}^{\infty} k \cdot \frac{\lambda^k}{k!}$$ (terme $$k=0$$ nul)

$$= e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^k}{(k-1)!}$$ (simplifier $$k/k!$$)

$$= e^{-\lambda} \lambda \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!}$$

Substitution $$j = k-1$$ :

$$= e^{-\lambda} \lambda \sum_{j=0}^{\infty} \frac{\lambda^{j}}{j!} = e^{-\lambda} \lambda e^{\lambda} = \lambda$$

**Résultat** : $$E[X] = \lambda$$

**Variance** (calcul similaire) :

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

On peut montrer que $$E[X^2] = \lambda^2 + \lambda$$

Donc : $$\text{Var}(X) = \lambda^2 + \lambda - \lambda^2 = \lambda$$

**Propriété remarquable** : Pour Poisson, **espérance = variance = λ** !

C'est une caractérisation : Si $$E[X] = \text{Var}(X)$$, indice fort que $$X \sim \text{Poisson}$$.

#### 3.5 Propriété Additive

**Somme de Poissons indépendants** :

Si $$X \sim \text{Poisson}(\lambda_1)$$ et $$Y \sim \text{Poisson}(\lambda_2)$$ indépendantes :

$$X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$$

**Intuition** : Fusionner 2 flux d'événements indépendants de taux $$\lambda_1$$ et $$\lambda_2$$ → flux combiné de taux $$\lambda_1 + \lambda_2$$.

**Démonstration** (convolution) : Omise ici mais utilise fonctions génératrices.

---

### 4. Distribution Géométrique

#### 4.1 Définition

**Contexte** : Répéter épreuves de Bernoulli(p) **jusqu'au premier succès**.

**Variable aléatoire** : $$X$$ = nombre d'épreuves nécessaires jusqu'au 1er succès.

$$X \sim \text{Géométrique}(p)$$

**PMF** :

$$P(X = k) = (1-p)^{k-1} p, \quad k \in \{1, 2, 3, \ldots\}$$

**Interprétation** : $$(k-1)$$ échecs puis 1 succès.

**Espérance** : $$E[X] = \frac{1}{p}$$

**Variance** : $$\text{Var}(X) = \frac{1-p}{p^2}$$

**Propriété de "perte de mémoire"** :

$$P(X > n+m | X > n) = P(X > m)$$

La distribution géométrique est la **seule distribution discrète** ayant cette propriété.

**Interprétation** : Si on a déjà attendu $$n$$ épreuves sans succès, la probabilité d'attendre encore $$m$$ épreuves est la même que si on recommençait à zéro.

---

## PARTIE II : DISTRIBUTIONS CONTINUES

### 5. Distribution Uniforme Continue

#### 5.1 Définition

**Contexte** : Toutes les valeurs dans un intervalle $$[a, b]$$ sont **également probables**.

$$X \sim \text{Uniforme}(a, b)$$

**PDF** :

$$f_X(x) = \begin{cases} \frac{1}{b-a} & \text{si } x \in [a, b] \\ 0 & \text{sinon} \end{cases}$$

**CDF** :

$$F_X(x) = \begin{cases} 0 & \text{si } x < a \\ \frac{x-a}{b-a} & \text{si } x \in [a, b] \\ 1 & \text{si } x > b \end{cases}$$

**Espérance** :

$$E[X] = \int_a^b x \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\Big|_a^b = \frac{1}{b-a} \cdot \frac{b^2 - a^2}{2} = \frac{a+b}{2}$$

**Point milieu** de l'intervalle (symétrie).

**Variance** :

$$\text{Var}(X) = \int_a^b (x - \mu)^2 \cdot \frac{1}{b-a} dx = \frac{(b-a)^2}{12}$$

**Usage** : Génération nombres pseudo-aléatoires, prior bayésien non informatif.

---

### 6. Distribution Exponentielle

#### 6.1 Motivation et Lien avec Poisson

**Contexte** : Si les événements suivent un **processus de Poisson** (taux $$\lambda$$), le **temps d'attente** jusqu'au prochain événement suit une loi **Exponentielle**.

**Variable aléatoire** : $$T$$ = temps jusqu'au prochain événement.

$$T \sim \text{Exponentielle}(\lambda)$$

où $$\lambda > 0$$ est le **taux** (événements par unité de temps).

#### 6.2 Dérivation de la PDF

**Question** : Quelle est $$P(T > t)$$ ?

$$T > t$$ signifie "aucun événement dans l'intervalle $$[0, t]$$".

Si événements suivent Poisson$$(\lambda t)$$ sur intervalle $$[0, t]$$ :

$$P(\text{0 événements en }[0,t]) = \frac{(\lambda t)^0 e^{-\lambda t}}{0!} = e^{-\lambda t}$$

Donc : $$P(T > t) = e^{-\lambda t}$$

**CDF** :

$$F_T(t) = P(T \leq t) = 1 - P(T > t) = 1 - e^{-\lambda t}, \quad t \geq 0$$

**PDF** :

$$f_T(t) = \frac{d}{dt} F_T(t) = \frac{d}{dt}(1 - e^{-\lambda t}) = \lambda e^{-\lambda t}, \quad t \geq 0$$

#### 6.3 Espérance et Variance

**Espérance** :

$$E[T] = \int_0^{\infty} t \cdot \lambda e^{-\lambda t} dt$$

**Intégration par parties** : $$u = t$$, $$dv = \lambda e^{-\lambda t} dt$$

$$du = dt$$, $$v = -e^{-\lambda t}$$

$$E[T] = \left[-t e^{-\lambda t}\right]_0^{\infty} + \int_0^{\infty} e^{-\lambda t} dt$$

$$= 0 + \left[-\frac{1}{\lambda} e^{-\lambda t}\right]_0^{\infty} = \frac{1}{\lambda}$$

**Résultat** : $$E[T] = \frac{1}{\lambda}$$

**Interprétation** : Temps moyen d'attente = inverse du taux.

**Exemple** : Si $$\lambda = 5$$ clients/heure, temps moyen entre clients = $$\frac{1}{5}$$ heure = 12 minutes.

**Variance** :

$$\text{Var}(T) = \frac{1}{\lambda^2}$$

**Écart-type** : $$\sigma(T) = \frac{1}{\lambda}$$

**Propriété remarquable** : Écart-type = Espérance !

#### 6.4 Propriété de Perte de Mémoire

**Propriété fondamentale** :

$$P(T > s+t | T > s) = P(T > t)$$

**Démonstration** :

$$P(T > s+t | T > s) = \frac{P(T > s+t \cap T > s)}{P(T > s)} = \frac{P(T > s+t)}{P(T > s)}$$

$$= \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t)$$

**Interprétation** : Le "futur" ne dépend pas du "passé". Si on a déjà attendu $$s$$ unités de temps sans événement, la distribution du temps d'attente **restant** est identique à la distribution initiale.

**Conséquence** : L'exponentielle est la **seule distribution continue** ayant cette propriété (analogue géométrique pour discret).

**Application** : Modélisation durées de vie **sans vieillissement** (composants électroniques jeunes).

---

### 7. Distribution Normale (Gaussienne)

#### 7.1 Importance Historique et Universalité

**Carl Friedrich Gauss (1809)** et **Pierre-Simon Laplace** développent cette distribution pour modéliser **erreurs de mesure astronomiques**.

**Pourquoi est-elle omniprésente ?**

**Théorème Central Limite** (TCL) : La **somme (ou moyenne) de nombreuses VA indépendantes** de distributions **quelconques** converge vers une distribution Normale.

**Conséquence** : Tout phénomène résultant de l'**agrégation de multiples facteurs aléatoires indépendants** suit approximativement une Normale.

**Exemples** :
- Taille humaine = somme de milliers de gènes + facteurs environnementaux
- Erreur de mesure = somme de multiples perturbations
- Score QI = performance sur multiples tâches cognitives
- Rendement boursier (sur court terme) = somme de multiples chocs économiques

#### 7.2 PDF de la Normale

**Variable aléatoire** : $$X \sim \mathcal{N}(\mu, \sigma^2)$$

**Paramètres** :
- $$\mu \in \mathbb{R}$$ : Espérance (centre)
- $$\sigma^2 > 0$$ : Variance (dispersion)

**PDF** :

$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

**Pourquoi cette forme ?**

**Dérivation intuitive** (Gauss) :

1. **Symétrie** : Erreurs positives et négatives également probables → $$f(x)$$ dépend de $$(x-\mu)^2$$

2. **Décroissance exponentielle** : Grandes erreurs très rares → forme $$\exp(-\alpha(x-\mu)^2)$$

3. **Normalisation** : $$\int_{-\infty}^{\infty} f(x)dx = 1$$ impose $$\alpha = \frac{1}{2\sigma^2}$$ et facteur $$\frac{1}{\sigma\sqrt{2\pi}}$$

**Vérification normalisation** (intégrale Gaussienne) :

$$\int_{-\infty}^{\infty} \exp\left(-\frac{x^2}{2}\right) dx = \sqrt{2\pi}$$

(Démonstration classique par passage en coordonnées polaires)

#### 7.3 Espérance et Variance

**Espérance** :

Par symétrie autour de $$\mu$$ :

$$E[X] = \int_{-\infty}^{\infty} x \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx = \mu$$

**Variance** :

$$\text{Var}(X) = E[(X-\mu)^2] = \int_{-\infty}^{\infty} (x-\mu)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx = \sigma^2$$

#### 7.4 Loi Normale Standard

**Standardisation** : Transformer toute Normale en Normale Standard.

Si $$X \sim \mathcal{N}(\mu, \sigma^2)$$, alors :

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

**Normale Standard** : $$Z \sim \mathcal{N}(0, 1)$$

**PDF** :

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

**CDF** :

$$\Phi(z) = \int_{-\infty}^{z} \phi(t) dt$$

**Pas de forme fermée** ! Calculée numériquement (tables historiquement).

#### 7.5 Règle Empirique 68-95-99.7

Pour $$X \sim \mathcal{N}(\mu, \sigma^2)$$ :

- $$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.68$$ (68%)
- $$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.95$$ (95%)
- $$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.997$$ (99.7%)

**Interprétation** : 
- ~68% des données dans $$[\mu - \sigma, \mu + \sigma]$$
- ~95% dans $$[\mu - 2\sigma, \mu + 2\sigma]$$
- ~99.7% dans $$[\mu - 3\sigma, \mu + 3\sigma]$$

**Application** : Détection outliers (valeurs > 3σ de la moyenne sont suspectes).

#### 7.6 Propriétés Fondamentales

**Linéarité** :

Si $$X \sim \mathcal{N}(\mu, \sigma^2)$$, alors :

$$aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$$

**Somme de Normales indépendantes** :

Si $$X \sim \mathcal{N}(\mu_1, \sigma_1^2)$$ et $$Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$$ **indépendantes** :

$$X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

**Stabilité** : Somme de Normales = Normale (famille **stable**).

---

### 8. Autres Distributions Importantes

#### 8.1 Distribution Log-Normale

**Définition** : Si $$\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$$, alors $$X \sim \text{Log-Normale}(\mu, \sigma^2)$$

**Équivalent** : $$X = e^Y$$ où $$Y \sim \mathcal{N}(\mu, \sigma^2)$$

**PDF** :

$$f_X(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

**Espérance** : $$E[X] = e^{\mu + \sigma^2/2}$$

**Variance** : $$\text{Var}(X) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$$

**Applications** :
- Prix d'actifs financiers (toujours positifs, asymétriques)
- Revenus (asymétrie droite)
- Taille de fichiers informatiques
- Particules en suspension (distribution de tailles)

**Pourquoi Log-Normale ?**

Si un processus **multiplicatif** (croissance composée) : $$X_t = X_0 \cdot r_1 \cdot r_2 \cdots r_n$$

Alors $$\ln(X_t) = \ln(X_0) + \ln(r_1) + \cdots + \ln(r_n)$$ → Somme → TCL → Normale !

Donc $$X_t$$ Log-Normale.

#### 8.2 Distribution Chi-carré ($$\chi^2$$)

**Définition** : Si $$Z_1, \ldots, Z_k \sim \mathcal{N}(0,1)$$ indépendantes :

$$Q = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2(k)$$

où $$k$$ est le **nombre de degrés de liberté**.

**PDF** : (Forme complexe, omise)

**Espérance** : $$E[Q] = k$$

**Variance** : $$\text{Var}(Q) = 2k$$

**Applications** :
- Tests d'adéquation (goodness-of-fit)
- Tests d'indépendance (tableaux de contingence)
- Intervalles de confiance pour variance

#### 8.3 Distribution de Student (t)

**Définition** : Si $$Z \sim \mathcal{N}(0,1)$$ et $$V \sim \chi^2(k)$$ indépendantes :

$$T = \frac{Z}{\sqrt{V/k}} \sim t(k)$$

**Propriétés** :
- Plus "lourde" dans les queues que Normale
- Converge vers $$\mathcal{N}(0,1)$$ quand $$k \to \infty$$
- Symétrique autour de 0

**Application** : Tests t (échantillons petits, variance inconnue)

---

## 📊 Relations entre Distributions

### Arbre Généalogique des Distributions

```
                    Bernoulli(p)
                         |
              (somme de n Bernoulli)
                         ↓
                   Binomiale(n,p)
                         |
              (limite n→∞, p→0, np=λ)
                         ↓
                    Poisson(λ)
                         |
              (temps entre événements)
                         ↓
                Exponentielle(λ)


                   Normale(μ, σ²)
                    /    |    \
            (exp)  /     |     \ (carrés)
                  /      |      \
          Log-Normale  (TCL)   Chi-carré(k)
                         |         |
                    (moyenne)   (ratio)
                         ↓         ↓
                    Normale    t-Student(k)
```

### Approximations Importantes

**Binomiale → Normale** (TCL) :

Si $$n$$ grand et $$p$$ modéré ($$np > 5$$ et $$n(1-p) > 5$$) :

$$\text{Binomiale}(n,p) \approx \mathcal{N}(np, np(1-p))$$

**Binomiale → Poisson** (événements rares) :

Si $$n$$ grand, $$p$$ petit, $$np = \lambda$$ modéré :

$$\text{Binomiale}(n,p) \approx \text{Poisson}(\lambda)$$

**Poisson → Normale** ($$\lambda$$ grand) :

Si $$\lambda > 20$$ :

$$\text{Poisson}(\lambda) \approx \mathcal{N}(\lambda, \lambda)$$

---

## 💡 Compréhension Intuitive

### Comment Choisir la Bonne Distribution ?

**Questions à se poser** :

1. **Discrète ou Continue ?**
   - Comptage → Discrète
   - Mesure → Continue

2. **Bornes ?**
   - [0, n] → Binomiale
   - [0, ∞) entier → Poisson/Géométrique
   - [0, ∞) réel → Exponentielle/Log-Normale
   - (-∞, ∞) → Normale

3. **Processus générateur ?**
   - Succès/échecs répétés → Bernoulli/Binomiale
   - Événements rares temporels → Poisson
   - Temps d'attente → Exponentielle/Géométrique
   - Somme de variables → Normale (TCL)
   - Produit de variables → Log-Normale

4. **Symétrie ?**
   - Symétrique → Normale
   - Asymétrie droite → Log-Normale, Exponentielle

---

## ⚠️ Pièges Courants et Bonnes Pratiques

### ❌ Erreur 1 : Supposer Normalité sans Vérification

**Problème** : Beaucoup de phénomènes ne sont PAS normaux !

**Contre-exemples** :
- Revenus (Log-Normale, asymétrique)
- Prix d'actifs (Log-Normale, toujours positifs)
- Temps de survie (Exponentielle, Weibull)

**Bonne pratique** : **Toujours** vérifier avec :
- Histogramme
- QQ-plot (graphique quantile-quantile)
- Test de Shapiro-Wilk, Kolmogorov-Smirnov

### ❌ Erreur 2 : Confondre Taux λ et Espérance

**Poisson** : $$\lambda$$ = espérance ET variance

**Exponentielle** : $$\lambda$$ = taux, espérance = $$1/\lambda$$

### ✅ Bonne Pratique : Méthode des Moments vs Maximum de Vraisemblance

**Estimation paramètres** :

**Méthode des moments** : Égaler moments empiriques et théoriques

**Maximum de vraisemblance** : Maximiser $$L(\theta) = \prod_{i=1}^n f(x_i; \theta)$$

**Pour Normale** : Les deux donnent $$\hat{\mu} = \bar{x}$$, $$\hat{\sigma}^2 = s^2$$

---

## 🚀 Pour Aller Plus Loin

### 📄 Papers Fondamentaux

1. **"De Moivre-Laplace Theorem"**
   - Approximation Binomiale par Normale (1733)
   - Fondation du TCL

2. **"Poisson, S.D. (1837). Recherches sur la probabilité"**
   - Introduction distribution de Poisson

3. **"Gauss, C.F. (1809). Theoria Motus"**
   - Méthode des moindres carrés, loi Normale

### 📚 Ressources

- **Casella & Berger** - *Statistical Inference* (référence académique)
- **DeGroot & Schervish** - *Probability and Statistics* (pédagogique)
- [Probability Distributions - MIT](https://ocw.mit.edu/)

---

### 📖 Cours Connexes

**Suite naturelle** :
- [[central_limit_theorem]] - TCL et applications
- [[maximum_likelihood]] - Estimation paramètres
- [[hypothesis_testing]] - Tests statistiques

**Applications** :
- [[bayesian_inference]] - Prior/Posterior avec distributions
- [[time_series]] - Modèles ARIMA, processus stochastiques
- [[generalized_linear_models]] - Régression avec distributions non-Normales

---

## 📝 Résumé Rapide

### Tableau Comparatif

| Distribution | Type | Paramètres | Support | E[X] | Var(X) | Usage |
|--------------|------|------------|---------|------|--------|-------|
| **Bernoulli** | Discr | $$p$$ | {0,1} | $$p$$ | $$p(1-p)$$ | 1 épreuve |
| **Binomiale** | Discr | $$n, p$$ | {0,...,n} | $$np$$ | $$np(1-p)$$ | n épreuves |
| **Poisson** | Discr | $$\lambda$$ | {0,1,2,...} | $$\lambda$$ | $$\lambda$$ | Événements rares |
| **Géométrique** | Discr | $$p$$ | {1,2,3,...} | $$1/p$$ | $$(1-p)/p^2$$ | Attente 1er succès |
| **Uniforme** | Cont | $$a, b$$ | [a,b] | $$(a+b)/2$$ | $$(b-a)^2/12$$ | Équiprobable |
| **Exponentielle** | Cont | $$\lambda$$ | [0,∞) | $$1/\lambda$$ | $$1/\lambda^2$$ | Temps d'attente |
| **Normale** | Cont | $$\mu, \sigma^2$$ | ℝ | $$\mu$$ | $$\sigma^2$$ | TCL, erreurs |
| **Log-Normale** | Cont | $$\mu, \sigma^2$$ | (0,∞) | $$e^{\mu+\sigma^2/2}$$ | complexe | Prix, revenus |

---

## 🔗 Intégration Repository

**Mise à jour INDEX** :

```markdown
| **Distributions classiques** | Bernoulli, Binomiale, Poisson, Normale, Exponentielle | [[common_distributions]] ✅ | ⭐⭐⭐⭐🔥 |
```

**Prochaine étape** : [[central_limit_theorem]] - Théorème Central Limite 🎯

---

**Cours créé le 2026-03-20** ✅
