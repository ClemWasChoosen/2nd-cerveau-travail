# Interprétabilité des modèles : Integrated Gradients

## 1. Introduction

**Integrated Gradients (IG)** est une méthode d'attribution qui explique les prédictions d'un modèle en attribuant un score d'importance à chaque feature d'entrée. Développée par Sundararajan et al. (2017), cette méthode répond à la question : *"Quelle est la contribution de chaque feature à la prédiction du modèle ?"*

**Cas d'usage typiques :**
- Classification de texte : identifier les mots/phrases importants
- Classification d'images : pixels responsables de la prédiction
- Modèles tabulaires : features les plus influentes

---

## 2. Principe fondamental

### 2.1 L'intuition

IG mesure comment la prédiction évolue quand on "voyage" progressivement d'une **baseline** (entrée neutre) vers l'**entrée réelle**. Au lieu de regarder uniquement le gradient à l'entrée finale, IG accumule les gradients le long de tout ce chemin.

**Analogie :** Pour comprendre pourquoi une voiture est arrivée au point B, on observe tous les changements de direction durant le trajet complet depuis A jusqu'à B, pas uniquement la direction finale.

### 2.2 Les composants clés

1. **Baseline** $$x'$$ : entrée "neutre" qui ne devrait déclencher aucune prédiction forte
   - Texte : tokens [PAD] ou embeddings zéro
   - Images : pixels noirs (0) ou image floue
   - Choix crucial qui influence les attributions

2. **Chemin d'interpolation** : trajet linéaire de la baseline vers l'entrée
   $$x(\alpha) = x' + \alpha \times (x - x')$$
   où $$\alpha \in [0, 1]$$

3. **Gradients le long du chemin** : à chaque point $$\alpha$$, on calcule
   $$\frac{\partial F(x(\alpha))}{\partial x}$$

---

## 3. Formulation mathématique

### 3.1 Formule complète

Pour une feature $$i$$, l'attribution Integrated Gradients est :

$$
\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x'+ \alpha \times (x - x'))}{\partial x_i} d\alpha
$$

**Décomposition :**
- $$(x_i - x'_i)$$ : différence entre entrée et baseline pour la feature $$i$$
- $$\int_{\alpha=0}^{1} ... d\alpha$$ : intégrale des gradients le long du chemin
- $$\frac{\partial F(...)}{\partial x_i}$$ : gradient de la sortie par rapport à la feature $$i$$

### 3.2 Approximation pratique (Riemann)

En pratique, l'intégrale est approximée par une somme discrète :

$$
\text{IG}_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m} \times (x - x'))}{\partial x_i}
$$

où $$m$$ = nombre de pas (typiquement 20-100)

---

## 4. Propriétés axiomatiques (pourquoi ça marche)

### 4.1 Sensibilité (Sensitivity)

**Énoncé :** Si deux entrées diffèrent sur une seule feature et que les prédictions sont différentes, cette feature doit avoir une attribution non-nulle.

**Pourquoi c'est important :** Le gradient simple peut violer cette propriété à cause de la saturation (zones plates). IG garantit cette propriété en intégrant sur tout le chemin.

### 4.2 Complétude d'implémentation (Implementation Invariance)

**Énoncé :** Deux réseaux fonctionnellement équivalents (même sortie pour tout input) doivent donner les mêmes attributions.

**Pourquoi c'est important :** Les explications ne dépendent que du comportement du modèle, pas de son architecture interne.

### 4.3 Complétude (Completeness)

**Formule :**
$$
\sum_{i=1}^{n} \text{IG}_i(x) = F(x) - F(x')
$$

**Signification :** La somme de toutes les attributions égale la différence de prédiction entre l'entrée et la baseline. 100% de la différence est "expliquée".

**Exemple concret :** Si le modèle prédit 0.9 pour l'entrée et 0.1 pour la baseline, les attributions sommeront à 0.8.

---

## 5. Aspects pratiques

### 5.1 Choix de la baseline

**Options courantes pour le NLP :**
- Tokens [PAD] : standard pour BERT/CamemBERT
- Embeddings zéro : baseline "neutre"
- Tokens [MASK] : pour modèles de masquage
- Tokens [UNK] : pour modèles sans PAD

**Impact :** La baseline change les attributions relatives. Si baseline ≠ neutre, les attributions peuvent être biaisées.

**Recommandation :** Tester plusieurs baselines et documenter le choix.

### 5.2 Nombre de pas (m)

**Règles empiriques :**
- $$m < 20$$ : approximation grossière, résultats instables
- $$m = 50-100$$ : bon compromis précision/vitesse (recommandé)
- $$m > 200$$ : coût élevé, gain marginal

**Validation :** Le paramètre $$\delta$$ (erreur d'approximation) doit être proche de 0.

$$
\delta = \left| \sum_{i=1}^{n} \text{IG}_i(x) - (F(x) - F(x')) \right|
$$

### 5.3 Application aux Transformers (BERT, CamemBERT, etc.)

**Spécificités :**
1. **Niveau d'application :** Appliquer IG à la couche d'embeddings, pas aux IDs de tokens
2. **Agrégation :** Sommer les attributions sur la dimension embedding pour obtenir un score par token
3. **Sous-tokens :** Regrouper les sous-tokens (ex: "##able") pour obtenir l'importance du mot complet
4. **Normalisation :** Normaliser les attributions pour la visualisation (min-max ou L2)

**Formule pour un token :**

$$
\text{Attr}_{\text{token}} = \sum_{d=1}^{D} \text{IG}_{i,d}
$$

où $\$D$$ = dimension de l'embedding

---

## 6. Comparaison avec autres méthodes

| Méthode | Coût | Axiomes | Stabilité | Interactions |
|---------|------|---------|-----------|--------------|
| **Gradient simple** | Très faible | ❌ | Faible (bruit) | ❌ |
| **Gradient × Input** | Très faible | ❌ | Moyenne | ❌ |
| **Integrated Gradients** | Moyen | ✅ | Bonne | ❌ |
| **SHAP** | Très élevé | ✅ | Variable | ✅ |
| **LIME** | Élevé | ❌ | Variable | ✅ (local) |
| **Attention weights** | Nul | ❌ | N/A | ⚠️ |

**Pourquoi choisir IG :**
- Fondement théorique solide (axiomes)
- Bon compromis coût/précision
- Applicable à tout modèle différentiable
- Résultats stables et reproductibles

---

## 7. Avantages et limites

### ✅ Avantages

1. **Axiomatiquement fondé** : propriétés mathématiques prouvées
2. **Model-agnostic** : fonctionne sur tout modèle différentiable
3. **Complétude** : explique 100% de la différence de prédiction
4. **Stabilité** : moins de bruit que le gradient simple
5. **Interprétabilité locale** : explique une prédiction spécifique

### ⚠️ Limites

1. **Coût computationnel** : $$m$$ forward + backward passes (vs 1 pour gradient simple)
2. **Dépendance baseline** : choix subjectif qui influence les résultats
3. **Pas d'interactions** : attribution indépendante par feature (pas de "feature A est importante *avec* feature B")
4. **Linéarité du chemin** : le chemin linéaire peut ne pas être le plus "naturel" (alternatives : chemin brownien)
5. **Non-causalité** : corrélation ≠ causalité (comme toutes les méthodes d'attribution)

### 🔍 Quand NE PAS utiliser IG

- Modèle non différentiable (utiliser LIME/SHAP)
- Besoin d'interactions features (utiliser SHAP)
- Contraintes temps réel strictes (gradient simple plus rapide)
- Baseline non définissable (certains domaines spécifiques)

---

## 8. Points d'attention (bonnes pratiques)

### 8.1 Validation des résultats

**Test de suppression (sanity check) :**
1. Identifier les tokens avec top-k attributions
2. Les masquer/supprimer
3. Re-prédire : la probabilité de la classe doit diminuer significativement

**Cohérence multi-baselines :**
Tester 2-3 baselines différentes. Si résultats radicalement différents, le modèle est instable ou les baselines mal choisies.

### 8.2 Interprétation des valeurs

**Attributions positives :** Features qui augmentent la prédiction de la classe cible

**Attributions négatives :** Features qui diminuent la prédiction de la classe cible

**Attributions proches de zéro :** Features peu influentes pour cette prédiction

**Attention :** L'importance relative compte plus que les valeurs absolues.

### 8.3 Visualisation

**Pour le texte :**
- Heatmap colorée (rouge = négatif, vert = positif)
- Afficher top-k mots/phrases
- Normaliser pour une meilleure lisibilité

**Pour les images :**
- Overlay heatmap sur l'image originale
- Segmentation en régions importantes

---

## 9. Formule de calcul complète (résumé)

**Étapes algorithmiques :**

1. **Définir la baseline** $$x'$$

2. **Créer m points d'interpolation :**
   $$x^{(k)} = x' + \frac{k}{m}(x - x'), \quad k = 1, ..., m$$

3. **Calculer les gradients à chaque point :**
   $$g^{(k)}_i = \frac{\partial F(x^{(k)})}{\partial x_i}$$

4. **Approximer l'intégrale (moyenne des gradients) :**
   $$\overline{g}_i = \frac{1}{m} \sum_{k=1}^{m} g^{(k)}_i$$

5. **Multiplier par la différence :**
   $$\text{IG}_i = (x_i - x'_i) \times \overline{g}_i$$

6. **Normaliser si nécessaire :**
   $$\text{IG}^{\text{norm}}_i = \frac{\text{IG}_i}{\sqrt{\sum_j \text{IG}_j^2}}$$

---

## 10. Exemple concret (classification sentiment)

**Texte :** "Ce film est génial !"  
**Baseline :** Tous tokens remplacés par [PAD]  
**Prédiction :** Positif (0.92)  
**Prédiction baseline :** Neutre (0.50)

**Après calcul IG (m=50) :**

| Token | Attribution | Interprétation |
|-------|------------|----------------|
| "génial" | +0.35 | Contribue fortement au positif |
| "film" | +0.05 | Légèrement positif (contexte) |
| "est" | +0.01 | Quasi neutre |
| "!" | +0.01 | Ponctuation peu informative |

**Vérification complétude :**
$\$0.35 + 0.05 + 0.01 + 0.01 = 0.42 = 0.92 - 0.50$$ ✓

---

## 11. Pourquoi les Integrated Gradients plutôt que l'attention ?

**Idée reçue :** "L'attention montre ce qui est important dans les Transformers"

**Réalité :** Les poids d'attention ne sont **pas** des explications causales :
- Attention ≠ importance pour la prédiction
- Attention peut être diffuse même si une seule feature est cruciale
- IG mesure l'impact causal sur la sortie, l'attention mesure la dépendance contextuelle

**Recommandation :** Utiliser IG comme explication principale, attention comme signal exploratoire complémentaire.

---

## 12. Sources académiques fondamentales

**Papier fondateur :**
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *"Axiomatic Attribution for Deep Networks"*. ICML 2017.

**Extensions et analyses :**
- Sturmfels, P. et al. (2020). *"Visualizing the Impact of Feature Attribution Baselines"*. Distill.
- Kindermans, P.-J. et al. (2019). *"The (Un)reliability of saliency methods"*. Explainable AI.

**Applications aux Transformers :**
- Chefer, H. et al. (2021). *"Transformer Interpretability Beyond Attention Visualization"*. CVPR 2021.

---

## 13. Équation récapitulative finale

$$
\boxed{
\text{IG}_i(x) = \underbrace{(x_i - x'_i)}_{\text{différence}} \times \underbrace{\int_{\alpha=0}^{1} \frac{\partial F\big(x' + \alpha(x - x')\big)}{\partial x_i} d\alpha}_{\text{gradients accumulés}}
}
$$

**Propriété clé :**
$$
\boxed{
\sum_{i} \text{IG}_i(x) = F(x) - F(x') \quad \text{(Complétude)}
}
$$

---

## Résumé en 3 points

1. **Quoi ?** IG attribue un score d'importance à chaque feature en accumulant les gradients entre une baseline et l'entrée réelle.

2. **Pourquoi ?** Méthode axiomatiquement fondée (sensibilité, complétude) qui garantit des explications fidèles et reproductibles.

3. **Comment ?** Créer m points d'interpolation, calculer les gradients à chaque point, moyenner, et multiplier par la différence input-baseline.

---

**Pour aller plus loin :** Captum (PyTorch) implémente IG de manière optimisée et offre des outils de visualisation prêts à l'emploi.
