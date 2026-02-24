# Partie 1 : Les Concepts Fondamentaux 

## L'Intuition Géométrique
Imagine que tu as des points de deux classes différentes dans un espace. Le SVM cherche à tracer la meilleure ligne de séparation (ou hyperplan en dimension > 2).

Mais qu'est-ce qui rend une frontière "meilleure" qu'une autre ?
La réponse du SVM : **la frontière qui maximise la marge** !
La marge = la distance minimale entre la frontière de décision et les points les plus proches de chaque classe.

<img width="330" height="321" alt="SVM_margin" src="https://github.com/user-attachments/assets/5b7c3dcd-de44-4cd3-bf5f-5e9eece1b26d" />

Les points bleus et verts sont les support vectors : ce sont les points critiques qui définissent la frontière. Si tu enlèves les autres points, la frontière reste la même !


## Les Trois Types de SVM

- **Hard Margin SVM :** Séparation parfaite, aucune erreur tolérée (données linéairement séparables)
- **Soft Margin SVM :** Tolère quelques erreurs (paramètre C) - plus réaliste
- **Kernel SVM :** Pour les données non-linéairement séparables

# Partie 2 : Formulation Mathématique - Cas Linéaire 
## L'Hyperplan de Décision

Un hyperplan est défini par :
$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

Où :
- **$w$ (weight)** : vecteur normal à l'hyperplan (définit l'orientation)
- **$x$**: vecteur de features
- **$b$ (bias) :** terme de biais (déplace l'hyperplan)

Fonction de décision :
Pour un point x, on prédit :

- Classe +1 si $w · x + b ≥ 0$
- Classe -1 si $w · x + b < 0$

$$f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$$

## La Marge Mathématique
Pour un point $x_i$ de classe $y_i$ (où $y_i ∈ {-1, +1}$), la distance à l'hyperplan est :

$$\text{distance} = \frac{|{\mathbf{w} \cdot \mathbf{x}_i + b}|}{\left \lVert w \right \rVert}$$

Pour les support vectors (points sur les marges) :

$$\mathbf{w} \cdot \mathbf{x}_i + b = +1 \quad \text{(classe +1)}$$

$$\mathbf{w} \cdot \mathbf{x}_i + b = -1 \quad \text{(classe -1)}$$

La marge totale (largeur entre les deux hyperplans de marge) est :

$$\text{marge} = \frac{2}{\left \lVert \mathbf{w} \right \rVert}$$

## Le Problème d'Optimisation (Hard Margin)
Objectif : Maximiser la marge ⟺ Minimiser $\left \lVert \mathbf{w} \right \rVert$

Problème primal (Hard Margin SVM) :

$\min_{\mathbf{w}, b} \frac{1}{2} \left \lVert \mathbf{w}\right \rVert ^2 $

$$\text{sous contrainte : } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \forall i$$

Interprétation de la contrainte :
Chaque point doit être du bon côté de sa marge $y_i(w · x_i + b) ≥ 1$ garantit que les points sont correctement classés ET au-delà de la marge

# Soft Margin SVM - Le Monde Réel 🌍
Dans la réalité, les données sont rarement parfaitement séparables. On introduit des variables de relâchement (slack variables) $\varepsilon_i$.

## Les Slack Variables
$\varepsilon_i (xi)$ mesure l'erreur pour le point $i$ :

- $\varepsilon_i = 0$ : point correctement classé au-delà de la marge
- $0 < \varepsilon_i < 1$ : point dans la marge mais du bon côté
- $\varepsilon_i ≥ 1$ : point mal classé

<img width="1132" height="651" alt="1_RgFWpCEG5AvnmGF5ESy1Tg" src="https://github.com/user-attachments/assets/521d0d7e-f530-47f2-8cda-fe4a10e06652" />

## Problème d'Optimisation (Soft Margin)

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \left \lVert \mathbf{w} \right \rVert^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{sous contraintes : } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

## Le Paramètre C
C est l'hyperparamètre le plus important du SVM :

- **C grand :** Pénalise fortement les erreurs → marge étroite, peu d'erreurs (risque d'overfitting)
- **C petit :** Tolère plus d'erreurs → marge large, plus d'erreurs (régularisation forte)

Trade-off : Marge maximale vs Erreurs minimales

$$C = \frac{1}{\lambda}$$

où $\lambda$ est le paramètre de régularisation
