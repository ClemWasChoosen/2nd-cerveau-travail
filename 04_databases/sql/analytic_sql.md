# SQL Analytique, de la Question Métier à une Requête Fiable

> **Niveau** : Débutant à intermédiaire  
> **Durée estimée** : 3 à 4 heures de lecture active, puis pratique régulière  
> **Tags** : `#sql` `#data-analyst` `#analytics` `#joins` `#window-functions` `#data-quality` `#kpi`

## Table des Matières

0. [Avant de commencer](#0-avant-de-commencer)
1. [Introduction, répondre à une question sans déformer les données](#1-introduction-répondre-à-une-question-sans-déformer-les-données)
2. [Lire et filtrer, SELECT, WHERE, ORDER BY et CASE](#2-lire-et-filtrer-select-where-order-by-et-case)
3. [Résumer, agrégations, GROUP BY et HAVING](#3-résumer-agrégations-group-by-et-having)
4. [Relier les tables, JOIN et contrôle de cardinalité](#4-relier-les-tables-join-et-contrôle-de-cardinalité)
5. [Construire une requête lisible, CTE, sous-requêtes et ensembles](#5-construire-une-requête-lisible-cte-sous-requêtes-et-ensembles)
6. [Analyser sans perdre le détail, fonctions de fenêtre](#6-analyser-sans-perdre-le-détail-fonctions-de-fenêtre)
7. [Fiabiliser le résultat, NULL, qualité et réconciliation](#7-fiabiliser-le-résultat-null-qualité-et-réconciliation)
8. [Travailler efficacement et en sécurité](#8-travailler-efficacement-et-en-sécurité)
9. [Cas fil rouge, analyser le chiffre d'affaires et la rétention](#9-cas-fil-rouge-analyser-le-chiffre-daffaires-et-la-rétention)
10. [Guide de choix](#10-guide-de-choix)
11. [⚠️ Pièges courants](#11-️pièges-courants)
12. [🎯 Exercices d'auto-évaluation](#12--exercices-dauto-évaluation)
13. [Sources et références](#13-sources-et-références)

---

## 0. Avant de commencer

### Prérequis

- Savoir distinguer une **ligne** d'une **colonne** dans un tableau.
- Comprendre qu'une table représente un type d'objet ou d'événement, par exemple un client ou une commande.
- Connaître les notions de clé primaire et de clé étrangère. Voir [Création de tables Oracle →](./creation_tables.md).
- Avoir accès à un environnement SQL en lecture seule. DBeaver, SQL Developer, BigQuery, Snowflake ou un notebook conviennent.

### Objectifs d'apprentissage

| Niveau | Objectif |
|---|---|
| **Comprendre** | Expliquer le niveau de détail, la jointure et l'agrégation d'une requête. |
| **Appliquer** | Écrire une requête utilisant filtres, agrégats, jointures, CTE et fonctions de fenêtre. |
| **Analyser** | Diagnostiquer une inflation de métrique, des valeurs manquantes ou un biais de filtre. |
| **Évaluer** | Vérifier qu'un KPI est conforme à sa définition métier avant sa restitution. |
| **Créer** | Concevoir une requête lisible qui répond à une question de performance commerciale. |

### TL;DR en 3 phrases

> SQL analytique consiste à transformer une question métier précise en une requête qui respecte le niveau de détail des données. Les outils fondamentaux sont les filtres, agrégations, jointures, CTE et fonctions de fenêtre, mais leur usage n'est fiable que si l'on contrôle les `NULL`, la cardinalité et la définition du KPI. Une bonne requête est lisible, minimale, vérifiée et capable d'être expliquée à une personne métier.

---

## 1. Introduction, répondre à une question sans déformer les données

### 1.1 Analogie, le ticket de caisse et le tableau de bord

Imagine un magasin. Une ligne de ticket représente un article acheté, tandis qu'un ticket représente une commande. Si tu comptes les lignes pour calculer le nombre de commandes, un panier de cinq articles pèse cinq fois plus qu'un panier d'un article.

L'analyste ne doit donc pas seulement demander « quelle requête fonctionne ? ». Il doit d'abord demander : **une ligne de ma table représente quoi ?** Cette question s'appelle le **niveau de détail**, ou *grain*.

```
Question métier
     │
     ▼
Définition du KPI et de sa population
     │
     ▼
Tables, grain et clés disponibles
     │
     ▼
Requête SQL, contrôles qualité, restitution
```

### 1.2 Le jeu de données fil rouge

Nous utiliserons un modèle e-commerce simplifié.

| Table | Une ligne représente | Clé | Colonnes utiles |
|---|---|---|---|
| `clients` | un client | `client_id` | pays, date_inscription, canal_acquisition |
| `commandes` | une commande | `commande_id` | client_id, date_commande, statut, montant_ht |
| `lignes_commande` | un produit dans une commande | `ligne_id` | commande_id, produit_id, quantite, prix_unitaire |
| `produits` | un produit | `produit_id` | categorie, nom_produit |

Une définition exploitable du KPI doit préciser le calcul, le périmètre, la période, le grain et les exclusions.

| Élément | Définition du chiffre d'affaires net |
|---|---|
| **Calcul** | somme de `montant_ht` des commandes finalisées |
| **Population** | commandes e-commerce, tous clients confondus |
| **Période** | mois civil de la date de commande |
| **Grain de sortie** | une ligne par mois |
| **Exclusions** | commandes annulées et tests internes |

> 📌 **Points clés, section 1**
> - Le grain est la protection principale contre les KPI faux mais plausibles.
> - Avant SQL, définir la métrique, la population, la période et les exclusions.
> - Une requête répond à une question, elle ne remplace pas sa définition métier.

---

## 2. Lire et filtrer, SELECT, WHERE, ORDER BY et CASE

### 2.1 Intuition

`SELECT` choisit ce que l'on affiche. `WHERE` choisit les lignes qui entrent dans l'analyse. `ORDER BY` rend le résultat lisible. Ces trois clauses suffisent pour répondre à de nombreuses questions exploratoires, à condition de ne sélectionner que les colonnes utiles.

### 2.2 Structure minimale

```sql
SELECT
    colonne_1,
    colonne_2
FROM nom_table
WHERE condition
ORDER BY colonne_1 DESC;
```

L'ordre logique d'évaluation est important : SQL sélectionne d'abord les lignes avec `FROM` et `WHERE`, puis construit le résultat avec `SELECT`, enfin le trie avec `ORDER BY`.

### 2.3 Exemple, commandes finalisées de France

```sql
SELECT
    commande_id,
    client_id,
    date_commande,
    montant_ht
FROM commandes
WHERE statut = 'FINALISEE'
  AND date_commande >= DATE '2026-01-01'
  AND date_commande < DATE '2026-02-01'
ORDER BY montant_ht DESC;
```

Utiliser un intervalle fermé à gauche et ouvert à droite, `>= début` et `< début_suivant`, évite des erreurs lorsque `date_commande` contient aussi une heure.

### 2.4 Filtrer correctement les valeurs manquantes

`NULL` ne signifie pas zéro, ni chaîne vide. Il signifie « valeur inconnue, absente ou non applicable ». Toute comparaison avec `NULL` produit une valeur inconnue, donc `= NULL` ne fonctionne pas.

```sql
SELECT
    client_id,
    pays,
    canal_acquisition
FROM clients
WHERE pays IS NULL;
```

`COALESCE` remplace une valeur manquante par une valeur de repli explicite.

```sql
SELECT
    client_id,
    COALESCE(canal_acquisition, 'non_renseigne') AS canal_acquisition
FROM clients;
```

### 2.5 Catégoriser avec CASE

`CASE` permet de transformer une règle métier en colonne analysable.

```sql
SELECT
    commande_id,
    montant_ht,
    CASE
        WHEN montant_ht >= 200 THEN 'panier_eleve'
        WHEN montant_ht >= 80 THEN 'panier_moyen'
        ELSE 'petit_panier'
    END AS segment_panier
FROM commandes
WHERE statut = 'FINALISEE';
```

> 📌 **Points clés, section 2**
> - Sélectionner explicitement les colonnes rend le résultat lisible et limite les coûts.
> - Utiliser `IS NULL` et `IS NOT NULL`, jamais `= NULL`.
> - Les règles `CASE` doivent être nommées et documentées lorsqu'elles définissent un segment métier.

---

## 3. Résumer, agrégations, GROUP BY et HAVING

### 3.1 Intuition

Une agrégation transforme plusieurs lignes en un résumé. Une requête « chiffre d'affaires par pays » ne décrit plus une commande, elle décrit un pays. Toute colonne non agrégée dans le `SELECT` doit donc apparaître dans `GROUP BY`.

### 3.2 Les agrégats fondamentaux

| Fonction | Rôle | Attention |
|---|---|---|
| `COUNT(*)` | compte les lignes | compte aussi les lignes ayant des valeurs nulles dans d'autres colonnes |
| `COUNT(colonne)` | compte les valeurs non nulles | ne compte pas les `NULL` |
| `COUNT(DISTINCT colonne)` | compte les valeurs uniques non nulles | parfois coûteux sur gros volumes |
| `SUM()` | additionne | retourne souvent `NULL` si toutes les valeurs sont nulles |
| `AVG()` | calcule la moyenne | ignore les `NULL` |
| `MIN()` / `MAX()` | extrait les extrêmes | utile pour contrôler les périodes |

### 3.3 Exemple, CA et nombre de clients par pays

```sql
SELECT
    c.pays,
    COUNT(DISTINCT co.commande_id) AS nb_commandes,
    COUNT(DISTINCT co.client_id) AS nb_clients_acheteurs,
    SUM(co.montant_ht) AS chiffre_affaires_ht,
    AVG(co.montant_ht) AS panier_moyen_ht
FROM commandes AS co
JOIN clients AS c
    ON c.client_id = co.client_id
WHERE co.statut = 'FINALISEE'
  AND co.date_commande >= DATE '2026-01-01'
  AND co.date_commande < DATE '2026-02-01'
GROUP BY c.pays
ORDER BY chiffre_affaires_ht DESC;
```

### 3.4 WHERE ou HAVING ?

`WHERE` filtre les lignes **avant** l'agrégation. `HAVING` filtre les groupes **après** l'agrégation.

```sql
SELECT
    client_id,
    SUM(montant_ht) AS ca_client
FROM commandes
WHERE statut = 'FINALISEE'
GROUP BY client_id
HAVING SUM(montant_ht) >= 500;
```

Utiliser `WHERE` dès que possible réduit le volume traité. `HAVING` est réservé à une condition portant sur une agrégation.

### 3.5 Taux et division sûre

Un taux de conversion peut s'écrire ainsi :

$$
\text{taux de conversion} = \frac{\text{nombre de clients acheteurs}}{\text{nombre de clients ciblés}}
$$

Le numérateur et le dénominateur doivent porter sur la même population et la même période. `NULLIF` évite une erreur de division par zéro.

```sql
SELECT
    COUNT(DISTINCT CASE WHEN a_achete = 1 THEN client_id END) * 1.0
    / NULLIF(COUNT(DISTINCT client_id), 0) AS taux_conversion
FROM population_ciblee;
```

Le `* 1.0` force une division décimale dans plusieurs dialectes SQL. En Oracle, tu peux aussi employer `CAST(... AS NUMBER)`.

> 📌 **Points clés, section 3**
> - `GROUP BY` fixe le nouveau grain du résultat.
> - `WHERE` filtre les lignes, `HAVING` filtre les groupes agrégés.
> - Pour un ratio, vérifier que numérateur et dénominateur décrivent la même population.

---

## 4. Relier les tables, JOIN et contrôle de cardinalité

### 4.1 Intuition

Une jointure est une mise en relation entre deux tables par une clé. Elle ne crée pas automatiquement une relation un-à-un. Si une commande possède trois lignes de commande, joindre `commandes` à `lignes_commande` crée trois lignes pour cette commande.

```
commandes                     lignes_commande
1 ligne / commande            plusieurs lignes / commande
       commande_id ─────────────────────┐
                                         ▼
                              résultat : 1 ligne / article commandé
```

### 4.2 Les jointures principales

| Jointure | Conserve | Cas d'usage analytique |
|---|---|---|
| `INNER JOIN` | seulement les correspondances | analyser les commandes ayant un client connu |
| `LEFT JOIN` | toutes les lignes de gauche | repérer les clients sans commande ou les clés manquantes |
| `FULL OUTER JOIN` | toutes les lignes des deux côtés | réconcilier deux référentiels, si le dialecte le supporte |
| `CROSS JOIN` | toutes les combinaisons | créer une grille de dates ou segments, avec prudence |

### 4.3 Exemple, trouver les clients sans commande

```sql
SELECT
    c.client_id,
    c.pays,
    c.date_inscription
FROM clients AS c
LEFT JOIN commandes AS co
    ON co.client_id = c.client_id
   AND co.statut = 'FINALISEE'
WHERE co.commande_id IS NULL;
```

La condition sur `statut` est volontairement placée dans `ON`. La placer dans `WHERE` éliminerait les `NULL` produits par le `LEFT JOIN` et le transformerait en pratique en `INNER JOIN`.

### 4.4 Contrôler la cardinalité avant et après la jointure

Avant d'agréger une métrique monétaire, mesure le nombre de lignes et de clés distinctes.

```sql
SELECT
    COUNT(*) AS nb_lignes,
    COUNT(DISTINCT co.commande_id) AS nb_commandes,
    SUM(co.montant_ht) AS ca_avant_jointure
FROM commandes AS co
WHERE co.statut = 'FINALISEE';

SELECT
    COUNT(*) AS nb_lignes_apres_jointure,
    COUNT(DISTINCT co.commande_id) AS nb_commandes,
    SUM(co.montant_ht) AS ca_apres_jointure
FROM commandes AS co
JOIN lignes_commande AS li
    ON li.commande_id = co.commande_id
WHERE co.statut = 'FINALISEE';
```

Si `ca_apres_jointure` augmente, ce n'est généralement pas une croissance commerciale. C'est une duplication du montant de commande, une fois par ligne de commande. Il faut alors agréger les lignes au grain commande avant de les joindre, ou calculer le CA avec `SUM(li.quantite * li.prix_unitaire)` si cette valeur est la source de vérité.

> 📌 **Points clés, section 4**
> - Une jointure peut changer le grain du résultat, donc changer un KPI.
> - Choisir `LEFT JOIN` lorsqu'on veut conserver la population de gauche.
> - Comparer lignes, identifiants distincts et métriques avant et après une jointure.

---

## 5. Construire une requête lisible, CTE, sous-requêtes et ensembles

### 5.1 Intuition

Une requête analytique complexe doit se lire comme un raisonnement. Une CTE, *Common Table Expression*, découpe ce raisonnement en étapes nommées. Elle améliore la maintenance et rend les contrôles intermédiaires possibles.

### 5.2 CTE avec WITH

```sql
WITH commandes_valides AS (
    SELECT
        commande_id,
        client_id,
        date_commande,
        montant_ht
    FROM commandes
    WHERE statut = 'FINALISEE'
),
ca_par_client AS (
    SELECT
        client_id,
        SUM(montant_ht) AS ca_client
    FROM commandes_valides
    GROUP BY client_id
)
SELECT
    client_id,
    ca_client
FROM ca_par_client
WHERE ca_client >= 500
ORDER BY ca_client DESC;
```

Les noms doivent décrire le contenu, non une étape technique vague. Préférer `commandes_valides` à `table1`.

### 5.3 Sous-requêtes, quand les préférer

Une sous-requête est pertinente pour tester l'existence ou comparer une ligne à une valeur agrégée.

```sql
SELECT
    c.client_id,
    c.pays
FROM clients AS c
WHERE EXISTS (
    SELECT 1
    FROM commandes AS co
    WHERE co.client_id = c.client_id
      AND co.statut = 'FINALISEE'
);
```

`EXISTS` exprime clairement la question « ce client a-t-il au moins une commande ? ». Il évite les doublons qu'une jointure pourrait produire.

### 5.4 UNION et UNION ALL

| Opérateur | Comportement | Usage |
|---|---|---|
| `UNION` | concatène et déduplique | deux listes quand un même élément ne doit apparaître qu'une fois |
| `UNION ALL` | concatène sans dédupliquer | empiler des partitions ou sources déjà distinctes, plus performant |

```sql
SELECT client_id, 'web' AS source
FROM clients_web
UNION ALL
SELECT client_id, 'magasin' AS source
FROM clients_magasin;
```

> 📌 **Points clés, section 5**
> - Les CTE structurent la logique en étapes vérifiables.
> - `EXISTS` est souvent préférable à une jointure lorsqu'on cherche seulement une existence.
> - Utiliser `UNION ALL` par défaut si la déduplication n'est pas une exigence métier explicite.

---

## 6. Analyser sans perdre le détail, fonctions de fenêtre

### 6.1 Intuition

`GROUP BY` réduit plusieurs lignes à une ligne par groupe. Une fonction de fenêtre calcule au niveau d'un groupe tout en gardant chaque ligne visible. C'est l'outil naturel pour les rangs, cumulés, comparaisons au mois précédent et parts du total.

### 6.2 Structure

```sql
fonction() OVER (
    PARTITION BY colonne_de_groupe
    ORDER BY colonne_ordre
)
```

`PARTITION BY` découpe les groupes indépendants. `ORDER BY` ordonne les lignes à l'intérieur de chaque groupe.

### 6.3 Rang des clients par pays

```sql
WITH ca_client AS (
    SELECT
        c.pays,
        co.client_id,
        SUM(co.montant_ht) AS ca_client
    FROM commandes AS co
    JOIN clients AS c
        ON c.client_id = co.client_id
    WHERE co.statut = 'FINALISEE'
    GROUP BY c.pays, co.client_id
)
SELECT
    pays,
    client_id,
    ca_client,
    DENSE_RANK() OVER (
        PARTITION BY pays
        ORDER BY ca_client DESC
    ) AS rang_dans_le_pays
FROM ca_client;
```

`ROW_NUMBER()` attribue un numéro unique. `RANK()` laisse des trous après une égalité. `DENSE_RANK()` ne laisse pas de trou.

### 6.4 CA cumulé et évolution mensuelle

```sql
WITH ca_mensuel AS (
    SELECT
        DATE_TRUNC('month', date_commande) AS mois,
        SUM(montant_ht) AS ca_ht
    FROM commandes
    WHERE statut = 'FINALISEE'
    GROUP BY DATE_TRUNC('month', date_commande)
)
SELECT
    mois,
    ca_ht,
    SUM(ca_ht) OVER (ORDER BY mois) AS ca_cumule,
    LAG(ca_ht) OVER (ORDER BY mois) AS ca_mois_precedent,
    (ca_ht - LAG(ca_ht) OVER (ORDER BY mois)) * 1.0
        / NULLIF(LAG(ca_ht) OVER (ORDER BY mois), 0) AS evolution_mensuelle
FROM ca_mensuel
ORDER BY mois;
```

`DATE_TRUNC('month', ...)` est utilisé par PostgreSQL et Snowflake. En Oracle, employer `TRUNC(date_commande, 'MM')`. En BigQuery, employer `DATE_TRUNC(date_commande, MONTH)` pour une colonne `DATE`.

### 6.5 Cohorte de première commande

Une cohorte regroupe les clients par période de première action. Elle permet de mesurer la rétention sans confondre anciens et nouveaux clients.

```sql
WITH commandes_valides AS (
    SELECT client_id, DATE_TRUNC('month', date_commande) AS mois
    FROM commandes
    WHERE statut = 'FINALISEE'
),
cohortes AS (
    SELECT
        client_id,
        MIN(mois) AS mois_cohorte
    FROM commandes_valides
    GROUP BY client_id
)
SELECT
    c.mois_cohorte,
    cv.mois,
    COUNT(DISTINCT cv.client_id) AS clients_actifs
FROM cohortes AS c
JOIN commandes_valides AS cv
    ON cv.client_id = c.client_id
GROUP BY c.mois_cohorte, cv.mois
ORDER BY c.mois_cohorte, cv.mois;
```

> 📌 **Points clés, section 6**
> - Une fenêtre conserve le grain de départ, contrairement à `GROUP BY`.
> - `PARTITION BY` sépare les groupes, `ORDER BY` organise le calcul dans chaque groupe.
> - Agréger au bon grain avant une fenêtre, par exemple au mois avant un cumul mensuel.

---

## 7. Fiabiliser le résultat, NULL, qualité et réconciliation

### 7.1 Intuition

Une requête exécutée sans erreur n'est pas forcément vraie. La validation doit faire partie de la requête et de son commentaire de livraison. L'objectif est de détecter une rupture de volume, une clé absente, un montant inattendu ou une période incomplète avant de créer le dashboard.

### 7.2 Contrôles minimaux

| Contrôle | Question posée | Exemple |
|---|---|---|
| Volumétrie | le nombre de lignes est-il cohérent ? | commandes par jour |
| Unicité | une clé censée être unique l'est-elle ? | doublons de `commande_id` |
| Complétude | quelle part des données est manquante ? | taux de `pays` nul |
| Référentialité | toutes les clés trouvent-elles leur parent ? | commandes sans client |
| Fraîcheur | les données sont-elles à jour ? | `MAX(date_commande)` |
| Réconciliation | le total retrouve-t-il une source de référence ? | CA SQL contre finance |

### 7.3 Requêtes de contrôle

```sql
-- Unicité attendue de commande_id
SELECT
    commande_id,
    COUNT(*) AS occurrences
FROM commandes
GROUP BY commande_id
HAVING COUNT(*) > 1;

-- Commandes sans client correspondant
SELECT
    COUNT(*) AS commandes_orphelines
FROM commandes AS co
LEFT JOIN clients AS c
    ON c.client_id = co.client_id
WHERE c.client_id IS NULL;

-- Fraîcheur et couverture temporelle
SELECT
    MIN(date_commande) AS premiere_commande,
    MAX(date_commande) AS derniere_commande,
    COUNT(*) AS nb_lignes
FROM commandes;
```

### 7.4 Contrat de restitution

Avant de publier un chiffre, documente au minimum : nom du KPI, requête ou vue source, date de rafraîchissement, filtres actifs, propriétaire métier et contrôles exécutés. La traçabilité et les portes de qualité sont aussi des principes centraux de l'infrastructure de données. Voir [Infrastructure IA →](../../03_infrastructure/infra_ia.md).

> 📌 **Points clés, section 7**
> - Tester les données est une étape de production, pas une option après coup.
> - Les contrôles d'unicité, de complétude, de référentialité et de fraîcheur sont le minimum.
> - Un KPI livrable doit pouvoir être reproduit, réconcilié et expliqué.

---

## 8. Travailler efficacement et en sécurité

### 8.1 Efficacité

Une requête analytique doit limiter les données lues avant les opérations coûteuses. Éviter `SELECT *`, filtrer tôt, ne joindre que les tables nécessaires, et agrèger avant une jointure lorsqu'elle réduit réellement le volume.

| Situation | Préférer | Éviter |
|---|---|---|
| Exploration | `LIMIT` et quelques colonnes | scanner toute une table large |
| Recherche d'existence | `EXISTS` | jointure qui duplique les lignes |
| Sources disjointes | `UNION ALL` | `UNION` sans besoin de déduplication |
| Filtre temporel | borne début et borne suivante | fonction appliquée à la colonne filtrée, si cela désactive un index |
| Gros calcul récurrent | vue, table agrégée ou modèle documenté | copier-coller de la même requête dans plusieurs dashboards |

Utilise le plan d'exécution de ton moteur, `EXPLAIN` dans PostgreSQL et BigQuery, `EXPLAIN PLAN` dans Oracle, lorsqu'une requête devient lente ou coûteuse.

### 8.2 Sécurité et gouvernance

Un analyste doit privilégier un rôle lecture seule et n'extraire que les colonnes nécessaires. Les identifiants directs, coordonnées, données RH ou informations de santé ne doivent pas être exportés dans un fichier local ou un dashboard sans justification et droits appropriés.

```sql
-- Préférer des agrégats pour une restitution métier
SELECT
    pays,
    COUNT(DISTINCT client_id) AS nb_clients,
    SUM(montant_ht) AS ca_ht
FROM commandes_analytiques
GROUP BY pays;
```

> 📌 **Points clés, section 8**
> - La meilleure optimisation est souvent une requête qui lit moins de données.
> - Examiner les plans d'exécution seulement après avoir vérifié grain, filtres et jointures.
> - Le principe de minimisation protège les personnes et réduit les risques opérationnels.

---

## 9. Cas fil rouge, analyser le chiffre d'affaires et la rétention

### 9.1 Question métier

« Quel est le chiffre d'affaires mensuel par canal d'acquisition, quel canal évolue le plus par rapport au mois précédent, et ces résultats reposent-ils sur des données fiables ? »

### 9.2 Requête structurée

```sql
WITH commandes_valides AS (
    SELECT
        commande_id,
        client_id,
        DATE_TRUNC('month', date_commande) AS mois,
        montant_ht
    FROM commandes
    WHERE statut = 'FINALISEE'
      AND date_commande >= DATE '2026-01-01'
      AND date_commande < DATE '2027-01-01'
),
commandes_enrichies AS (
    SELECT
        cv.commande_id,
        cv.client_id,
        cv.mois,
        cv.montant_ht,
        COALESCE(c.canal_acquisition, 'non_renseigne') AS canal_acquisition
    FROM commandes_valides AS cv
    JOIN clients AS c
        ON c.client_id = cv.client_id
),
ca_mensuel_canal AS (
    SELECT
        mois,
        canal_acquisition,
        COUNT(DISTINCT commande_id) AS nb_commandes,
        COUNT(DISTINCT client_id) AS nb_clients_acheteurs,
        SUM(montant_ht) AS ca_ht,
        AVG(montant_ht) AS panier_moyen_ht
    FROM commandes_enrichies
    GROUP BY mois, canal_acquisition
)
SELECT
    mois,
    canal_acquisition,
    nb_commandes,
    nb_clients_acheteurs,
    ca_ht,
    panier_moyen_ht,
    LAG(ca_ht) OVER (
        PARTITION BY canal_acquisition
        ORDER BY mois
    ) AS ca_ht_mois_precedent,
    (ca_ht - LAG(ca_ht) OVER (
        PARTITION BY canal_acquisition
        ORDER BY mois
    )) * 1.0 / NULLIF(LAG(ca_ht) OVER (
        PARTITION BY canal_acquisition
        ORDER BY mois
    ), 0) AS evolution_ca_mensuelle
FROM ca_mensuel_canal
ORDER BY mois, ca_ht DESC;
```

### 9.3 Validation associée

1. Vérifier que `commandes_valides` contient un seul enregistrement par `commande_id`.
2. Comparer son CA total à la source financière de référence sur une période fermée.
3. Mesurer le taux de canal `non_renseigne`, puis signaler son évolution.
4. Vérifier que le mois le plus récent est complet avant de comparer son évolution au mois précédent.

### 9.4 Interprétation responsable

Une baisse de CA d'un canal ne suffit pas à conclure que le canal « performe moins ». Vérifie simultanément le nombre de clients acheteurs, le nombre de commandes, le panier moyen, la saisonnalité, les changements de tracking et le caractère complet de la période.

---

## 10. Guide de choix

### 10.1 Tableau comparatif

| Besoin | Construction SQL prioritaire | Point de vigilance |
|---|---|---|
| Voir quelques enregistrements | `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` | ne pas utiliser `SELECT *` par défaut |
| Calculer un KPI par segment | `GROUP BY`, agrégats | grain de la sortie et dénominateur |
| Relier deux objets métier | `JOIN` | cardinalité et clés manquantes |
| Conserver chaque ligne avec un calcul de groupe | fenêtre `OVER()` | différence avec `GROUP BY` |
| Organiser plusieurs étapes | CTE `WITH` | noms explicites et contrôles intermédiaires |
| Identifier une absence de relation | `LEFT JOIN ... IS NULL` ou `NOT EXISTS` | filtre du côté droit dans `ON` |
| Empiler deux sources | `UNION ALL` | colonnes compatibles et doubles comptes |
| Comparer au passé | `LAG()` / `LEAD()` | ordre temporel et périodes complètes |

### 10.2 Arbre de décision

```text
Ai-je besoin de résumer plusieurs lignes ?
    ├── Non → SELECT, WHERE, ORDER BY
    └── Oui → Dois-je conserver le détail de chaque ligne ?
                  ├── Oui → fonction de fenêtre OVER(...)
                  └── Non → GROUP BY + agrégats

Dois-je combiner deux tables ?
    ├── Oui, uniquement les correspondances → INNER JOIN
    ├── Oui, conserver toute la population de gauche → LEFT JOIN
    └── Je veux seulement tester qu'une correspondance existe → EXISTS
```

---

## 11. ⚠️ Pièges courants

**Piège 1 : Confondre lignes et entités**  
`COUNT(*)` sur une table de lignes de commande ne compte pas les commandes. **Identifier le grain puis employer `COUNT(DISTINCT commande_id)` si le KPI porte sur les commandes.**

**Piège 2 : Gonfler le chiffre d'affaires après une jointure**  
Joindre une table commande à une table ligne puis sommer le montant de la commande le répète pour chaque ligne. **Comparer les métriques avant et après jointure, ou agréger au grain adéquat avant de joindre.**

**Piège 3 : Casser un LEFT JOIN avec WHERE**  
Un filtre sur la table de droite dans `WHERE` supprime les `NULL` et annule l'effet du `LEFT JOIN`. **Placer le filtre de la table de droite dans `ON` quand les lignes sans correspondance doivent rester visibles.**

**Piège 4 : Utiliser `= NULL`**  
SQL applique une logique à trois valeurs, donc une égalité avec `NULL` n'est jamais vraie. **Utiliser `IS NULL`, `IS NOT NULL` et `COALESCE` selon l'intention.**

**Piège 5 : Diviser des populations différentes**  
Un numérateur sur les utilisateurs actifs et un dénominateur sur tous les comptes crée un taux ambigu. **Documenter et appliquer le même périmètre, la même période et les mêmes exclusions.**

**Piège 6 : Inclure un mois incomplet dans une tendance**  
Le mois en cours paraît mécaniquement plus faible si toutes ses journées ne sont pas chargées. **Exclure la période incomplète ou comparer à la même portion du mois précédent.**

**Piège 7 : Faire confiance au dashboard sans réconciliation**  
Une métrique peut être techniquement calculée mais différente du chiffre finance à cause des annulations, devises ou règles de reconnaissance. **Réconcilier toute nouvelle métrique avec une source de référence et tracer les écarts.**

**Piège 8 : Optimiser avant de comprendre**  
Un index ou un plan d'exécution ne répare pas une métrique dont le grain est faux. **Valider la logique métier, les filtres et les jointures avant le tuning.**

---

## 12. 🎯 Exercices d'auto-évaluation

### Questions de révision

1. Quelle différence y a-t-il entre `COUNT(*)` et `COUNT(colonne)` ?
2. Pourquoi une jointure entre `commandes` et `lignes_commande` peut-elle fausser `SUM(commandes.montant_ht)` ?
3. Dans quel cas une condition doit-elle se trouver dans `HAVING` plutôt que dans `WHERE` ?
4. Quelle est la différence principale entre `GROUP BY` et une fonction de fenêtre ?
5. Pourquoi `COALESCE` est-il plus explicite que de considérer implicitement un `NULL` comme zéro ?
6. Quelles dimensions dois-tu contrôler avant de publier un taux de conversion ?

### Exercice applicatif

Tu disposes de `clients(client_id, pays)` et de `commandes(commande_id, client_id, date_commande, statut, montant_ht)`.

Écris une requête qui retourne, pour chaque pays et pour le mois de mars 2026 : le nombre de clients inscrits, le nombre de clients ayant effectué au moins une commande finalisée, le CA HT, et le taux de clients acheteurs. Les pays sans commande doivent apparaître. Exclure les commandes annulées.

*Solution attendue :* partir de `clients`, utiliser un `LEFT JOIN` avec le filtre de statut et de date dans la condition de jointure, puis agréger au grain pays. Le dénominateur est `COUNT(DISTINCT c.client_id)`.

<details>
<summary>Réponses aux questions de révision et correction</summary>

1. `COUNT(*)` compte toutes les lignes du résultat, tandis que `COUNT(colonne)` ne compte que les lignes où cette colonne n'est pas `NULL`.
2. Une commande peut avoir plusieurs lignes de commande. Son montant est alors répété une fois par ligne après la jointure.
3. `HAVING` est nécessaire lorsque la condition porte sur un agrégat, par exemple `HAVING SUM(montant_ht) > 500`.
4. `GROUP BY` réduit les lignes à un résultat par groupe. Une fonction de fenêtre garde les lignes et ajoute un calcul relatif à leur groupe.
5. `COALESCE` exprime une règle métier de remplacement. Un `NULL` peut représenter une inconnue et ne doit pas être confondu avec zéro sans décision explicite.
6. Définition de l'action, population éligible, période, numérateur, dénominateur, exclusions, déduplication et complétude de la donnée.

```sql
SELECT
    c.pays,
    COUNT(DISTINCT c.client_id) AS nb_clients_inscrits,
    COUNT(DISTINCT co.client_id) AS nb_clients_acheteurs,
    COALESCE(SUM(co.montant_ht), 0) AS ca_ht,
    COUNT(DISTINCT co.client_id) * 1.0
        / NULLIF(COUNT(DISTINCT c.client_id), 0) AS taux_clients_acheteurs
FROM clients AS c
LEFT JOIN commandes AS co
    ON co.client_id = c.client_id
   AND co.statut = 'FINALISEE'
   AND co.date_commande >= DATE '2026-03-01'
   AND co.date_commande < DATE '2026-04-01'
GROUP BY c.pays
ORDER BY ca_ht DESC;
```

</details>

---

## 13. Sources et références

### Documentation technique

- [PostgreSQL, SELECT](https://www.postgresql.org/docs/current/sql-select.html)
- [PostgreSQL, fonctions de fenêtre](https://www.postgresql.org/docs/current/tutorial-window.html)
- [Oracle Database SQL Language Reference](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/)
- [GoogleSQL, fonctions analytiques](https://cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls)
- [Microsoft Power BI, comprendre le schéma en étoile](https://learn.microsoft.com/power-bi/guidance/star-schema)

### Références de pratique analytique

- **The Data Warehouse Toolkit** : Ralph Kimball, Margy Ross, 3e édition, Wiley, 2013.
- **Storytelling with Data** : Cole Nussbaumer Knaflic, Wiley, 2015.
- **Designing Data-Intensive Applications** : Martin Kleppmann, O'Reilly, 2017.

## 🔗 Liens connexes

- [Création de tables Oracle →](./creation_tables.md)
- [Prétraitement des données →](../../01_machine_learning/01_fundamentals/preprocessing_data.md)
- [Visualisation de données →](../../00_statistics_foundations/data_visualization_principles.md)
- [Infrastructure, ETL et Data Warehouse →](../../03_infrastructure/infra_ia.md)
- [Créer un document de présentation →](../../05_methodologie/creer_document_presentation.md)

*Dernière mise à jour : septembre 2026*
