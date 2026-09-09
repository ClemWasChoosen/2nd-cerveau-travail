# Outils de Manipulation de Données, Tableurs et Pandas

> **Niveau** : Débutant à intermédiaire  
> **Durée estimée** : 4 à 6 heures de lecture active et de pratique  
> **Tags** : `#data-analyst` `#excel` `#google-sheets` `#pandas` `#python` `#data-cleaning` `#eda`

## Table des Matières

0. [Avant de commencer](#0-avant-de-commencer)
1. [Introduction, choisir le bon outil avant de manipuler](#1-introduction-choisir-le-bon-outil-avant-de-manipuler)
2. [Tableurs, explorer, nettoyer et synthétiser rapidement](#2-tableurs-explorer-nettoyer-et-synthétiser-rapidement)
3. [Pandas, le modèle DataFrame](#3-pandas-le-modèle-dataframe)
4. [Importer, inspecter et sélectionner les données](#4-importer-inspecter-et-sélectionner-les-données)
5. [Nettoyer et typer, valeurs manquantes, texte et dates](#5-nettoyer-et-typer-valeurs-manquantes-texte-et-dates)
6. [Transformer, agréger et combiner les données](#6-transformer-agréger-et-combiner-les-données)
7. [Contrôler la qualité et exporter un résultat reproductible](#7-contrôler-la-qualité-et-exporter-un-résultat-reproductible)
8. [Cheatsheet Pandas dépliable](#8-cheatsheet-pandas-dépliable)
9. [Guide de choix](#9-guide-de-choix)
10. [⚠️ Pièges courants](#10-️pièges-courants)
11. [🎯 Exercices d'auto-évaluation](#11--exercices-dauto-évaluation)
12. [Sources et références](#12-sources-et-références)

---

## 0. Avant de commencer

### Prérequis

- Savoir lire un tableau comportant lignes, colonnes et en-têtes.
- Connaître les concepts de fichier CSV et XLSX.
- Avoir Python 3 installé pour la partie Pandas, ainsi qu'un environnement de travail tel que JupyterLab, VS Code ou Google Colab.
- Avoir suivi ou parcouru [SQL analytique →](../04_databases/sql/analytic_sql.md), notamment les notions de filtre, jointure, agrégation et contrôle de grain.

### Objectifs d'apprentissage

| Niveau | Objectif |
|---|---|
| **Comprendre** | Distinguer les cas d'usage d'un tableur, de Pandas et de SQL. |
| **Appliquer** | Importer, explorer, filtrer, nettoyer et exporter un jeu de données avec Pandas. |
| **Analyser** | Choisir une jointure, une agrégation et un contrôle qualité adaptés au grain des données. |
| **Évaluer** | Identifier les manipulations manuelles ou ambiguës qui compromettent la fiabilité du résultat. |
| **Créer** | Construire un petit pipeline reproductible qui produit un dataset analytique propre. |

### TL;DR en 3 phrases

> Les tableurs sont excellents pour une exploration rapide, une vérification ponctuelle et une restitution légère, tandis que Pandas convient aux transformations répétables, aux volumes intermédiaires et aux analyses documentées. Quel que soit l'outil, il faut conserver le grain, expliciter les règles de nettoyage et contrôler les lignes avant et après chaque étape critique. Le code Pandas n'est fiable que lorsqu'il est lisible, déterministe et accompagné de contrôles de qualité.

---

## 1. Introduction, choisir le bon outil avant de manipuler

### 1.1 Analogie, établi, atelier et entrepôt

Un tableur est comme un établi : on voit immédiatement les pièces, on peut tester une idée rapidement et la montrer à un collègue. Pandas est un atelier automatisé : chaque opération est décrite, rejouable et appliquée à un lot entier. SQL est l'entrepôt : il permet de filtrer et d'agréger directement au plus près de données potentiellement volumineuses.

Le mauvais outil augmente les risques. Un copier-coller manuel est fragile pour une opération répétée. Charger tout un warehouse dans Pandas est inutile quand SQL peut retourner le dataset analytique déjà agrégé.

### 1.2 Comparaison des outils

| Critère | Excel / Google Sheets | Pandas | SQL |
|---|---|---|---|
| Exploration visuelle immédiate | Excellent | Bon | Moyen |
| Transformation répétable | Faible à moyen | Excellent | Excellent |
| Collaboration métier | Excellent | Moyen | Moyen |
| Données relationnelles volumineuses | Faible | Moyen | Excellent |
| Traçabilité exacte des étapes | Moyen | Excellent | Excellent |
| Dashboard et tableau de synthèse | Bon | Faible | Faible |

### 1.3 Workflow recommandé

```text
Source brute
    │
    ├── Question ponctuelle, faible volume → tableur
    │
    ├── Extraction et agrégation en base → SQL
    │
    └── Nettoyage, enrichissement, contrôle reproductible → Pandas
                                                          │
                                                          ▼
                                           fichier analytique ou visualisation
```

> 📌 **Points clés, section 1**
> - Le choix de l'outil dépend de la répétition, du volume, de la collaboration et de la sensibilité des données.
> - SQL réduit les données à la source, Pandas les transforme de manière rejouable, le tableur facilite l'exploration et la restitution.
> - Aucun outil ne dispense de vérifier le grain, les filtres et les règles métier.

---

## 2. Tableurs, explorer, nettoyer et synthétiser rapidement

### 2.1 Intuition

Excel et Google Sheets sont souvent la première interface avec les équipes métier. Ils sont pertinents pour comprendre rapidement un export, produire un contrôle ponctuel ou partager une synthèse simple. En revanche, une suite d'étapes manuelles répétées doit être remplacée par une requête, un script ou une procédure documentée.

### 2.2 Rendre un tableau analysable

Un tableau fiable possède une seule ligne d'en-têtes, une colonne par variable, une ligne par observation et un type de donnée cohérent par colonne.

| À faire | À éviter | Pourquoi |
|---|---|---|
| Une ligne d'en-têtes explicites | Cellules fusionnées dans les données | Les filtres et imports restent fiables |
| Une valeur atomique par cellule | `Paris / Lyon` dans une seule cellule | La donnée reste filtrable et joignable |
| Une colonne date au vrai format date | Dates écrites sous plusieurs formats | Les périodes et calculs restent corrects |
| Une table source séparée du rendu | Formules et notes mélangées aux données | Les transformations restent auditables |

### 2.3 Fonctions de base utiles

Les noms peuvent légèrement différer entre Excel localisé et Google Sheets. La logique est identique.

| Besoin | Fonction type | Exemple |
|---|---|---|
| Condition | `SI` / `IF` | `=SI(C2>=100;"élevé";"standard")` |
| Somme conditionnelle | `SOMME.SI.ENS` / `SUMIFS` | CA pour un pays et un mois |
| Comptage conditionnel | `NB.SI.ENS` / `COUNTIFS` | nombre de commandes finalisées |
| Recherche de clé | `RECHERCHEX` / `XLOOKUP` | enrichir une commande avec le pays client |
| Gestion d'erreur | `SIERREUR` / `IFERROR` | afficher un libellé si la clé est absente |
| Nettoyage texte | `SUPPRESPACE`, `MAJUSCULE`, `MINUSCULE` | normaliser un libellé ou un e-mail |

### 2.4 Tableau croisé dynamique

Un tableau croisé dynamique, ou TCD, est un agrégateur visuel. Il permet de passer rapidement d'une table transactionnelle à un résultat au grain métier, par exemple le chiffre d'affaires par mois et par pays.

```text
Lignes      : mois
Colonnes    : pays
Valeurs     : somme de montant_ht
Filtre      : statut = FINALISEE
```

Avant de communiquer le TCD, vérifie le champ agrégé, les filtres actifs et la présence éventuelle de doublons. Un TCD résume fidèlement une table, même lorsque la table source est erronée.

### 2.5 Quand quitter le tableur

Quitte le tableur lorsqu'une opération doit être rejouée régulièrement, qu'elle dépend de nombreux copier-coller, qu'elle combine plusieurs sources ou qu'elle utilise des données sensibles. La version automatisée doit préserver l'export brut et produire un résultat dans un dossier distinct.

> 📌 **Points clés, section 2**
> - Le tableur est un excellent outil d'exploration et de dialogue métier, pas une chaîne de production cachée.
> - Une table source propre est une condition préalable aux filtres, formules et TCD fiables.
> - Dès qu'une manipulation devient récurrente, automatiser et documenter le processus.

---

## 3. Pandas, le modèle DataFrame

### 3.1 Intuition

Un `DataFrame` est un tableau en mémoire dont les colonnes ont un nom et un type. Il se manipule par opérations vectorisées : une instruction décrit une opération appliquée à toute une colonne, plutôt qu'une boucle manuelle cellule par cellule.

```text
CSV / Excel / SQL
       │
       ▼
   DataFrame brut
       │
       ├── inspection et contrôles
       ├── nettoyage et typage
       ├── jointures et agrégations
       ▼
DataFrame analytique → CSV, Parquet, graphique, notebook
```

### 3.2 Importer Pandas et conventions

```python
import pandas as pd

# Convention universelle : df désigne un DataFrame.
df = pd.read_csv("data/commandes.csv")
```

Préférer des noms qui renseignent le contenu et le grain. `commandes_brutes`, `commandes_valides` et `ca_mensuel` sont plus sûrs que réutiliser `df` à toutes les étapes.

### 3.3 Vectorisation plutôt que boucle

```python
# Bon : l'opération est appliquée à toute la colonne.
commandes["montant_ttc"] = commandes["montant_ht"] * 1.20

# À éviter pour une transformation simple : plus lent et moins lisible.
for index, ligne in commandes.iterrows():
    commandes.loc[index, "montant_ttc"] = ligne["montant_ht"] * 1.20
```

Une boucle n'est pas toujours interdite, mais elle est rarement le bon premier choix pour une transformation de colonne.

> 📌 **Points clés, section 3**
> - Un DataFrame est un tableau typé et nommé, pas un simple fichier Excel ouvert dans Python.
> - Conserver des noms de variables décrivant le contenu rend le pipeline vérifiable.
> - Les opérations vectorisées sont en général plus claires et mieux adaptées que les boucles ligne par ligne.

---

## 4. Importer, inspecter et sélectionner les données

### 4.1 Intuition

La première action n'est pas de transformer. Il faut d'abord regarder la taille, les colonnes, les types, les valeurs manquantes et quelques exemples. Une mauvaise lecture initiale, par exemple un identifiant converti en nombre ou une date lue comme texte, contamine toutes les étapes suivantes.

### 4.2 Importer explicitement

```python
import pandas as pd

commandes = pd.read_csv(
    "data/commandes.csv",
    dtype={"commande_id": "string", "client_id": "string"},  # Préserve les zéros initiaux.
    parse_dates=["date_commande"],                                # Lit directement la colonne comme date.
    na_values=["", "NA", "N/A", "null"]                        # Normalise les valeurs manquantes connues.
)

clients = pd.read_excel(
    "data/clients.xlsx",
    sheet_name="clients",
    dtype={"client_id": "string"}
)
```

### 4.3 Inspecter systématiquement

```python
# Taille et aperçu des observations.
print(commandes.shape)
display(commandes.head())

# Types, non-nullité et mémoire.
commandes.info()

# Statistiques numériques et catégorielles.
display(commandes.describe(include="all"))

# Valeurs manquantes par colonne, triées du plus problématique au moins problématique.
missing_rate = commandes.isna().mean().sort_values(ascending=False)
display(missing_rate)
```

### 4.4 Sélectionner lignes et colonnes

```python
# Une colonne, résultat de type Series.
montants = commandes["montant_ht"]

# Plusieurs colonnes, résultat de type DataFrame.
colonnes_utiles = commandes[["commande_id", "client_id", "montant_ht"]]

# Filtre avec loc, explicite et recommandé.
commandes_finalisees = commandes.loc[
    commandes["statut"].eq("FINALISEE"),
    ["commande_id", "client_id", "date_commande", "montant_ht"]
].copy()  # copy évite les ambiguïtés lors des modifications suivantes.

# Plusieurs conditions : parenthèses autour de chaque condition.
masque = (
    commandes["statut"].eq("FINALISEE")
    & commandes["montant_ht"].gt(0)
    & commandes["date_commande"].ge("2026-01-01")
)
commandes_2026 = commandes.loc[masque].copy()
```

`&`, `|` et `~` servent respectivement à « et », « ou » et « non ». Les mots Python `and` et `or` ne s'appliquent pas directement à une Series Pandas.

> 📌 **Points clés, section 4**
> - Définir les types critiques dès l'import évite des erreurs invisibles, notamment sur les dates et identifiants.
> - Inspecter systématiquement structure, exemples, types, statistiques et valeurs manquantes.
> - Utiliser `.loc[filtre, colonnes].copy()` rend la sélection et les futures modifications explicites.

---

## 5. Nettoyer et typer, valeurs manquantes, texte et dates

### 5.1 Intuition

Nettoyer n'est pas effacer tout ce qui paraît gênant. Chaque transformation est une décision : une valeur manquante peut traduire une absence d'information, une règle métier ou une erreur de collecte. La décision doit être justifiée et mesurée.

### 5.2 Valeurs manquantes

```python
# Compter et mesurer le taux de valeurs manquantes.
quality_missing = pd.DataFrame({
    "nb_manquants": commandes.isna().sum(),
    "taux_manquants": commandes.isna().mean()
}).sort_values("taux_manquants", ascending=False)

# Retirer seulement les lignes où la clé indispensable est absente.
commandes = commandes.dropna(subset=["commande_id", "client_id"])

# Rendre explicite une catégorie inconnue, sans masquer le phénomène.
clients["canal_acquisition"] = clients["canal_acquisition"].fillna("non_renseigne")
```

### 5.3 Nettoyer du texte

```python
# Le type string de Pandas conserve les manquants comme <NA>.
clients["email_normalise"] = (
    clients["email"]
    .astype("string")
    .str.strip()                         # Supprime les espaces aux extrémités.
    .str.lower()                         # Uniformise la casse.
)

clients["pays_normalise"] = (
    clients["pays"]
    .astype("string")
    .str.strip()
    .str.upper()
)
```

### 5.4 Dates et nombres

```python
# Conversion robuste, les erreurs deviennent NaT et sont donc mesurables.
commandes["date_commande"] = pd.to_datetime(
    commandes["date_commande"],
    errors="coerce",
    dayfirst=True  # À utiliser seulement si le format source est jour/mois/année.
)

# Conversion numérique robuste, utile après suppression d'un symbole monétaire.
commandes["montant_ht"] = pd.to_numeric(
    commandes["montant_ht"],
    errors="coerce"
)

# Colonnes calendaires réutilisables pour les analyses temporelles.
commandes["mois"] = commandes["date_commande"].dt.to_period("M")
commandes["jour_semaine"] = commandes["date_commande"].dt.day_name()
```

### 5.5 Dédupliquer avec une règle explicite

```python
# Contrôle des clés dupliquées avant action.
doublons = commandes[commandes.duplicated(subset="commande_id", keep=False)]

# Conserver la version la plus récente seulement si date_mise_a_jour est la règle validée.
commandes = (
    commandes
    .sort_values("date_mise_a_jour")
    .drop_duplicates(subset="commande_id", keep="last")
)
```

> 📌 **Points clés, section 5**
> - Toute imputation, suppression ou déduplication doit suivre une règle métier explicite.
> - Employer `errors="coerce"` rend les échecs de conversion visibles et quantifiables.
> - Un identifiant est le plus souvent une chaîne, même s'il ne contient que des chiffres.

---

## 6. Transformer, agréger et combiner les données

### 6.1 Intuition

Les transformations créent une table analytique au bon grain pour une question. Une agrégation réduit des transactions à une ligne par client, par pays ou par mois. Une jointure enrichit ce résultat, mais elle peut aussi le dupliquer si la relation attendue n'est pas respectée.

### 6.2 Créer des colonnes dérivées

```python
commandes_valides = commandes.loc[
    commandes["statut"].eq("FINALISEE")
].copy()

commandes_valides = commandes_valides.assign(
    montant_ttc=lambda x: x["montant_ht"] * 1.20,
    segment_panier=lambda x: pd.cut(
        x["montant_ht"],
        bins=[-float("inf"), 80, 200, float("inf")],
        labels=["petit", "moyen", "eleve"]
    )
)
```

### 6.3 Agréger avec groupby

```python
ca_mensuel = (
    commandes_valides
    .groupby("mois", as_index=False)
    .agg(
        nb_commandes=("commande_id", "nunique"),
        nb_clients=("client_id", "nunique"),
        chiffre_affaires_ht=("montant_ht", "sum"),
        panier_moyen_ht=("montant_ht", "mean")
    )
    .sort_values("mois")
)
```

`as_index=False` fournit un résultat tabulaire immédiatement exploitable. Les agrégats nommés rendent la définition du KPI visible dans le code.

### 6.4 Joindre et valider

```python
# Un client doit apparaître une seule fois dans la table clients.
clients_uniques = clients.drop_duplicates(subset="client_id")

commandes_enrichies = commandes_valides.merge(
    clients_uniques[["client_id", "pays_normalise", "canal_acquisition"]],
    on="client_id",
    how="left",
    validate="many_to_one",  # Échoue si un client est dupliqué à droite.
    indicator=True             # Ajoute _merge pour contrôler les correspondances.
)

# Taux de commandes sans client correspondant.
taux_sans_client = commandes_enrichies["_merge"].eq("left_only").mean()
assert taux_sans_client == 0, "Des commandes ne trouvent aucun client."
```

`validate="many_to_one"` est l'un des contrôles les plus utiles de Pandas. Il rend la cardinalité attendue exécutable plutôt que supposée.

### 6.5 Restructurer une table

| Opération | Objectif | Exemple |
|---|---|---|
| `pivot_table()` | passer d'un format long à une matrice de synthèse | CA par mois et pays |
| `melt()` | repasser de colonnes de mesures à un format long | préparer une visualisation |
| `concat()` | empiler des tables de même structure | concaténer des fichiers mensuels |
| `merge()` | relier deux tables par clé | enrichir commandes avec clients |

```python
ca_mois_pays = pd.pivot_table(
    commandes_enrichies,
    index="mois",
    columns="pays_normalise",
    values="montant_ht",
    aggfunc="sum",
    fill_value=0
)
```

> 📌 **Points clés, section 6**
> - Une transformation utile produit une table au grain de la question métier.
> - Donner un nom à chaque agrégat transforme le code en définition de KPI lisible.
> - Utiliser `validate` et `indicator` dans `merge` pour prévenir les jointures silencieusement erronées.

---

## 7. Contrôler la qualité et exporter un résultat reproductible

### 7.1 Intuition

Un notebook utile permet de répondre à une question aujourd'hui. Un pipeline fiable permet d'obtenir le même résultat demain, à partir des mêmes données et des mêmes règles. Chaque étape sensible doit donc produire un contrôle qui peut échouer ou être comparé à une valeur attendue.

### 7.2 Contrôles simples et efficaces

```python
# Conservation d'un indicateur de volume après chaque étape importante.
print(f"Commandes brutes : {len(commandes):,}")
print(f"Commandes finalisées : {len(commandes_valides):,}")
print(f"Commandes enrichies : {len(commandes_enrichies):,}")

# Assertions : le pipeline s'arrête si une hypothèse essentielle est violée.
assert commandes_valides["commande_id"].notna().all()
assert commandes_valides["montant_ht"].ge(0).all()
assert commandes_enrichies["commande_id"].is_unique

# Contrôle de fraîcheur des données.
derniere_date = commandes_valides["date_commande"].max()
print(f"Dernière date de commande : {derniere_date:%Y-%m-%d}")
```

### 7.3 Exporter avec le bon format

| Format | Quand l'utiliser | Limite principale |
|---|---|---|
| CSV | partage universel et fichier simple | types et métadonnées peu préservés |
| XLSX | restitution métier et plusieurs onglets | moins adapté aux pipelines automatisés volumineux |
| Parquet | stockage analytique et échanges Python/BI | moins lisible sans outil dédié |

```python
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Fichier léger et universel pour consommation métier.
ca_mensuel.to_csv(output_dir / "ca_mensuel.csv", index=False)

# Format typé et compact pour réutilisation analytique.
commandes_enrichies.to_parquet(output_dir / "commandes_enrichies.parquet", index=False)
```

### 7.4 Structure minimale d'un projet

```text
analyse_ca/
├── data/
│   ├── raw/           # Fichiers bruts, non modifiés
│   └── processed/     # Datasets intermédiaires ou validés
├── notebooks/         # Exploration et explication
├── src/               # Fonctions ou scripts rejouables
├── outputs/           # Tableaux et graphiques produits
├── README.md          # Objectif, sources, règles, exécution
└── requirements.txt   # Dépendances Python
```

Les données brutes ne doivent pas être écrasées. Conserver une source immuable facilite l'audit, la correction et la reproduction des résultats.

> 📌 **Points clés, section 7**
> - Mesurer le nombre de lignes et les valeurs critiques après chaque grande transformation.
> - Les `assert` rendent les hypothèses du pipeline testables.
> - Séparer les sources brutes, données transformées, code et résultats protège la reproductibilité.

---

## 8. Cheatsheet Pandas dépliable

> Utilise cette section comme référence rapide. Chaque bloc est indépendant et peut être ouvert uniquement quand le besoin apparaît. Les exemples supposent `import pandas as pd` et un DataFrame nommé `df`.

<details>
<summary><strong>1. Importer et exporter des fichiers</strong></summary>

```python
import pandas as pd

# Lire.
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", sep=";", encoding="utf-8")
df = pd.read_excel("data.xlsx", sheet_name="Feuil1")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")

# Lire avec types et dates explicites.
df = pd.read_csv(
    "commandes.csv",
    dtype={"client_id": "string"},
    parse_dates=["date_commande"],
    na_values=["", "NA", "null"]
)

# Écrire.
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
df.to_parquet("output.parquet", index=False)
```
</details>

<details>
<summary><strong>2. Inspecter la structure et la qualité</strong></summary>

```python
df.head()                       # Premières lignes.
df.tail()                       # Dernières lignes.
df.sample(5, random_state=42)   # Échantillon reproductible.
df.shape                        # (nombre_lignes, nombre_colonnes).
df.columns                      # Noms de colonnes.
df.dtypes                       # Types des colonnes.
df.info()                       # Types, non-nullité, mémoire.
df.describe()                   # Statistiques numériques.
df.describe(include="all")      # Inclut les colonnes texte.
df.nunique()                    # Nombre de valeurs distinctes.
df["statut"].value_counts(dropna=False)
df.isna().sum()                 # Nombre de manquants.
df.isna().mean()                # Taux de manquants.
df.duplicated().sum()           # Lignes entièrement dupliquées.
df.duplicated(subset="id").sum()  # Doublons sur une clé.
```
</details>

<details>
<summary><strong>3. Sélectionner, filtrer et trier</strong></summary>

```python
# Colonnes.
df["montant"]
df[["client_id", "montant"]]

# Lignes et colonnes avec loc.
df.loc[df["statut"].eq("FINALISEE"), ["commande_id", "montant"]]

# Position avec iloc.
df.iloc[0:10, 0:3]

# Conditions.
df[df["montant"].gt(100)]
df[df["pays"].isin(["FR", "BE", "CH"])]
df[df["email"].str.contains("@entreprise.com", na=False)]
df.query("montant > 100 and statut == 'FINALISEE'")

# Trier et limiter.
df.sort_values(["date", "montant"], ascending=[False, False])
df.nlargest(10, "montant")
df.nsmallest(10, "montant")
```
</details>

<details>
<summary><strong>4. Créer, renommer et supprimer des colonnes</strong></summary>

```python
# Créer ou remplacer une colonne.
df["montant_ttc"] = df["montant_ht"] * 1.20

# Chaîner des transformations avec assign.
df = df.assign(
    marge=lambda x: x["prix_vente"] - x["cout"],
    mois=lambda x: x["date"].dt.to_period("M")
)

# Condition vectorisée.
df["segment"] = pd.cut(
    df["montant_ht"],
    bins=[-float("inf"), 80, 200, float("inf")],
    labels=["petit", "moyen", "eleve"]
)

# Renommer et supprimer.
df = df.rename(columns={"Montant HT": "montant_ht"})
df = df.drop(columns=["colonne_obsolete"])
```
</details>

<details>
<summary><strong>5. Valeurs manquantes, doublons et conversions de types</strong></summary>

```python
# Manquants.
df["pays"] = df["pays"].fillna("non_renseigne")
df = df.dropna(subset=["commande_id"])
df = df.dropna(how="all")

# Doublons.
df = df.drop_duplicates()
df = df.drop_duplicates(subset="commande_id", keep="last")

# Conversions sûres.
df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["client_id"] = df["client_id"].astype("string")
df["statut"] = df["statut"].astype("category")
```
</details>

<details>
<summary><strong>6. Nettoyer du texte et travailler avec des dates</strong></summary>

```python
# Texte, le type string préserve les valeurs manquantes.
df["email"] = df["email"].astype("string").str.strip().str.lower()
df["code"] = df["code"].astype("string").str.replace(" ", "", regex=False)
df["ville"] = df["ville"].astype("string").str.title()

# Dates.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["annee"] = df["date"].dt.year
df["mois"] = df["date"].dt.to_period("M")
df["jour"] = df["date"].dt.day
df["jour_semaine"] = df["date"].dt.day_name()

# Filtre temporel avec dates inclusives et explicites.
masque = df["date"].between("2026-01-01", "2026-03-31", inclusive="both")
df_t1 = df.loc[masque].copy()
```
</details>

<details>
<summary><strong>7. Agréger, compter et transformer par groupe</strong></summary>

```python
# Un agrégat par groupe.
df.groupby("pays")["montant_ht"].sum()

# Plusieurs KPI nommés, résultat tabulaire.
resume = (
    df.groupby(["mois", "pays"], as_index=False)
    .agg(
        nb_commandes=("commande_id", "nunique"),
        nb_clients=("client_id", "nunique"),
        ca_ht=("montant_ht", "sum"),
        panier_moyen=("montant_ht", "mean")
    )
)

# Ajouter un calcul de groupe sans réduire les lignes.
df["ca_client"] = df.groupby("client_id")["montant_ht"].transform("sum")
df["rang_client"] = df.groupby("pays")["montant_ht"].rank(ascending=False)
```
</details>

<details>
<summary><strong>8. Joindre, concaténer et restructurer</strong></summary>

```python
# Jointure avec validation de cardinalité.
df = commandes.merge(
    clients[["client_id", "pays"]],
    on="client_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

# Vérifier les clés non reliées.
df["_merge"].value_counts()

# Empiler des tables de même structure.
df = pd.concat([janvier, fevrier, mars], ignore_index=True)

# Format long vers matrice.
pivot = df.pivot_table(
    index="mois",
    columns="pays",
    values="montant_ht",
    aggfunc="sum",
    fill_value=0
)

# Colonnes de mesures vers format long.
long = df.melt(
    id_vars=["mois", "pays"],
    value_vars=["ca_ht", "nb_commandes"],
    var_name="metrique",
    value_name="valeur"
)
```
</details>

<details>
<summary><strong>9. Opérations fréquentes, ordre, fenêtres et performance</strong></summary>

```python
# Rangs, écarts et cumul dans un ordre explicite.
df = df.sort_values(["client_id", "date"])
df["montant_precedent"] = df.groupby("client_id")["montant_ht"].shift(1)
df["ca_cumule"] = df.groupby("client_id")["montant_ht"].cumsum()

# Réduire la mémoire sur une grande table.
df["statut"] = df["statut"].astype("category")
df = pd.read_csv("grand_fichier.csv", usecols=["id", "date", "montant"])

# Lire un grand CSV par morceaux.
for chunk in pd.read_csv("grand_fichier.csv", chunksize=100_000):
    chunk = chunk.loc[chunk["statut"].eq("FINALISEE")]
    # Traiter ou agréger chaque chunk ici.
```
</details>

<details>
<summary><strong>10. Contrôles à copier dans un pipeline</strong></summary>

```python
# Contrôles d'identifiants, de montants et de fraîcheur.
assert df["commande_id"].notna().all()
assert df["commande_id"].is_unique
assert df["montant_ht"].ge(0).all()
assert df["date"].notna().all()

# Rapport synthétique de qualité.
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "nb_manquants": df.isna().sum(),
    "taux_manquants": df.isna().mean(),
    "nb_uniques": df.nunique()
})
```
</details>

---

## 9. Guide de choix

### 9.1 Tableau de décision

| Situation | Outil ou fonction à privilégier | Raison |
|---|---|---|
| Vérifier rapidement un export de quelques milliers de lignes | Tableur + filtres ou TCD | feedback visuel et partage rapide |
| Répéter chaque mois le même nettoyage | Pandas, script versionné | opérations rejouables et auditables |
| Agréger des millions de transactions | SQL, puis Pandas sur le résultat | le moteur de base est conçu pour ce volume |
| Enrichir des commandes avec un référentiel client | `merge(..., validate=...)` | jointure contrôlée par clé et cardinalité |
| Construire une synthèse par segment | `groupby().agg()` ou TCD | agrégation explicite au grain souhaité |
| Détecter les lignes sans correspondance | `merge(..., indicator=True)` | contrôle des clés orphelines |
| Préparer un dashboard | SQL ou Pandas, puis Parquet/CSV propre | séparation entre transformation et restitution |

### 9.2 Arbre de décision

```text
La question est-elle ponctuelle et doit-elle être discutée avec le métier ?
    ├── Oui → tableur, avec une source et des filtres documentés
    └── Non → l'opération est-elle répétée ou complexe ?
                  ├── Oui → Pandas ou SQL, avec contrôles automatisés
                  └── Non → le volume est-il déjà dans une base ?
                                ├── Oui → SQL en priorité
                                └── Non → Pandas pour l'exploration et la transformation
```

---

## 10. ⚠️ Pièges courants

**Piège 1 : Modifier le fichier brut**  
Un fichier source écrasé empêche de comprendre ou de corriger une transformation. **Conserver les données brutes en lecture seule et écrire les résultats dans un répertoire distinct.**

**Piège 2 : Traiter un identifiant comme un nombre**  
Les zéros initiaux disparaissent et deux systèmes peuvent ne plus se joindre. **Importer les identifiants comme `string`, sauf si un calcul numérique sur eux est réellement voulu.**

**Piège 3 : Ignorer les erreurs de conversion**  
Une date invalide ou un montant contenant une devise peut être lu comme texte sans erreur visible. **Utiliser `pd.to_datetime` ou `pd.to_numeric` avec `errors="coerce"`, puis mesurer les valeurs converties en manquantes.**

**Piège 4 : Utiliser une jointure sans contrôle de cardinalité**  
Une clé dupliquée dans un référentiel peut multiplier les lignes et gonfler une métrique. **Utiliser `validate` dans `merge` et comparer les volumes avant et après la jointure.**

**Piège 5 : Enchaîner des modifications sur une vue**  
Pandas peut afficher un avertissement `SettingWithCopyWarning` et le résultat devient ambigu. **Créer une copie explicite avec `.copy()` après un filtre avant de modifier le résultat.**

**Piège 6 : Faire des boucles pour une opération de colonne**  
Le code devient lent et difficile à relire. **Privilégier les opérations vectorisées, `assign`, `groupby`, `merge` et les méthodes `.str` ou `.dt`.**

**Piège 7 : Produire un CSV sans documenter les règles**  
Le fichier final n'explique pas les exclusions, dates, définitions de KPI ou version de source. **Créer un `README`, conserver le script et enregistrer les principaux contrôles qualité.**

**Piège 8 : Utiliser Pandas pour contourner SQL**  
Télécharger de très grandes tables augmente les coûts et les risques de données. **Filtrer et agréger au plus près de la source avec SQL, puis utiliser Pandas pour les transformations analytiques nécessaires.**

---

## 11. 🎯 Exercices d'auto-évaluation

### Questions de révision

1. Dans quel cas un tableur est-il préférable à Pandas, et à quel signal faut-il automatiser avec Pandas ou SQL ?
2. Pourquoi un `client_id` constitué uniquement de chiffres doit-il souvent être importé comme une chaîne de caractères ?
3. Quelle différence pratique existe-t-il entre `groupby().agg()` et `groupby().transform()` ?
4. Pourquoi `validate="many_to_one"` est-il utile pendant un `merge` ?
5. Quelle décision faut-il prendre avant d'appliquer `fillna()` ou `dropna()` ?
6. Cite quatre contrôles à exécuter avant d'exporter un dataset analytique.

### Exercice applicatif

Les fichiers `clients.csv` et `commandes.csv` contiennent respectivement :

```text
clients : client_id, pays, canal_acquisition
commandes : commande_id, client_id, date_commande, statut, montant_ht
```

Construis un script Pandas qui :

1. importe les deux fichiers avec les types adaptés ;
2. conserve les commandes finalisées de mars 2026 ;
3. vérifie l'unicité de `commande_id` et de `client_id` dans leur table de référence ;
4. enrichit les commandes par pays et canal, avec validation de cardinalité ;
5. calcule le nombre de commandes, le nombre de clients acheteurs, le CA HT et le panier moyen par pays ;
6. exporte le résultat au format CSV sans index.

*Solution attendue :* un DataFrame final avec une ligne par pays, accompagné d'au moins un contrôle de clé non reliée.

<details>
<summary>Réponses aux questions de révision et correction</summary>

1. Un tableur convient pour une exploration ponctuelle, visuelle et collaborative. Dès que les étapes sont répétées, multi-sources, sensibles ou difficiles à tracer, il faut automatiser avec Pandas ou SQL.
2. Un identifiant n'est pas une quantité. Son import en entier peut supprimer des zéros initiaux et compromettre les jointures.
3. `agg()` réduit chaque groupe à une ou plusieurs lignes de synthèse. `transform()` conserve le nombre de lignes initial et ajoute une valeur calculée au niveau du groupe.
4. Cette validation fait échouer le script si un client apparaît plusieurs fois dans le référentiel, au lieu de créer silencieusement une jointure plusieurs-à-plusieurs.
5. Il faut comprendre et documenter la signification métier du manque : erreur, absence de collecte, non-applicabilité ou information inconnue.
6. Par exemple : volume de lignes, unicité des clés, taux de valeurs manquantes, clés orphelines, montants négatifs inattendus, fraîcheur temporelle et réconciliation d'un total.

```python
from pathlib import Path
import pandas as pd

# 1. Import avec identifiants préservés et date interprétée explicitement.
clients = pd.read_csv(
    "data/clients.csv",
    dtype={"client_id": "string"}
)
commandes = pd.read_csv(
    "data/commandes.csv",
    dtype={"commande_id": "string", "client_id": "string"},
    parse_dates=["date_commande"]
)

# 2. Contrôles d'unicité avant la jointure.
assert clients["client_id"].is_unique, "client_id est dupliqué dans clients."
assert commandes["commande_id"].is_unique, "commande_id est dupliqué dans commandes."

# 3. Filtre métier et temporel, puis copie explicite.
masque = (
    commandes["statut"].eq("FINALISEE")
    & commandes["date_commande"].ge("2026-03-01")
    & commandes["date_commande"].lt("2026-04-01")
)
commandes_mars = commandes.loc[masque].copy()

# 4. Enrichissement avec contrôle de relation plusieurs commandes vers un client.
commandes_mars = commandes_mars.merge(
    clients[["client_id", "pays", "canal_acquisition"]],
    on="client_id",
    how="left",
    validate="many_to_one",
    indicator=True
)
assert commandes_mars["_merge"].eq("both").all(), "Client absent du référentiel."

# 5. Agrégation au grain pays.
resume_pays = (
    commandes_mars
    .groupby("pays", as_index=False)
    .agg(
        nb_commandes=("commande_id", "nunique"),
        nb_clients_acheteurs=("client_id", "nunique"),
        ca_ht=("montant_ht", "sum"),
        panier_moyen_ht=("montant_ht", "mean")
    )
    .sort_values("ca_ht", ascending=False)
)

# 6. Export sans index technique.
Path("outputs").mkdir(exist_ok=True)
resume_pays.to_csv("outputs/ca_mars_2026_par_pays.csv", index=False)
```

</details>

---

## 12. Sources et références

### Documentation technique

- [Pandas, documentation utilisateur](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas, lecture et écriture de fichiers](https://pandas.pydata.org/docs/user_guide/io.html)
- [Pandas, sélection et indexation](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [Pandas, regroupements](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pandas, fusion et jointure de tables](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Microsoft Support, tableaux croisés dynamiques](https://support.microsoft.com/excel)
- [Google Sheets, fonction QUERY](https://support.google.com/docs/answer/3093343)

### Références de pratique

- **Python for Data Analysis** : Wes McKinney, 3e édition, O'Reilly, 2022.
- **The Data Warehouse Toolkit** : Ralph Kimball, Margy Ross, 3e édition, Wiley, 2013.
- **Designing Data-Intensive Applications** : Martin Kleppmann, O'Reilly, 2017.

## 🔗 Liens connexes

- [SQL analytique →](../04_databases/sql/analytic_sql.md)
- [Création de tables Oracle →](../04_databases/sql/creation_tables.md)
- [Prétraitement des données →](../01_machine_learning/01_fundamentals/preprocessing_data.md)
- [Visualisation de données →](../00_statistics_foundations/data_visualization_principles.md)
- [Infrastructure, ETL et Data Warehouse →](../03_infrastructure/infra_ia.md)
- [Créer un document de présentation →](../05_methodologie/creer_document_presentation.md)

*Dernière mise à jour : septembre 2026*
