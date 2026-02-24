# Création de Tables dans Oracle 11g et 19c

## 🎯 Versions Couvertes

- **Oracle 11g Enterprise Edition 11.2.0.2.0** (version non majoritaire mais énormément utilisée)
- **Oracle 19c Enterprise Edition 19.27.0.0.0** (version majoritaire)

> ⚠️ **Légende** : 
> - 🟢 **11g+** : Disponible dans les deux versions
> - 🟡 **19c uniquement** : Non disponible en 11g
> - 🔵 **12c+** : Disponible en 19c, pas en 11g (introduit en 12c)

---

## 1. Introduction

### 1.1 Qu'est-ce qu'une table Oracle ? 🟢 **11g+**

Une table est la structure de stockage fondamentale dans Oracle. Elle stocke les données sous forme de lignes (enregistrements) et de colonnes (attributs).

**Structure logique :**
- **Tablespace** : Espace de stockage logique contenant les tables
- **Segment** : Espace alloué à une table
- **Extent** : Ensemble de blocs contigus
- **Block** : Plus petite unité de stockage Oracle (généralement 8KB)

**Pourquoi comprendre cette hiérarchie ?**
- Optimisation des performances
- Gestion de l'espace disque
- Stratégies de backup/recovery
- Partitionnement efficace

**Différences de gestion entre versions :**
- **11g** : Gestion manuelle plus importante des extents
- **19c** : Automatic Storage Management (ASM) plus mature et automatique

---

## 2. Syntaxe de Base CREATE TABLE 🟢 **11g+**

### 2.1 Structure minimale

```sql
CREATE TABLE nom_table (
    colonne1 type_donnees [contraintes],
    colonne2 type_donnees [contraintes],
    ...
    [contraintes_table]
);
```

### 2.2 Exemple simple

```sql
CREATE TABLE employes (
    employe_id NUMBER(10),
    nom VARCHAR2(50),
    prenom VARCHAR2(50),
    date_embauche DATE,
    salaire NUMBER(10,2)
);
```

**Pourquoi cette syntaxe ?**
- Oracle utilise une approche déclarative (on décrit QUOI, pas COMMENT)
- La séparation type/contrainte permet une maintenance claire
- Les tailles de colonnes optimisent le stockage

---

## 3. Types de Données Oracle

### 3.1 Types numériques 🟢 **11g+**

#### NUMBER(p,s)
- **p** (précision) : nombre total de chiffres (1-38)
- **s** (échelle) : nombre de décimales

```sql
-- Exemples identiques en 11g et 19c
age NUMBER(3)           -- 0 à 999
prix NUMBER(10,2)       -- 99999999.99 max
pourcentage NUMBER(5,2) -- 999.99 max
```

**Pourquoi NUMBER plutôt qu'INT/FLOAT ?**
- NUMBER est un type universel qui évite les problèmes d'overflow
- Précision exacte (pas d'erreurs d'arrondi comme avec FLOAT)
- Stockage optimal selon la valeur réelle

#### Types alternatifs
- **INTEGER** : alias de NUMBER(38) 🟢 **11g+**
- **FLOAT(p)** : nombre à virgule flottante (déconseillé pour l'argent) 🟢 **11g+**
- **BINARY_FLOAT/BINARY_DOUBLE** : IEEE 754 🟢 **11g+**

### 3.2 Types caractères

#### VARCHAR2(n) 🟢 **11g+**

**Limites selon les versions :**
- **11g** : Maximum 4000 bytes en table standard
- **19c** : Maximum 4000 bytes (ou 32767 bytes avec MAX_STRING_SIZE=EXTENDED)

```sql
-- 11g et 19c : Différence BYTE vs CHAR
nom VARCHAR2(50 BYTE)  -- 50 octets (défaut)
nom VARCHAR2(50 CHAR)  -- 50 caractères (UTF-8: important !)

-- Exemple pratique
CREATE TABLE test_encoding (
    nom_byte VARCHAR2(10 BYTE),
    nom_char VARCHAR2(10 CHAR)
);

-- "François" = 8 caractères mais 9 bytes (ç = 2 bytes en UTF-8)
-- Passe dans nom_char, pas dans nom_byte si < 10 bytes restants
```

**🟡 19c : Extended String Size**

```sql
-- Oracle 19c avec MAX_STRING_SIZE=EXTENDED permet :
description VARCHAR2(32767)  -- Avant 12c : maximum 4000 bytes

-- Vérifier la configuration (11g et 19c)
SELECT name, value FROM v$parameter WHERE name = 'max_string_size';
-- STANDARD = 4000 bytes max
-- EXTENDED = 32767 bytes max (12c+)
```

**Pourquoi VARCHAR2 et pas VARCHAR ?**
- VARCHAR réservé pour évolutions futures Oracle
- VARCHAR2 garantit la compatibilité

#### CHAR(n) 🟢 **11g+**
- Taille fixe (1 à 2000 bytes)
- Complété par des espaces

```sql
code_postal CHAR(5)     -- Toujours 5 caractères
code_pays CHAR(2)       -- ISO 3166-1 alpha-2
```

**Quand utiliser CHAR vs VARCHAR2 ?**
- **CHAR** : données de longueur fixe (codes, flags)
- **VARCHAR2** : données de longueur variable (noms, descriptions)
- **Performance** : CHAR peut être plus rapide en comparaison (pas de calcul de longueur)

#### CLOB (Character Large Object) 🟢 **11g+**
- Jusqu'à 4 GB de texte (dans les deux versions)
- Pour descriptions longues, documents

```sql
description CLOB
```

### 3.3 Types date/temps

#### DATE 🟢 **11g+**
- Précision à la seconde
- Stocke : année, mois, jour, heure, minute, seconde

```sql
date_embauche DATE DEFAULT SYSDATE
```

#### TIMESTAMP 🟢 **11g+**
- Précision à la fraction de seconde (jusqu'à 9 chiffres)

```sql
log_timestamp TIMESTAMP(6)  -- 6 décimales de seconde
```

#### TIMESTAMP WITH TIME ZONE 🟢 **11g+**
- Inclut le fuseau horaire

```sql
CREATE TABLE evenements (
    evt_id NUMBER,
    evt_date TIMESTAMP WITH TIME ZONE
);

INSERT INTO evenements VALUES (
    1,
    TIMESTAMP '2024-01-15 14:30:00 +01:00'
);
```

**Pourquoi plusieurs types temporels ?**
- **DATE** : applications métier classiques (RH, compta)
- **TIMESTAMP** : logs, audit, précision élevée
- **TIMESTAMP WITH TIME ZONE** : applications internationales

### 3.4 Types binaires 🟢 **11g+**

- **BLOB** : Binary Large Object (jusqu'à 4 GB)
- **RAW(n)** : Données binaires brutes (max 2000 bytes)
- **BFILE** : Pointeur vers fichier externe OS

```sql
CREATE TABLE documents (
    doc_id NUMBER,
    fichier BLOB,
    miniature RAW(2000)
);
```

### 3.5 Types spéciaux selon versions

#### JSON 🔵 **12c+** (19c uniquement)

**Oracle 19c :**
```sql
CREATE TABLE produits (
    prod_id NUMBER,
    attributs VARCHAR2(4000) CHECK (attributs IS JSON)
);

-- Ou avec type JSON natif (21c+, mais contrainte en 19c)
CREATE TABLE produits_v2 (
    prod_id NUMBER,
    attributs CLOB CHECK (attributs IS JSON)
);

-- Requêtage JSON en 19c
SELECT p.prod_id, 
       JSON_VALUE(p.attributs, '$.couleur') AS couleur
FROM produits p;
```

**Oracle 11g : Pas de support JSON natif**
```sql
-- Workaround en 11g : Stocker en CLOB ou XML
CREATE TABLE produits (
    prod_id NUMBER,
    attributs CLOB  -- Pas de validation JSON native
);

-- Validation manuelle en PL/SQL nécessaire
```

#### XML 🟢 **11g+**

```sql
-- 11g et 19c : Support XMLType
CREATE TABLE documents_xml (
    doc_id NUMBER,
    contenu XMLTYPE
);

INSERT INTO documents_xml VALUES (
    1,
    XMLTYPE('<client><nom>Dupont</nom></client>')
);
```

---

## 4. Contraintes d'Intégrité 🟢 **11g+**

### 4.1 NOT NULL

Force une colonne à avoir toujours une valeur.

```sql
CREATE TABLE clients (
    client_id NUMBER NOT NULL,
    email VARCHAR2(100) NOT NULL,
    telephone VARCHAR2(20)  -- Peut être NULL
);
```

**Pourquoi NOT NULL ?**
- Évite les NULL inattendus qui compliquent les requêtes
- Force la qualité des données à l'insertion
- Améliore les performances (pas de vérification IS NULL)

### 4.2 PRIMARY KEY (Clé Primaire)

Identifiant unique de chaque ligne (NOT NULL + UNIQUE).

```sql
-- Méthode 1 : Inline
CREATE TABLE departements (
    dept_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL
);

-- Méthode 2 : Table-level (recommandé)
CREATE TABLE departements (
    dept_id NUMBER,
    nom VARCHAR2(50) NOT NULL,
    CONSTRAINT pk_departements PRIMARY KEY (dept_id)
);

-- Méthode 3 : Clé composite
CREATE TABLE affectations (
    employe_id NUMBER,
    projet_id NUMBER,
    date_debut DATE,
    CONSTRAINT pk_affectations PRIMARY KEY (employe_id, projet_id)
);
```

**Pourquoi nommer les contraintes (CONSTRAINT pk_...) ?**
- Messages d'erreur clairs
- Facilite la maintenance (DROP, DISABLE)
- Convention de nommage : pk_ (primary key), fk_ (foreign key), uk_ (unique), ck_ (check)

**Index automatique :**
Oracle crée automatiquement un index unique sur la PK pour des performances optimales (11g et 19c).

### 4.3 FOREIGN KEY (Clé Étrangère)

Référence une clé primaire d'une autre table (intégrité référentielle).

```sql
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL,
    dept_id NUMBER,
    CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) 
        REFERENCES departements(dept_id)
);
```

#### Options ON DELETE

```sql
-- ON DELETE CASCADE : Supprime les lignes enfants
CREATE TABLE commandes (
    commande_id NUMBER PRIMARY KEY,
    client_id NUMBER,
    CONSTRAINT fk_cmd_client FOREIGN KEY (client_id) 
        REFERENCES clients(client_id) 
        ON DELETE CASCADE
);

-- ON DELETE SET NULL : Met la FK à NULL
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    manager_id NUMBER,
    CONSTRAINT fk_emp_manager FOREIGN KEY (manager_id) 
        REFERENCES employes(employe_id) 
        ON DELETE SET NULL
);

-- Par défaut : ON DELETE RESTRICT (interdit la suppression)
```

**Pourquoi quelle option ?**
- **CASCADE** : Relations fortes (commande → lignes_commande)
- **SET NULL** : Relations optionnelles (employé → manager)
- **RESTRICT** : Protection des données critiques

### 4.4 UNIQUE

Garantit l'unicité (NULL autorisé, contrairement à PK).

```sql
CREATE TABLE utilisateurs (
    user_id NUMBER PRIMARY KEY,
    email VARCHAR2(100),
    numero_secu VARCHAR2(15),
    CONSTRAINT uk_email UNIQUE (email),
    CONSTRAINT uk_secu UNIQUE (numero_secu)
);
```

**Pourquoi UNIQUE plutôt qu'une seconde PK ?**
- Une seule PK par table (par définition)
- UNIQUE pour identifiants alternatifs (email, numéro sécu, etc.)

### 4.5 CHECK

Validation de règles métier.

```sql
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL,
    age NUMBER,
    salaire NUMBER(10,2),
    sexe CHAR(1),
    CONSTRAINT ck_age CHECK (age >= 18 AND age <= 65),
    CONSTRAINT ck_salaire CHECK (salaire > 0),
    CONSTRAINT ck_sexe CHECK (sexe IN ('M', 'F', 'A'))  -- A = Autre
);
```

**Limitations des CHECK :**
- Pas de sous-requêtes
- Pas de fonctions non-déterministes (SYSDATE, CURRENT_TIMESTAMP)
- Pas de références à d'autres lignes

```sql
-- INVALIDE (11g et 19c)
CHECK (salaire < (SELECT AVG(salaire) FROM employes))

-- SOLUTION : Utiliser un TRIGGER (voir section 11)
```

### 4.6 DEFAULT 🟢 **11g+**

Valeur par défaut si non spécifiée.

```sql
CREATE TABLE logs (
    log_id NUMBER PRIMARY KEY,
    message VARCHAR2(500),
    log_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severite VARCHAR2(10) DEFAULT 'INFO',
    actif CHAR(1) DEFAULT 'Y'
);

-- Insertion sans spécifier les colonnes DEFAULT
INSERT INTO logs (log_id, message) VALUES (1, 'Test');
-- log_date = maintenant, severite = 'INFO', actif = 'Y'
```

---

## 5. Auto-incrémentation : Différences Majeures

### 5.1 🟡 Oracle 19c : IDENTITY Columns 🔵 **12c+**

```sql
-- Oracle 19c : Méthode moderne (introduite en 12c)
CREATE TABLE employes (
    employe_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nom VARCHAR2(50)
);

-- Options avancées
CREATE TABLE produits (
    produit_id NUMBER GENERATED BY DEFAULT AS IDENTITY 
        (START WITH 1000 INCREMENT BY 10) PRIMARY KEY,
    nom VARCHAR2(100)
);

-- ALWAYS : Impossible d'insérer une valeur manuellement
-- BY DEFAULT : Permet insertion manuelle si besoin

-- Insertion automatique
INSERT INTO employes (nom) VALUES ('Dupont');
-- employe_id sera auto-généré (1, 2, 3...)
```

### 5.2 🟢 Oracle 11g : Séquences + Triggers **11g+**

En Oracle 11g, pas de colonnes IDENTITY. Il faut utiliser des séquences et triggers :

```sql
-- Étape 1 : Créer la table
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL
);

-- Étape 2 : Créer une séquence
CREATE SEQUENCE seq_employe_id 
    START WITH 1 
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;

-- Étape 3 : Créer un trigger pour auto-increment
CREATE OR REPLACE TRIGGER trg_employe_id
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    -- Si l'ID n'est pas fourni, utiliser la séquence
    IF :NEW.employe_id IS NULL THEN
        SELECT seq_employe_id.NEXTVAL INTO :NEW.employe_id FROM DUAL;
    END IF;
END;
/

-- Insertion (ID auto-généré)
INSERT INTO employes (nom) VALUES ('Dupont');

-- Ou insertion manuelle possible
INSERT INTO employes (employe_id, nom) VALUES (seq_employe_id.NEXTVAL, 'Martin');
```

**Pourquoi cette approche en 11g ?**
- Pas de IDENTITY columns avant Oracle 12c
- Séquences = objets réutilisables (plusieurs tables peuvent partager)
- Triggers = personnalisation complète de la logique

**🔵 En Oracle 19c : Les deux méthodes fonctionnent**

```sql
-- Oracle 19c : Approche moderne (IDENTITY) OU classique (Séquence+Trigger)
-- Recommandation : Utiliser IDENTITY pour nouveau code
-- Maintenir Séquence+Trigger pour compatibilité avec code existant
```

---

## 6. Options Avancées de CREATE TABLE

### 6.1 Spécification du Tablespace 🟢 **11g+**

Un tablespace est un espace logique qui contient les objets stockés dans la base de données comme les tables ou les index.
Un tablespace est composé d'au moins un datafile, c'est-à-dire un fichier de données qui est physiquement présent sur le serveur à l'endroit stipulé lors de sa création.
Chaque datafile est constitué de segments d'au moins un extent (ou page) lui-même constitué d'au moins 3 blocs : l'élément le plus petit d'une base de données.
L'extent n'a aucune signification particulière, c'est juste un groupe de blocs contigus pouvant accueillir des données, nous verrons néanmoins que cette notion d'extent peut poser des problèmes de gestion d'espace disque. 

```sql
CREATE TABLE grosses_donnees (
    id NUMBER PRIMARY KEY,
    data VARCHAR2(4000)
)
TABLESPACE data_tbs
STORAGE (
    INITIAL 10M
    NEXT 10M
    PCTINCREASE 0
);
```

**Pourquoi spécifier le tablespace ?**
- Séparation données/index
- Optimisation I/O (différents disques)
- Gestion des backups (tablespaces critiques vs non-critiques)

**Différences 11g vs 19c :**
- **11g** : Gestion STORAGE plus importante (INITIAL, NEXT, PCTINCREASE)
- **19c** : ASM gère automatiquement, paramètres STORAGE moins critiques

### 6.2 Colonnes Virtuelles (Computed Columns) 🟢 **11g+**

```sql
-- Disponible en 11g et 19c
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    prenom VARCHAR2(50),
    nom VARCHAR2(50),
    salaire_brut NUMBER(10,2),
    
    -- Colonnes virtuelles (calculées automatiquement)
    nom_complet VARCHAR2(101) GENERATED ALWAYS AS (prenom || ' ' || nom) VIRTUAL,
    salaire_net NUMBER(10,2) GENERATED ALWAYS AS (salaire_brut * 0.77) VIRTUAL
);

-- Index sur colonne virtuelle (11g et 19c)
CREATE INDEX idx_emp_nom_complet ON employes(nom_complet);

-- Utilisation transparente
SELECT nom_complet, salaire_net FROM employes WHERE nom_complet LIKE 'Jean%';
```

**Pourquoi les colonnes virtuelles ?**
- Pas de stockage supplémentaire
- Toujours à jour (pas de risque d'incohérence)
- Peuvent être indexées
- Simplifient les requêtes

### 6.3 Invisible Columns 🔵 **12c+** (19c uniquement)

```sql
-- Oracle 19c uniquement
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50),
    salaire NUMBER(10,2) INVISIBLE  -- Colonne masquée par défaut
);

-- SELECT * ne retourne pas 'salaire'
SELECT * FROM employes;  -- Retourne uniquement employe_id, nom

-- Accès explicite possible
SELECT employe_id, nom, salaire FROM employes;  -- OK

-- Modifier la visibilité
ALTER TABLE employes MODIFY (salaire VISIBLE);
```

**⚠️ Oracle 11g : Pas de colonnes invisibles**

Workaround en 11g : Créer des vues

```sql
-- 11g : Vue sans colonnes sensibles
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50),
    salaire NUMBER(10,2)  -- Visible dans la table
);

CREATE VIEW v_employes_public AS
SELECT employe_id, nom FROM employes;  -- Vue sans salaire

-- Utiliser la vue au lieu de la table
SELECT * FROM v_employes_public;
```

### 6.4 Table Partitionnée 🟢 **11g+** (avec différences syntaxiques)

#### Partitionnement RANGE

**Oracle 11g :**
```sql
-- 11g : Partitions définies manuellement
CREATE TABLE ventes (
    vente_id NUMBER,
    produit_id NUMBER,
    montant NUMBER(10,2),
    date_vente DATE
)
PARTITION BY RANGE (date_vente) (
    PARTITION ventes_2023 VALUES LESS THAN (TO_DATE('2024-01-01', 'YYYY-MM-DD')),
    PARTITION ventes_2024 VALUES LESS THAN (TO_DATE('2025-01-01', 'YYYY-MM-DD')),
    PARTITION ventes_2025 VALUES LESS THAN (TO_DATE('2026-01-01', 'YYYY-MM-DD'))
);

-- Pour ajouter une nouvelle partition (manuel en 11g)
ALTER TABLE ventes ADD PARTITION ventes_2026 
    VALUES LESS THAN (TO_DATE('2027-01-01', 'YYYY-MM-DD'));
```

**Oracle 19c : Partitionnement automatique INTERVAL** 🔵 **12c+**
```sql
-- 19c : Création automatique des partitions
CREATE TABLE ventes (
    vente_id NUMBER,
    produit_id NUMBER,
    montant NUMBER(10,2),
    date_vente DATE
)
PARTITION BY RANGE (date_vente) 
INTERVAL (NUMTOYMINTERVAL(1, 'MONTH'))  -- Partition auto par mois
(
    PARTITION ventes_init VALUES LESS THAN (TO_DATE('2024-01-01', 'YYYY-MM-DD'))
);

-- Oracle crée automatiquement les partitions au fur et à mesure des insertions
-- Plus besoin de ALTER TABLE ADD PARTITION manuellement !
```

#### Partitionnement LIST 🟢 **11g+**

```sql
-- Identique en 11g et 19c
CREATE TABLE clients_region (
    client_id NUMBER,
    nom VARCHAR2(100),
    region VARCHAR2(20)
)
PARTITION BY LIST (region) (
    PARTITION europe VALUES ('FR', 'DE', 'UK'),
    PARTITION amerique VALUES ('US', 'CA', 'MX'),
    PARTITION asie VALUES ('CN', 'JP', 'IN')
);
```

#### Partitionnement HASH 🟢 **11g+**

```sql
-- Identique en 11g et 19c
CREATE TABLE transactions (
    trans_id NUMBER,
    montant NUMBER
)
PARTITION BY HASH (trans_id)
PARTITIONS 8;
```

**Pourquoi partitionner ?**
- **Performance** : Requêtes n'accèdent qu'aux partitions nécessaires (partition pruning)
- **Maintenance** : Purge/archivage par partition entière
- **Disponibilité** : Backup/recovery partition par partition
- **Volumétrie** : Tables de plusieurs TB gérables

**Différence majeure 11g vs 19c :**
- **11g** : Gestion manuelle des partitions RANGE (ALTER TABLE ADD PARTITION)
- **19c** : INTERVAL partitioning automatique (12c+)

### 6.5 Tables Temporaires 🟢 **11g+**

#### Global Temporary Table (GTT)

```sql
-- Identique en 11g et 19c

-- Données conservées pendant la transaction
CREATE GLOBAL TEMPORARY TABLE temp_calculs (
    id NUMBER,
    resultat NUMBER
)
ON COMMIT DELETE ROWS;

-- Données conservées pendant la session
CREATE GLOBAL TEMPORARY TABLE temp_session (
    id NUMBER,
    info VARCHAR2(100)
)
ON COMMIT PRESERVE ROWS;
```

**Différences avec tables normales :**
- Structure permanente, données temporaires
- Chaque session voit uniquement ses données
- Pas de génération de REDO logs (performances)
- Idéal pour traitements batch intermédiaires

### 6.6 Tables en Lecture Seule 🟢 **11g+**

```sql
-- 11g et 19c
CREATE TABLE referentiel_pays (
    code CHAR(2) PRIMARY KEY,
    nom VARCHAR2(100)
);

-- Remplissage initial
INSERT INTO referentiel_pays VALUES ('FR', 'France');
INSERT INTO referentiel_pays VALUES ('US', 'États-Unis');
COMMIT;

-- Passer en lecture seule
ALTER TABLE referentiel_pays READ ONLY;

-- Tentative de modification : erreur
UPDATE referentiel_pays SET nom = 'USA' WHERE code = 'US';
-- ORA-12081: update operation not allowed on table "REFERENTIEL_PAYS"
```

---

## 7. CREATE TABLE AS SELECT (CTAS) 🟢 **11g+**

Crée une table à partir d'une requête.

```sql
-- Copie simple (11g et 19c)
CREATE TABLE employes_backup AS
SELECT * FROM employes;

-- Copie partielle avec transformation
CREATE TABLE employes_actifs AS
SELECT employe_id, nom, prenom, UPPER(email) AS email
FROM employes
WHERE statut = 'ACTIF';

-- Table vide avec même structure
CREATE TABLE employes_template AS
SELECT * FROM employes WHERE 1=0;
```

**Avantages CTAS :**
- Très rapide (chemin direct, moins de REDO en NOLOGGING)
- Création automatique de structure
- Idéal pour tables de reporting/datawarehouse

**Limitations (11g et 19c) :**
- Ne copie PAS les contraintes (sauf NOT NULL)
- Ne copie PAS les index
- Ne copie PAS les triggers

**🟡 Performance 19c vs 11g :**

```sql
-- 19c : Option NOLOGGING plus performante
CREATE TABLE employes_backup 
NOLOGGING  -- Réduit REDO logs
AS SELECT * FROM employes;

-- Recréer les contraintes après CTAS
ALTER TABLE employes_backup 
    ADD CONSTRAINT pk_emp_backup PRIMARY KEY (employe_id);
```

---

## 8. Index : Performances et Optimisation

### 8.1 Introduction aux Index 🟢 **11g+**

Un index est une structure de données qui améliore la vitesse de récupération des données en créant un chemin d'accès rapide.

**Analogie :** Un index SQL est comme l'index d'un livre : au lieu de lire toutes les pages, on va directement à la bonne page.

**Pourquoi créer des index ?**
- Accélérer les SELECT avec WHERE, JOIN, ORDER BY
- Enforcer l'unicité (UNIQUE, PRIMARY KEY)
- Améliorer les performances des FK

**Coût des index :**
- Espace disque supplémentaire
- Ralentissement des INSERT/UPDATE/DELETE
- Maintenance (reconstruction périodique)

**Principe : Équilibrer lectures vs écritures**

### 8.2 Types d'Index

#### 8.2.1 B-Tree Index (Index par défaut) 🟢 **11g+**

Structure arborescente équilibrée, idéale pour la plupart des cas.

```sql
-- Création d'index simple
CREATE INDEX idx_emp_nom ON employes(nom);

-- Index composite (plusieurs colonnes)
CREATE INDEX idx_emp_dept_nom ON employes(dept_id, nom);

-- Index unique (enforce unicité)
CREATE UNIQUE INDEX idx_emp_email ON employes(email);
```

**Quand utiliser un B-Tree ?**
- Colonnes avec forte cardinalité (beaucoup de valeurs distinctes)
- Recherches par égalité (=) ou plage (BETWEEN, <, >)
- Tri (ORDER BY)

**Ordre des colonnes dans index composite :**

```sql
-- Index sur (dept_id, nom)
CREATE INDEX idx_emp_dept_nom ON employes(dept_id, nom);

-- ✅ Utilise l'index
SELECT * FROM employes WHERE dept_id = 10;
SELECT * FROM employes WHERE dept_id = 10 AND nom = 'Dupont';

-- ❌ N'utilise PAS l'index (colonne dept_id non utilisée)
SELECT * FROM employes WHERE nom = 'Dupont';

-- Règle : L'index est utilisé si on commence par la première colonne
```

**Pourquoi cet ordre ?**
- Oracle lit l'index de gauche à droite
- La première colonne est le critère de tri principal
- Mettre les colonnes les plus sélectives en premier

#### 8.2.2 Bitmap Index 🟢 **11g+**

Index optimisé pour colonnes avec faible cardinalité (peu de valeurs distinctes).

```sql
-- Colonnes idéales pour bitmap : sexe, statut, oui/non
CREATE BITMAP INDEX idx_emp_sexe ON employes(sexe);
CREATE BITMAP INDEX idx_emp_actif ON employes(actif);
CREATE BITMAP INDEX idx_client_region ON clients(region);
```

**Quand utiliser un Bitmap Index ?**
- Colonnes avec peu de valeurs distinctes (< 100-1000)
- Tables en lecture intensive (datawarehouse)
- Requêtes avec multiples conditions AND/OR

**⚠️ Ne PAS utiliser sur tables OLTP avec beaucoup d'UPDATE/DELETE**
- Lock de multiples lignes simultanément
- Dégradation importante des écritures

**Exemple de performance :**

```sql
-- Sans bitmap index
SELECT COUNT(*) FROM employes 
WHERE sexe = 'F' AND actif = 'Y';
-- Full table scan : 1 million de lignes scannées

-- Avec bitmap index sur sexe et actif
CREATE BITMAP INDEX idx_emp_sexe ON employes(sexe);
CREATE BITMAP INDEX idx_emp_actif ON employes(actif);

-- Oracle combine les bitmaps (opération AND binaire ultra-rapide)
-- Scan réduit à quelques milliers de lignes
```

**🟢 11g et 19c : Bitmap disponible (Enterprise Edition uniquement)**

#### 8.2.3 Function-Based Index 🟢 **11g+**

Index sur le résultat d'une fonction ou expression.

```sql
-- Index sur UPPER(nom) pour recherches insensibles à la casse
CREATE INDEX idx_emp_nom_upper ON employes(UPPER(nom));

-- Requête qui utilise l'index
SELECT * FROM employes WHERE UPPER(nom) = 'DUPONT';

-- Index sur expression calculée
CREATE INDEX idx_emp_salaire_annuel ON employes(salaire * 12);

-- Requête qui utilise l'index
SELECT * FROM employes WHERE salaire * 12 > 50000;

-- Index sur extraction de date
CREATE INDEX idx_ventes_annee ON ventes(EXTRACT(YEAR FROM date_vente));

SELECT * FROM ventes WHERE EXTRACT(YEAR FROM date_vente) = 2024;
```

**Pourquoi function-based index ?**
- Sans index : Oracle doit appliquer la fonction à chaque ligne (lent)
- Avec index : Résultat précalculé et indexé (rapide)

**Exemple concret : Email insensible à la casse**

```sql
-- Table avec emails
CREATE TABLE utilisateurs (
    user_id NUMBER PRIMARY KEY,
    email VARCHAR2(100)
);

-- Index function-based
CREATE UNIQUE INDEX idx_user_email_lower ON utilisateurs(LOWER(email));

-- Insertion : email en minuscules stocké dans l'index
INSERT INTO utilisateurs VALUES (1, 'Jean.Dupont@Example.COM');

-- Recherche insensible à la casse (utilise l'index)
SELECT * FROM utilisateurs WHERE LOWER(email) = 'jean.dupont@example.com';

-- Bonus : Unicité insensible à la casse
-- Cette insertion échouera (doublon détecté par l'index)
INSERT INTO utilisateurs VALUES (2, 'jean.dupont@EXAMPLE.com');
-- ORA-00001: unique constraint violated
```

#### 8.2.4 Index sur Colonnes Virtuelles 🟢 **11g+**

Combinaison puissante : colonne virtuelle + index.

```sql
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    prenom VARCHAR2(50),
    nom VARCHAR2(50),
    
    -- Colonne virtuelle
    nom_complet VARCHAR2(101) GENERATED ALWAYS AS (prenom || ' ' || nom) VIRTUAL
);

-- Index sur colonne virtuelle
CREATE INDEX idx_emp_nom_complet ON employes(nom_complet);

-- Requête optimisée
SELECT * FROM employes WHERE nom_complet = 'Jean Dupont';
```

### 8.3 Stratégies d'Indexation

#### 8.3.1 Index sur Clés Étrangères 🟢 **11g+**

**Règle d'or : Toujours indexer les foreign keys**

```sql
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,  -- Index automatique sur PK
    dept_id NUMBER,
    manager_id NUMBER,
    CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) REFERENCES departements(dept_id),
    CONSTRAINT fk_emp_manager FOREIGN KEY (manager_id) REFERENCES employes(employe_id)
);

-- ⚠️ Oracle ne crée PAS automatiquement d'index sur les FK
-- Il faut les créer manuellement pour les performances

CREATE INDEX idx_emp_dept ON employes(dept_id);
CREATE INDEX idx_emp_manager ON employes(manager_id);
```

**Pourquoi indexer les FK ?**
1. **Performances JOIN**
   ```sql
   -- Sans index sur dept_id : nested loops lent
   SELECT e.nom, d.nom 
   FROM employes e 
   JOIN departements d ON e.dept_id = d.dept_id;
   ```

2. **Éviter les locks lors de DELETE parent**
   ```sql
   -- Sans index : Oracle lock toute la table employes
   DELETE FROM departements WHERE dept_id = 10;
   
   -- Avec index : Oracle lock uniquement les lignes concernées
   ```

3. **Performances ON DELETE CASCADE**

#### 8.3.2 Index Composites : Ordre Optimal

**Règle : Colonne la plus sélective en premier**

```sql
-- Exemple : Table avec 1M de lignes
-- dept_id : 50 départements (faible sélectivité)
-- nom : 500 000 noms distincts (forte sélectivité)

-- ❌ Mauvais ordre
CREATE INDEX idx_bad ON employes(dept_id, nom);

-- ✅ Bon ordre
CREATE INDEX idx_good ON employes(nom, dept_id);

-- Requête fréquente
SELECT * FROM employes WHERE nom = 'Dupont' AND dept_id = 10;

-- idx_good filtre d'abord par nom (500 000 valeurs possibles)
-- puis par dept_id (50 valeurs possibles)
-- Beaucoup plus sélectif !
```

**Cas particulier : Requêtes variées**

```sql
-- Requêtes métier
-- Q1 : Recherche par département
SELECT * FROM employes WHERE dept_id = 10;

-- Q2 : Recherche par nom
SELECT * FROM employes WHERE nom = 'Dupont';

-- Q3 : Recherche combinée
SELECT * FROM employes WHERE dept_id = 10 AND nom = 'Dupont';

-- Solution : Deux index
CREATE INDEX idx_emp_dept ON employes(dept_id);     -- Pour Q1 et Q3
CREATE INDEX idx_emp_nom ON employes(nom);          -- Pour Q2 et Q3

-- Oracle choisira automatiquement le meilleur index
```

#### 8.3.3 Covering Index (Index couvrant)

Index qui contient toutes les colonnes nécessaires à une requête.

```sql
-- Requête fréquente
SELECT employe_id, nom, dept_id FROM employes WHERE dept_id = 10;

-- Index couvrant : contient dept_id, employe_id, nom
CREATE INDEX idx_emp_covering ON employes(dept_id, employe_id, nom);

-- Oracle peut répondre UNIQUEMENT avec l'index, sans accéder à la table
-- = Index Only Scan (très rapide)
```

**Différence 11g vs 19c : Index Only Scan**
- **11g** : Moins d'optimisations automatiques
- **19c** : Cost-based optimizer (CBO) plus intelligent, détecte mieux les index couvrants

### 8.4 Gestion et Maintenance des Index

#### 8.4.1 Visualiser les Index 🟢 **11g+**

```sql
-- Lister tous les index d'une table
SELECT index_name, column_name, column_position
FROM user_ind_columns
WHERE table_name = 'EMPLOYES'
ORDER BY index_name, column_position;

-- Statistiques d'un index
SELECT index_name, blevel, leaf_blocks, distinct_keys, num_rows
FROM user_indexes
WHERE table_name = 'EMPLOYES';

-- Taille d'un index en Mo
SELECT segment_name, ROUND(bytes/1024/1024, 2) AS size_mb
FROM user_segments
WHERE segment_name = 'IDX_EMP_NOM' AND segment_type = 'INDEX';
```

#### 8.4.2 Identifier les Index Inutilisés 🔵 **19c amélioré**

**Oracle 11g : Monitoring manuel**
```sql
-- Activer le monitoring sur un index
ALTER INDEX idx_emp_nom MONITORING USAGE;

-- Attendre plusieurs jours/semaines...

-- Vérifier l'utilisation
SELECT index_name, table_name, monitoring, used
FROM v$object_usage
WHERE index_name = 'IDX_EMP_NOM';

-- Désactiver le monitoring
ALTER INDEX idx_emp_nom NOMONITORING USAGE;
```

**Oracle 19c : Automatic Indexing (Autonomous Database)** 🔵 **19c+**
```sql
-- 19c peut suggérer/créer/supprimer automatiquement des index
-- (Fonctionnalité Autonomous Database ou activation manuelle)

SELECT * FROM dba_auto_index_ind_actions;
```

#### 8.4.3 Reconstruire les Index 🟢 **11g+**

Les index fragmentés ralentissent les requêtes. Reconstruction périodique nécessaire.

```sql
-- Vérifier la fragmentation
SELECT index_name, blevel, pct_used
FROM user_indexes
WHERE table_name = 'EMPLOYES';

-- blevel > 3 ou 4 : index fragmenté
-- pct_used < 75% : beaucoup d'espace gaspillé

-- Reconstruire un index (OFFLINE : table locked)
ALTER INDEX idx_emp_nom REBUILD;

-- Reconstruction ONLINE (pas de lock table)
ALTER INDEX idx_emp_nom REBUILD ONLINE;

-- Reconstruire tous les index d'une table
BEGIN
    FOR idx IN (SELECT index_name FROM user_indexes WHERE table_name = 'EMPLOYES') LOOP
        EXECUTE IMMEDIATE 'ALTER INDEX ' || idx.index_name || ' REBUILD ONLINE';
    END LOOP;
END;
/
```

**Quand reconstruire ?**
- Après de grosses suppressions/mises à jour
- blevel > 4
- Baisse de performances constatée
- Recommandation : Mensuel ou trimestriel selon volumétrie

#### 8.4.4 Supprimer un Index

```sql
-- Supprimer un index
DROP INDEX idx_emp_nom;

-- ⚠️ Ne jamais supprimer un index sans analyser l'impact
-- 1. Vérifier qu'il n'est pas utilisé (v$object_usage)
-- 2. Tester les performances sans l'index sur environnement de test
```

### 8.5 Index et Performances : Cas Pratiques

#### Cas 1 : Recherche textuelle

```sql
-- Problème : Recherche "commence par" vs "contient"

-- Table de produits
CREATE TABLE produits (
    produit_id NUMBER PRIMARY KEY,
    nom VARCHAR2(200)
);

CREATE INDEX idx_prod_nom ON produits(nom);

-- ✅ Utilise l'index (commence par)
SELECT * FROM produits WHERE nom LIKE 'Sony%';

-- ❌ N'utilise PAS l'index (contient)
SELECT * FROM produits WHERE nom LIKE '%Sony%';

-- Solution pour "contient" : Oracle Text (Full-Text Search)
-- Disponible en 11g et 19c (Enterprise Edition)
CREATE INDEX idx_prod_nom_text ON produits(nom) 
    INDEXTYPE IS CTXSYS.CONTEXT;

-- Recherche full-text
SELECT * FROM produits WHERE CONTAINS(nom, 'Sony') > 0;
```

#### Cas 2 : NULL et Index

```sql
-- ⚠️ Les index B-Tree n'indexent PAS les valeurs NULL

CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    email VARCHAR2(100),
    telephone VARCHAR2(20)
);

CREATE INDEX idx_emp_telephone ON employes(telephone);

-- ❌ N'utilise PAS l'index (recherche de NULL)
SELECT * FROM employes WHERE telephone IS NULL;
-- Full table scan

-- Solution : Index composite avec une constante
CREATE INDEX idx_emp_telephone_null ON employes(telephone, 0);

-- Ou : Utiliser une colonne virtuelle avec valeur par défaut
ALTER TABLE employes ADD (
    telephone_idx VARCHAR2(20) GENERATED ALWAYS AS (NVL(telephone, 'NULL')) VIRTUAL
);
CREATE INDEX idx_emp_telephone_idx ON employes(telephone_idx);
```

#### Cas 3 : Index et Statistiques

**Les statistiques sont CRITIQUES pour le Cost-Based Optimizer**

```sql
-- Collecter les statistiques (11g et 19c)
-- Après création d'index ou grosses modifications de données

-- Table spécifique
EXEC DBMS_STATS.GATHER_TABLE_STATS('MON_SCHEMA', 'EMPLOYES');

-- Toutes les tables du schéma
EXEC DBMS_STATS.GATHER_SCHEMA_STATS('MON_SCHEMA');

-- Vérifier la date des dernières statistiques
SELECT table_name, last_analyzed, num_rows
FROM user_tables
WHERE table_name = 'EMPLOYES';
```

**Différence 11g vs 19c :**
- **11g** : Collecte automatique chaque nuit (DBMS_SCHEDULER job)
- **19c** : Collecte automatique améliorée + statistiques en temps réel (Real-Time Statistics)

### 8.6 Bonnes Pratiques Index

**✅ À FAIRE :**
1. Indexer TOUTES les foreign keys
2. Indexer les colonnes fréquentes dans WHERE/JOIN
3. Nommer les index explicitement (idx_table_colonnes)
4. Reconstruire périodiquement les index fragmentés
5. Collecter les statistiques après modifications massives

**❌ À ÉVITER :**
1. Sur-indexer (trop d'index ralentit les écritures)
2. Indexer les tables < 100 lignes (inutile)
3. Indexer les colonnes jamais utilisées dans WHERE/JOIN
4. Oublier de monitorer l'utilisation des index
5. Dupliquer les index (ex: (col1) et (col1, col2))

**Règle empirique :**
- Tables < 10 000 lignes : Peu d'index nécessaires (full scan rapide)
- Tables 10K-1M lignes : Index sur FK + colonnes critiques
- Tables > 1M lignes : Index stratégiques + partitionnement

---

## 9. Triggers : Automatisation et Logique Métier

### 9.1 Introduction aux Triggers 🟢 **11g+**

Un trigger est un bloc PL/SQL qui s'exécute automatiquement en réponse à un événement (INSERT, UPDATE, DELETE).

**Pourquoi utiliser des triggers ?**
- Audit automatique (qui, quand, quoi)
- Validation de règles métier complexes
- Calculs automatiques
- Synchronisation entre tables
- Historisation

**Types de triggers :**
1. **BEFORE** : Avant l'opération (peut modifier :NEW)
2. **AFTER** : Après l'opération
3. **INSTEAD OF** : Remplace l'opération (pour vues)

**Niveaux :**
1. **Statement-level** : Une fois par commande SQL
2. **Row-level** : Une fois par ligne affectée (FOR EACH ROW)

### 9.2 Syntaxe de Base

```sql
CREATE [OR REPLACE] TRIGGER nom_trigger
{BEFORE | AFTER | INSTEAD OF} {INSERT | UPDATE | DELETE} [OR {INSERT | UPDATE | DELETE}]
ON nom_table
[FOR EACH ROW]
[WHEN (condition)]
DECLARE
    -- Variables locales
BEGIN
    -- Code PL/SQL
    -- :NEW.colonne = nouvelle valeur (INSERT/UPDATE)
    -- :OLD.colonne = ancienne valeur (UPDATE/DELETE)
END;
/
```

### 9.3 Cas d'Usage Pratiques

#### 9.3.1 Audit Automatique (Traçabilité) 🟢 **11g+**

**Cas le plus fréquent : Remplir automatiquement les colonnes d'audit**

```sql
-- Table avec colonnes d'audit
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL,
    prenom VARCHAR2(50) NOT NULL,
    salaire NUMBER(10,2),
    
    -- Colonnes d'audit
    cree_par VARCHAR2(50),
    cree_le TIMESTAMP,
    modifie_par VARCHAR2(50),
    modifie_le TIMESTAMP
);

-- Trigger BEFORE INSERT : Remplir cree_par et cree_le
CREATE OR REPLACE TRIGGER trg_employes_insert
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    :NEW.cree_par := USER;  -- Utilisateur Oracle connecté
    :NEW.cree_le := CURRENT_TIMESTAMP;
END;
/

-- Trigger BEFORE UPDATE : Remplir modifie_par et modifie_le
CREATE OR REPLACE TRIGGER trg_employes_update
BEFORE UPDATE ON employes
FOR EACH ROW
BEGIN
    :NEW.modifie_par := USER;
    :NEW.modifie_le := CURRENT_TIMESTAMP;
    
    -- Conserver les valeurs de création (ne pas écraser)
    :NEW.cree_par := :OLD.cree_par;
    :NEW.cree_le := :OLD.cree_le;
END;
/

-- Test
INSERT INTO employes (employe_id, nom, prenom, salaire) 
VALUES (1, 'Dupont', 'Jean', 50000);

SELECT cree_par, cree_le FROM employes WHERE employe_id = 1;
-- cree_par = 'MON_USER', cree_le = timestamp actuel

UPDATE employes SET salaire = 55000 WHERE employe_id = 1;

SELECT modifie_par, modifie_le FROM employes WHERE employe_id = 1;
-- modifie_par = 'MON_USER', modifie_le = timestamp actuel
```

#### 9.3.2 Auto-Increment avec Séquence (Oracle 11g) 🟢 **11g**

Nous avons déjà vu cet exemple en section 5.2, voici une version plus complète :

```sql
-- 1. Créer la table
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL,
    email VARCHAR2(100)
);

-- 2. Créer la séquence
CREATE SEQUENCE seq_employe_id START WITH 1 INCREMENT BY 1;

-- 3. Trigger pour auto-increment
CREATE OR REPLACE TRIGGER trg_employe_id
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    -- Si l'ID est NULL, utiliser la séquence
    IF :NEW.employe_id IS NULL THEN
        :NEW.employe_id := seq_employe_id.NEXTVAL;
    END IF;
    
    -- Validation : Email unique en minuscules
    :NEW.email := LOWER(:NEW.email);
END;
/

-- Insertion sans ID (auto-généré)
INSERT INTO employes (nom, email) VALUES ('Dupont', 'Jean.Dupont@Example.COM');
-- employe_id = 1, email = 'jean.dupont@example.com'

INSERT INTO employes (nom, email) VALUES ('Martin', 'Sophie.Martin@Test.FR');
-- employe_id = 2, email = 'sophie.martin@test.fr'
```

#### 9.3.3 Validation de Règles Métier Complexes 🟢 **11g+**

**CHECK contraintes limitées → Triggers pour logique avancée**

```sql
-- Règle métier : Un employé ne peut pas gagner plus de 3x le salaire moyen de son département

CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50),
    dept_id NUMBER,
    salaire NUMBER(10,2)
);

CREATE OR REPLACE TRIGGER trg_verif_salaire
BEFORE INSERT OR UPDATE OF salaire, dept_id ON employes
FOR EACH ROW
DECLARE
    v_avg_salaire NUMBER;
    v_max_salaire NUMBER;
BEGIN
    -- Calculer le salaire moyen du département
    SELECT AVG(salaire) INTO v_avg_salaire
    FROM employes
    WHERE dept_id = :NEW.dept_id
      AND employe_id != :NEW.employe_id;  -- Exclure l'employé en cours
    
    v_max_salaire := v_avg_salaire * 3;
    
    -- Si le salaire dépasse le maximum autorisé, lever une erreur
    IF :NEW.salaire > v_max_salaire THEN
        RAISE_APPLICATION_ERROR(-20001, 
            'Salaire ' || :NEW.salaire || ' dépasse le maximum autorisé ' || 
            ROUND(v_max_salaire, 2) || ' (3x le salaire moyen du département)');
    END IF;
END;
/

-- Test
INSERT INTO employes VALUES (1, 'Dupont', 10, 50000);  -- OK
INSERT INTO employes VALUES (2, 'Martin', 10, 60000);  -- OK
INSERT INTO employes VALUES (3, 'Durand', 10, 200000); -- ERREUR
-- ORA-20001: Salaire 200000 dépasse le maximum autorisé 165000
```

#### 9.3.4 Historisation Automatique 🟢 **11g+**

**Conserver l'historique de toutes les modifications**

```sql
-- Table principale
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50),
    salaire NUMBER(10,2),
    actif CHAR(1) DEFAULT 'Y'
);

-- Table d'historique
CREATE TABLE employes_historique (
    hist_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- 19c
    employe_id NUMBER,
    nom VARCHAR2(50),
    salaire NUMBER(10,2),
    actif CHAR(1),
    operation VARCHAR2(10),  -- 'INSERT', 'UPDATE', 'DELETE'
    user_operation VARCHAR2(50),
    date_operation TIMESTAMP
);

-- Version 11g : Utiliser séquence pour hist_id
CREATE SEQUENCE seq_hist_id START WITH 1;

CREATE TABLE employes_historique (
    hist_id NUMBER PRIMARY KEY,  -- Rempli par trigger
    employe_id NUMBER,
    nom VARCHAR2(50),
    salaire NUMBER(10,2),
    actif CHAR(1),
    operation VARCHAR2(10),
    user_operation VARCHAR2(50),
    date_operation TIMESTAMP
);

-- Trigger d'historisation (INSERT)
CREATE OR REPLACE TRIGGER trg_emp_hist_insert
AFTER INSERT ON employes
FOR EACH ROW
BEGIN
    INSERT INTO employes_historique (
        hist_id,  -- 11g : utiliser séquence
        employe_id, nom, salaire, actif, 
        operation, user_operation, date_operation
    ) VALUES (
        seq_hist_id.NEXTVAL,  -- 11g : séquence manuelle
        :NEW.employe_id, :NEW.nom, :NEW.salaire, :NEW.actif,
        'INSERT', USER, CURRENT_TIMESTAMP
    );
END;
/

-- Trigger d'historisation (UPDATE)
CREATE OR REPLACE TRIGGER trg_emp_hist_update
AFTER UPDATE ON employes
FOR EACH ROW
BEGIN
    INSERT INTO employes_historique (
        hist_id,
        employe_id, nom, salaire, actif,
        operation, user_operation, date_operation
    ) VALUES (
        seq_hist_id.NEXTVAL,
        :NEW.employe_id, :NEW.nom, :NEW.salaire, :NEW.actif,
        'UPDATE', USER, CURRENT_TIMESTAMP
    );
END;
/

-- Trigger d'historisation (DELETE)
CREATE OR REPLACE TRIGGER trg_emp_hist_delete
AFTER DELETE ON employes
FOR EACH ROW
BEGIN
    INSERT INTO employes_historique (
        hist_id,
        employe_id, nom, salaire, actif,
        operation, user_operation, date_operation
    ) VALUES (
        seq_hist_id.NEXTVAL,
        :OLD.employe_id, :OLD.nom, :OLD.salaire, :OLD.actif,
        'DELETE', USER, CURRENT_TIMESTAMP
    );
END;
/

-- Test
INSERT INTO employes VALUES (1, 'Dupont', 50000, 'Y');
UPDATE employes SET salaire = 55000 WHERE employe_id = 1;
DELETE FROM employes WHERE employe_id = 1;

-- Consulter l'historique
SELECT * FROM employes_historique WHERE employe_id = 1 ORDER BY date_operation;
-- 3 lignes : INSERT (50000), UPDATE (55000), DELETE (55000)
```

**Amélioration 19c : Trigger unique pour toutes les opérations**

```sql
-- 19c : Compound trigger (plus performant)
CREATE OR REPLACE TRIGGER trg_emp_hist_all
FOR INSERT OR UPDATE OR DELETE ON employes
COMPOUND TRIGGER

    -- Variables partagées
    TYPE t_hist_tab IS TABLE OF employes_historique%ROWTYPE;
    g_hist_tab t_hist_tab := t_hist_tab();
    
    AFTER EACH ROW IS
    BEGIN
        g_hist_tab.EXTEND;
        g_hist_tab(g_hist_tab.COUNT).employe_id := COALESCE(:NEW.employe_id, :OLD.employe_id);
        g_hist_tab(g_hist_tab.COUNT).nom := COALESCE(:NEW.nom, :OLD.nom);
        g_hist_tab(g_hist_tab.COUNT).salaire := COALESCE(:NEW.salaire, :OLD.salaire);
        g_hist_tab(g_hist_tab.COUNT).actif := COALESCE(:NEW.actif, :OLD.actif);
        g_hist_tab(g_hist_tab.COUNT).user_operation := USER;
        g_hist_tab(g_hist_tab.COUNT).date_operation := CURRENT_TIMESTAMP;
        
        IF INSERTING THEN
            g_hist_tab(g_hist_tab.COUNT).operation := 'INSERT';
        ELSIF UPDATING THEN
            g_hist_tab(g_hist_tab.COUNT).operation := 'UPDATE';
        ELSIF DELETING THEN
            g_hist_tab(g_hist_tab.COUNT).operation := 'DELETE';
        END IF;
    END AFTER EACH ROW;
    
    AFTER STATEMENT IS
    BEGIN
        -- Insertion en bulk (plus performant)
        FORALL i IN 1..g_hist_tab.COUNT
            INSERT INTO employes_historique VALUES g_hist_tab(i);
        g_hist_tab.DELETE;
    END AFTER STATEMENT;
    
END trg_emp_hist_all;
/
```

#### 9.3.5 Synchronisation entre Tables 🟢 **11g+**

**Maintenir automatiquement une table dénormalisée ou cache**

```sql
-- Table de commandes
CREATE TABLE commandes (
    commande_id NUMBER PRIMARY KEY,
    client_id NUMBER,
    montant_total NUMBER(10,2) DEFAULT 0
);

-- Table de lignes de commandes
CREATE TABLE lignes_commande (
    ligne_id NUMBER PRIMARY KEY,
    commande_id NUMBER,
    produit_id NUMBER,
    quantite NUMBER,
    prix_unitaire NUMBER(10,2),
    CONSTRAINT fk_ligne_cmd FOREIGN KEY (commande_id) REFERENCES commandes(commande_id)
);

-- Trigger : Mettre à jour automatiquement le montant_total de la commande
CREATE OR REPLACE TRIGGER trg_maj_montant_commande
AFTER INSERT OR UPDATE OR DELETE ON lignes_commande
FOR EACH ROW
DECLARE
    v_commande_id NUMBER;
    v_nouveau_total NUMBER;
BEGIN
    -- Identifier la commande concernée
    v_commande_id := COALESCE(:NEW.commande_id, :OLD.commande_id);
    
    -- Calculer le nouveau total
    SELECT SUM(quantite * prix_unitaire) INTO v_nouveau_total
    FROM lignes_commande
    WHERE commande_id = v_commande_id;
    
    -- Mettre à jour la commande
    UPDATE commandes
    SET montant_total = NVL(v_nouveau_total, 0)
    WHERE commande_id = v_commande_id;
END;
/

-- Test
INSERT INTO commandes (commande_id, client_id) VALUES (1, 100);
-- montant_total = 0

INSERT INTO lignes_commande VALUES (1, 1, 501, 2, 50.00);  -- 2 x 50 = 100
INSERT INTO lignes_commande VALUES (2, 1, 502, 1, 30.00);  -- 1 x 30 = 30

SELECT montant_total FROM commandes WHERE commande_id = 1;
-- montant_total = 130.00 (auto-calculé)

UPDATE lignes_commande SET quantite = 5 WHERE ligne_id = 1;  -- 5 x 50 = 250

SELECT montant_total FROM commandes WHERE commande_id = 1;
-- montant_total = 280.00 (250 + 30)
```

#### 9.3.6 Empêcher des Opérations 🟢 **11g+**

**Interdire la suppression ou modification dans certains cas**

```sql
-- Empêcher la suppression d'un département si des employés y sont affectés

CREATE OR REPLACE TRIGGER trg_prevent_dept_delete
BEFORE DELETE ON departements
FOR EACH ROW
DECLARE
    v_count NUMBER;
BEGIN
    -- Compter les employés du département
    SELECT COUNT(*) INTO v_count
    FROM employes
    WHERE dept_id = :OLD.dept_id;
    
    IF v_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20002, 
            'Impossible de supprimer le département ' || :OLD.dept_id || 
            ' : ' || v_count || ' employé(s) y sont affectés');
    END IF;
END;
/

-- Test
DELETE FROM departements WHERE dept_id = 10;
-- ORA-20002: Impossible de supprimer le département 10 : 5 employé(s) y sont affectés
```

### 9.4 Triggers sur Vues (INSTEAD OF) 🟢 **11g+**

**Permettre INSERT/UPDATE/DELETE sur des vues complexes (non modifiables par défaut)**

```sql
-- Vue complexe (non modifiable directement)
CREATE VIEW v_employes_complet AS
SELECT e.employe_id, e.nom, e.prenom, e.salaire,
       d.dept_id, d.nom AS dept_nom
FROM employes e
JOIN departements d ON e.dept_id = d.dept_id;

-- Essayer d'insérer dans la vue : ERREUR
INSERT INTO v_employes_complet (employe_id, nom, prenom, salaire, dept_id) 
VALUES (10, 'Test', 'User', 50000, 1);
-- ORA-01779: cannot modify a column which maps to a non key-preserved table

-- Solution : INSTEAD OF trigger
CREATE OR REPLACE TRIGGER trg_insert_emp_complet
INSTEAD OF INSERT ON v_employes_complet
FOR EACH ROW
BEGIN
    -- Insérer dans la table sous-jacente
    INSERT INTO employes (employe_id, nom, prenom, salaire, dept_id)
    VALUES (:NEW.employe_id, :NEW.nom, :NEW.prenom, :NEW.salaire, :NEW.dept_id);
END;
/

-- Maintenant l'insertion fonctionne
INSERT INTO v_employes_complet (employe_id, nom, prenom, salaire, dept_id) 
VALUES (10, 'Test', 'User', 50000, 1);
-- OK !
```

### 9.5 Gestion des Triggers

#### 9.5.1 Lister les Triggers 🟢 **11g+**

```sql
-- Tous les triggers de l'utilisateur
SELECT trigger_name, table_name, triggering_event, status
FROM user_triggers
ORDER BY table_name, trigger_name;

-- Détail d'un trigger
SELECT trigger_body FROM user_triggers WHERE trigger_name = 'TRG_EMPLOYES_INSERT';

-- Ou utiliser DBMS_METADATA
SELECT DBMS_METADATA.GET_DDL('TRIGGER', 'TRG_EMPLOYES_INSERT') FROM DUAL;
```

#### 9.5.2 Désactiver/Activer les Triggers 🟢 **11g+**

```sql
-- Désactiver un trigger (temporairement)
ALTER TRIGGER trg_employes_insert DISABLE;

-- Activer un trigger
ALTER TRIGGER trg_employes_insert ENABLE;

-- Désactiver TOUS les triggers d'une table
ALTER TABLE employes DISABLE ALL TRIGGERS;

-- Activer TOUS les triggers d'une table
ALTER TABLE employes ENABLE ALL TRIGGERS;
```

**Cas d'usage :**
- Import massif de données (désactiver pour performance)
- Debugging (isoler le problème)
- Migration de données

#### 9.5.3 Supprimer un Trigger 🟢 **11g+**

```sql
DROP TRIGGER trg_employes_insert;
```

### 9.6 Ordre d'Exécution des Triggers

**Si plusieurs triggers sur le même événement :**

```sql
-- Oracle 11g : Ordre indéterminé (problème !)
-- Oracle 11g : Utiliser FOLLOWS clause (11g Release 2+)

CREATE OR REPLACE TRIGGER trg_employes_audit
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    :NEW.cree_le := SYSDATE;
END;
/

CREATE OR REPLACE TRIGGER trg_employes_validation
BEFORE INSERT ON employes
FOR EACH ROW
FOLLOWS trg_employes_audit  -- S'exécute APRÈS trg_employes_audit
BEGIN
    IF :NEW.salaire < 0 THEN
        RAISE_APPLICATION_ERROR(-20003, 'Salaire négatif interdit');
    END IF;
END;
/
```

**🔵 Oracle 19c : FOLLOWS clause disponible** (introduite en 11g R2)

### 9.7 Performances des Triggers

#### ⚠️ Pièges de Performance

```sql
-- ❌ MAUVAIS : Requête dans trigger row-level
CREATE OR REPLACE TRIGGER trg_slow
AFTER INSERT ON employes
FOR EACH ROW  -- Exécuté pour CHAQUE ligne
DECLARE
    v_count NUMBER;
BEGIN
    -- Requête lente exécutée N fois
    SELECT COUNT(*) INTO v_count FROM grosse_table WHERE condition = :NEW.dept_id;
END;
/

-- Si on insère 10 000 lignes : 10 000 exécutions de la requête !

-- ✅ MEILLEUR : Statement-level trigger avec BULK COLLECT
CREATE OR REPLACE TRIGGER trg_fast
AFTER INSERT ON employes
-- Pas de FOR EACH ROW : exécuté UNE FOIS pour tout l'INSERT
DECLARE
    v_result NUMBER;
BEGIN
    -- Traitement global, une seule fois
    SELECT COUNT(*) INTO v_result FROM grosse_table;
    -- ...
END;
/
```

#### 🔵 Oracle 19c : Compound Triggers

**Meilleure performance avec bulk operations**

```sql
CREATE OR REPLACE TRIGGER trg_emp_bulk
FOR INSERT ON employes
COMPOUND TRIGGER

    -- Collection pour stocker temporairement
    TYPE t_emp_tab IS TABLE OF NUMBER;
    g_emp_ids t_emp_tab := t_emp_tab();
    
    AFTER EACH ROW IS
    BEGIN
        -- Collecter les IDs
        g_emp_ids.EXTEND;
        g_emp_ids(g_emp_ids.COUNT) := :NEW.employe_id;
    END AFTER EACH ROW;
    
    AFTER STATEMENT IS
    BEGIN
        -- Traitement en bulk (une seule fois)
        FORALL i IN 1..g_emp_ids.COUNT
            INSERT INTO audit_table (employe_id, action_date)
            VALUES (g_emp_ids(i), SYSDATE);
        
        g_emp_ids.DELETE;
    END AFTER STATEMENT;
    
END trg_emp_bulk;
/
```

### 9.8 Bonnes Pratiques Triggers

**✅ À FAIRE :**
1. Nommer explicitement (trg_table_action)
2. Documenter la logique (commentaires)
3. Utiliser RAISE_APPLICATION_ERROR pour erreurs claires
4. Tester exhaustivement (INSERT, UPDATE, DELETE)
5. Éviter les requêtes coûteuses en row-level triggers

**❌ À ÉVITER :**
1. Triggers trop complexes (extraire en procédures stockées)
2. Triggers qui modifient d'autres tables (cascade de triggers = difficulté debugging)
3. COMMIT/ROLLBACK dans un trigger (interdit en row-level)
4. Triggers cachés (documenter l'existence dans le code application)
5. Dépendances circulaires entre triggers

**⚠️ Debugging :**

```sql
-- Tracer les triggers en problème
-- Ajouter des logs dans une table
CREATE TABLE trigger_logs (
    log_id NUMBER PRIMARY KEY,
    trigger_name VARCHAR2(50),
    message VARCHAR2(500),
    log_date TIMESTAMP
);

CREATE SEQUENCE seq_log_id;

-- Dans le trigger
CREATE OR REPLACE TRIGGER trg_employes_debug
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    -- Log pour debugging
    INSERT INTO trigger_logs VALUES (
        seq_log_id.NEXTVAL,
        'TRG_EMPLOYES_DEBUG',
        'Insert employe ' || :NEW.nom,
        CURRENT_TIMESTAMP
    );
    
    -- Logique du trigger
    :NEW.cree_le := SYSDATE;
    
    -- Log de fin
    INSERT INTO trigger_logs VALUES (
        seq_log_id.NEXTVAL,
        'TRG_EMPLOYES_DEBUG',
        'Fin insert employe ' || :NEW.nom,
        CURRENT_TIMESTAMP
    );
END;
/

-- Consulter les logs
SELECT * FROM trigger_logs ORDER BY log_date DESC;
```

---

## 10. Bonnes Pratiques Globales

### 10.1 Convention de Nommage 🟢 **11g+**

```sql
-- Tables : nom_pluriel ou nom_entite
CREATE TABLE employes (...);
CREATE TABLE client_commandes (...);

-- Colonnes : nom_descriptif
CREATE TABLE employes (
    employe_id NUMBER,        -- PK : entite_id
    dept_id NUMBER,           -- FK : entite_id (référence)
    date_embauche DATE,       -- Préfixe type pour clarté
    salaire_brut NUMBER
);

-- Contraintes : type_table_colonne
CONSTRAINT pk_employes PRIMARY KEY (employe_id)
CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) ...
CONSTRAINT uk_emp_email UNIQUE (email)
CONSTRAINT ck_emp_age CHECK (age >= 18)

-- Index : idx_table_colonnes
CREATE INDEX idx_emp_nom ON employes(nom);
CREATE INDEX idx_emp_dept_nom ON employes(dept_id, nom);

-- Triggers : trg_table_action
CREATE TRIGGER trg_employes_insert ...
CREATE TRIGGER trg_employes_audit ...

-- Séquences (11g) : seq_table_colonne
CREATE SEQUENCE seq_employe_id;
```

### 10.2 Choix des Types de Données 🟢 **11g+**

**Principe : Type le plus restrictif possible**

```sql
-- Mauvais
age VARCHAR2(100)           -- Permet 'abc', gaspille espace

-- Bon
age NUMBER(3) CHECK (age BETWEEN 0 AND 120)

-- Mauvais
actif VARCHAR2(10)          -- Permet n'importe quoi

-- Bon
actif CHAR(1) CHECK (actif IN ('Y', 'N'))
```

### 10.3 Gestion des NULL 🟢 **11g+**

```sql
-- Pièges avec NULL
SELECT * FROM employes WHERE manager_id != 100;
-- Ne retourne PAS les lignes avec manager_id = NULL !

-- Solution : être explicite
SELECT * FROM employes WHERE manager_id != 100 OR manager_id IS NULL;

-- Ou éviter NULL avec DEFAULT
CREATE TABLE employes (
    employe_id NUMBER,
    manager_id NUMBER DEFAULT -1,  -- -1 = pas de manager
    CONSTRAINT ck_manager CHECK (manager_id = -1 OR manager_id > 0)
);
```

### 10.4 Audit et Traçabilité 🟢 **11g+**

```sql
-- Colonnes d'audit standard
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50) NOT NULL,
    -- ... autres colonnes métier ...
    
    -- Audit
    cree_par VARCHAR2(50) DEFAULT USER NOT NULL,
    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    modifie_par VARCHAR2(50),
    modifie_le TIMESTAMP
);

-- Trigger de mise à jour automatique (voir section Triggers)
```

### 10.5 Suppression Logique vs Physique 🟢 **11g+**

```sql
-- Suppression physique : DELETE (perte définitive)
DELETE FROM employes WHERE employe_id = 100;

-- Suppression logique : UPDATE (historisation)
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    nom VARCHAR2(50),
    actif CHAR(1) DEFAULT 'Y',
    date_fin_validite DATE,
    CONSTRAINT ck_emp_actif CHECK (actif IN ('Y', 'N'))
);

-- "Suppression"
UPDATE employes 
SET actif = 'N', date_fin_validite = SYSDATE 
WHERE employe_id = 100;

-- Vue des employés actifs
CREATE VIEW v_employes_actifs AS
SELECT * FROM employes WHERE actif = 'Y';
```

---

## 11. Exemple Complet : Système de Gestion de Projet

### Version Oracle 11g

```sql
-- ==========================================
-- SÉQUENCES (pour auto-increment en 11g)
-- ==========================================
CREATE SEQUENCE seq_dept_id START WITH 1;
CREATE SEQUENCE seq_employe_id START WITH 1;
CREATE SEQUENCE seq_projet_id START WITH 1;
CREATE SEQUENCE seq_affectation_id START WITH 1;
CREATE SEQUENCE seq_log_id START WITH 1;

-- ==========================================
-- 1. TABLE DÉPARTEMENTS
-- ==========================================
CREATE TABLE departements (
    dept_id NUMBER PRIMARY KEY,
    nom VARCHAR2(100) NOT NULL,
    budget NUMBER(12,2) CHECK (budget >= 0),
    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_dept_nom UNIQUE (nom)
);

-- Trigger auto-increment 11g
CREATE OR REPLACE TRIGGER trg_dept_id
BEFORE INSERT ON departements
FOR EACH ROW
BEGIN
    IF :NEW.dept_id IS NULL THEN
        :NEW.dept_id := seq_dept_id.NEXTVAL;
    END IF;
END;
/

-- ==========================================
-- 2. TABLE EMPLOYÉS
-- ==========================================
CREATE TABLE employes (
    employe_id NUMBER PRIMARY KEY,
    dept_id NUMBER NOT NULL,
    manager_id NUMBER,
    nom VARCHAR2(50) NOT NULL,
    prenom VARCHAR2(50) NOT NULL,
    email VARCHAR2(100) NOT NULL,
    telephone VARCHAR2(20),
    date_embauche DATE DEFAULT TRUNC(SYSDATE) NOT NULL,
    salaire NUMBER(10,2) NOT NULL,
    actif CHAR(1) DEFAULT 'Y' NOT NULL,
    cree_par VARCHAR2(50) DEFAULT USER,
    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modifie_par VARCHAR2(50),
    modifie_le TIMESTAMP,
    
    CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) REFERENCES departements(dept_id),
    CONSTRAINT fk_emp_manager FOREIGN KEY (manager_id) REFERENCES employes(employe_id) ON DELETE SET NULL,
    CONSTRAINT uk_emp_email UNIQUE (email),
    CONSTRAINT ck_emp_salaire CHECK (salaire > 0),
    CONSTRAINT ck_emp_actif CHECK (actif IN ('Y', 'N'))
);

-- Trigger auto-increment
CREATE OR REPLACE TRIGGER trg_employe_id
BEFORE INSERT ON employes
FOR EACH ROW
BEGIN
    IF :NEW.employe_id IS NULL THEN
        :NEW.employe_id := seq_employe_id.NEXTVAL;
    END IF;
END;
/

-- Trigger audit
CREATE OR REPLACE TRIGGER trg_employes_audit
BEFORE UPDATE ON employes
FOR EACH ROW
BEGIN
    :NEW.modifie_par := USER;
    :NEW.modifie_le := CURRENT_TIMESTAMP;
END;
/

-- Index sur FK et colonnes fréquentes
CREATE INDEX idx_emp_dept ON employes(dept_id);
CREATE INDEX idx_emp_manager ON employes(manager_id);
CREATE INDEX idx_emp_nom ON employes(UPPER(nom));

-- ==========================================
-- 3. TABLE PROJETS
-- ==========================================
CREATE TABLE projets (
    projet_id NUMBER PRIMARY KEY,
    nom VARCHAR2(200) NOT NULL,
    description CLOB,
    date_debut DATE NOT NULL,
    date_fin DATE,
    budget NUMBER(12,2),
    chef_projet_id NUMBER NOT NULL,
    statut VARCHAR2(20) DEFAULT 'PLANIFIE',
    
    CONSTRAINT fk_proj_chef FOREIGN KEY (chef_projet_id) REFERENCES employes(employe_id),
    CONSTRAINT ck_proj_dates CHECK (date_fin IS NULL OR date_fin >= date_debut),
    CONSTRAINT ck_proj_budget CHECK (budget IS NULL OR budget > 0),
    CONSTRAINT ck_proj_statut CHECK (statut IN ('PLANIFIE', 'EN_COURS', 'TERMINE', 'ANNULE'))
);

CREATE OR REPLACE TRIGGER trg_projet_id
BEFORE INSERT ON projets
FOR EACH ROW
BEGIN
    IF :NEW.projet_id IS NULL THEN
        :NEW.projet_id := seq_projet_id.NEXTVAL;
    END IF;
END;
/

CREATE INDEX idx_proj_chef ON projets(chef_projet_id);

-- ==========================================
-- 4. TABLE AFFECTATIONS
-- ==========================================
CREATE TABLE affectations (
    affectation_id NUMBER PRIMARY KEY,
    employe_id NUMBER NOT NULL,
    projet_id NUMBER NOT NULL,
    role VARCHAR2(50),
    date_debut DATE DEFAULT TRUNC(SYSDATE) NOT NULL,
    date_fin DATE,
    taux_allocation NUMBER(5,2) DEFAULT 100,
    
    CONSTRAINT fk_aff_emp FOREIGN KEY (employe_id) REFERENCES employes(employe_id),
    CONSTRAINT fk_aff_proj FOREIGN KEY (projet_id) REFERENCES projets(projet_id) ON DELETE CASCADE,
    CONSTRAINT uk_aff_emp_proj UNIQUE (employe_id, projet_id, date_debut),
    CONSTRAINT ck_aff_dates CHECK (date_fin IS NULL OR date_fin >= date_debut),
    CONSTRAINT ck_aff_taux CHECK (taux_allocation BETWEEN 0 AND 100)
);

CREATE OR REPLACE TRIGGER trg_affectation_id
BEFORE INSERT ON affectations
FOR EACH ROW
BEGIN
    IF :NEW.affectation_id IS NULL THEN
        :NEW.affectation_id := seq_affectation_id.NEXTVAL;
    END IF;
END;
/

CREATE INDEX idx_aff_emp ON affectations(employe_id);
CREATE INDEX idx_aff_proj ON affectations(projet_id);

-- ==========================================
-- 5. TABLE LOGS (partitionnée par mois en 11g - manuel)
-- ==========================================
CREATE TABLE logs_activite (
    log_id NUMBER PRIMARY KEY,
    employe_id NUMBER,
    action VARCHAR2(50) NOT NULL,
    description VARCHAR2(500),
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
)
PARTITION BY RANGE (log_timestamp) (
    PARTITION logs_2024_01 VALUES LESS THAN (TO_DATE('2024-02-01', 'YYYY-MM-DD')),
    PARTITION logs_2024_02 VALUES LESS THAN (TO_DATE('2024-03-01', 'YYYY-MM-DD')),
    PARTITION logs_2024_03 VALUES LESS THAN (TO_DATE('2024-04-01', 'YYYY-MM-DD'))
    -- Ajouter manuellement les partitions futures
);

CREATE OR REPLACE TRIGGER trg_log_id
BEFORE INSERT ON logs_activite
FOR EACH ROW
BEGIN
    IF :NEW.log_id IS NULL THEN
        :NEW.log_id := seq_log_id.NEXTVAL;
    END IF;
END;
/

-- ==========================================
-- VUES MÉTIER
-- ==========================================
CREATE VIEW v_employes_actifs AS
SELECT e.employe_id, e.nom, e.prenom, e.email, d.nom AS departement
FROM employes e
JOIN departements d ON e.dept_id = d.dept_id
WHERE e.actif = 'Y';

CREATE VIEW v_projets_en_cours AS
SELECT p.projet_id, p.nom, p.date_debut, 
       e.nom || ' ' || e.prenom AS chef_projet,
       COUNT(a.employe_id) AS nb_ressources
FROM projets p
JOIN employes e ON p.chef_projet_id = e.employe_id
LEFT JOIN affectations a ON p.projet_id = a.projet_id 
    AND (a.date_fin IS NULL OR a.date_fin >= TRUNC(SYSDATE))
WHERE p.statut = 'EN_COURS'
GROUP BY p.projet_id, p.nom, p.date_debut, e.nom, e.prenom;
```

### Version Oracle 19c (avec améliorations)

```sql
-- ==========================================
-- VERSION ORACLE 19c
-- ==========================================

-- 1. DÉPARTEMENTS (IDENTITY column)
CREATE TABLE departements (
    dept_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nom VARCHAR2(100) NOT NULL,
    budget NUMBER(12,2) CHECK (budget >= 0),
    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_dept_nom UNIQUE (nom)
);

-- 2. EMPLOYÉS
CREATE TABLE employes (
    employe_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dept_id NUMBER NOT NULL,
    manager_id NUMBER,
    nom VARCHAR2(50) NOT NULL,
    prenom VARCHAR2(50) NOT NULL,
    email VARCHAR2(100) NOT NULL,
    telephone VARCHAR2(20),
    date_embauche DATE DEFAULT TRUNC(SYSDATE) NOT NULL,
    salaire NUMBER(10,2) NOT NULL,
    actif CHAR(1) DEFAULT 'Y' NOT NULL,
    cree_par VARCHAR2(50) DEFAULT USER,
    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modifie_par VARCHAR2(50),
    modifie_le TIMESTAMP,
    
    CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) REFERENCES departements(dept_id),
    CONSTRAINT fk_emp_manager FOREIGN KEY (manager_id) REFERENCES employes(employe_id) ON DELETE SET NULL,
    CONSTRAINT uk_emp_email UNIQUE (email),
    CONSTRAINT ck_emp_salaire CHECK (salaire > 0),
    CONSTRAINT ck_emp_actif CHECK (actif IN ('Y', 'N'))
);

-- Trigger audit (identique 11g/19c)
CREATE OR REPLACE TRIGGER trg_employes_audit
BEFORE UPDATE ON employes
FOR EACH ROW
BEGIN
    :NEW.modifie_par := USER;
    :NEW.modifie_le := CURRENT_TIMESTAMP;
END;
/

CREATE INDEX idx_emp_dept ON employes(dept_id);
CREATE INDEX idx_emp_manager ON employes(manager_id);
CREATE INDEX idx_emp_nom ON employes(UPPER(nom));

-- 3. PROJETS
CREATE TABLE projets (
    projet_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nom VARCHAR2(200) NOT NULL,
    description CLOB,
    date_debut DATE NOT NULL,
    date_fin DATE,
    budget NUMBER(12,2),
    chef_projet_id NUMBER NOT NULL,
    statut VARCHAR2(20) DEFAULT 'PLANIFIE',
    
    CONSTRAINT fk_proj_chef FOREIGN KEY (chef_projet_id) REFERENCES employes(employe_id),
    CONSTRAINT ck_proj_dates CHECK (date_fin IS NULL OR date_fin >= date_debut),
    CONSTRAINT ck_proj_budget CHECK (budget IS NULL OR budget > 0),
    CONSTRAINT ck_proj_statut CHECK (statut IN ('PLANIFIE', 'EN_COURS', 'TERMINE', 'ANNULE'))
);

CREATE INDEX idx_proj_chef ON projets(chef_projet_id);

-- 4. AFFECTATIONS
CREATE TABLE affectations (
    affectation_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    employe_id NUMBER NOT NULL,
    projet_id NUMBER NOT NULL,
    role VARCHAR2(50),
    date_debut DATE DEFAULT TRUNC(SYSDATE) NOT NULL,
    date_fin DATE,
    taux_allocation NUMBER(5,2) DEFAULT 100,
    
    CONSTRAINT fk_aff_emp FOREIGN KEY (employe_id) REFERENCES employes(employe_id),
    CONSTRAINT fk_aff_proj FOREIGN KEY (projet_id) REFERENCES projets(projet_id) ON DELETE CASCADE,
    CONSTRAINT uk_aff_emp_proj UNIQUE (employe_id, projet_id, date_debut),
    CONSTRAINT ck_aff_dates CHECK (date_fin IS NULL OR date_fin >= date_debut),
    CONSTRAINT ck_aff_taux CHECK (taux_allocation BETWEEN 0 AND 100)
);

CREATE INDEX idx_aff_emp ON affectations(employe_id);
CREATE INDEX idx_aff_proj ON affectations(projet_id);

-- 5. LOGS avec INTERVAL partitioning automatique (19c)
CREATE TABLE logs_activite (
    log_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    employe_id NUMBER,
    action VARCHAR2(50) NOT NULL,
    description VARCHAR2(500),
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
)
PARTITION BY RANGE (log_timestamp) 
INTERVAL (NUMTOYMINTERVAL(1, 'MONTH'))  -- Partition auto par mois
(
    PARTITION logs_init VALUES LESS THAN (TO_DATE('2024-01-01', 'YYYY-MM-DD'))
);
-- Oracle 19c crée automatiquement les partitions au fil des insertions !

-- VUES (identiques 11g/19c)
CREATE VIEW v_employes_actifs AS
SELECT e.employe_id, e.nom, e.prenom, e.email, d.nom AS departement
FROM employes e
JOIN departements d ON e.dept_id = d.dept_id
WHERE e.actif = 'Y';

CREATE VIEW v_projets_en_cours AS
SELECT p.projet_id, p.nom, p.date_debut, 
       e.nom || ' ' || e.prenom AS chef_projet,
       COUNT(a.employe_id) AS nb_ressources
FROM projets p
JOIN employes e ON p.chef_projet_id = e.employe_id
LEFT JOIN affectations a ON p.projet_id = a.projet_id 
    AND (a.date_fin IS NULL OR a.date_fin >= TRUNC(SYSDATE))
WHERE p.statut = 'EN_COURS'
GROUP BY p.projet_id, p.nom, p.date_debut, e.nom, e.prenom;
```

---

## 12. Commandes de Gestion 🟢 **11g+**

### 12.1 Modifier une Table

```sql
-- Ajouter une colonne
ALTER TABLE employes ADD (
    badge_id VARCHAR2(20),
    photo BLOB
);

-- Modifier une colonne
ALTER TABLE employes MODIFY (
    telephone VARCHAR2(25)  -- Agrandir
);

-- Renommer une colonne
ALTER TABLE employes RENAME COLUMN telephone TO tel_portable;

-- Supprimer une colonne
ALTER TABLE employes DROP COLUMN badge_id;

-- Ajouter une contrainte
ALTER TABLE employes ADD CONSTRAINT ck_emp_badge 
    CHECK (badge_id IS NOT NULL AND LENGTH(badge_id) = 10);

-- Désactiver/Activer une contrainte
ALTER TABLE employes DISABLE CONSTRAINT ck_emp_salaire;
ALTER TABLE employes ENABLE CONSTRAINT ck_emp_salaire;
```

### 12.2 Consulter les Métadonnées

```sql
-- Lister toutes vos tables
SELECT table_name FROM user_tables ORDER BY table_name;

-- Structure d'une table
DESCRIBE employes;

-- Ou via SQL
SELECT column_name, data_type, data_length, nullable, data_default
FROM user_tab_columns
WHERE table_name = 'EMPLOYES'
ORDER BY column_id;

-- Contraintes d'une table
SELECT constraint_name, constraint_type, search_condition
FROM user_constraints
WHERE table_name = 'EMPLOYES';

-- Colonnes des contraintes
SELECT uc.constraint_name, uc.constraint_type, ucc.column_name
FROM user_constraints uc
JOIN user_cons_columns ucc ON uc.constraint_name = ucc.constraint_name
WHERE uc.table_name = 'EMPLOYES'
ORDER BY uc.constraint_name, ucc.position;

-- Index d'une table
SELECT index_name, column_name, column_position
FROM user_ind_columns
WHERE table_name = 'EMPLOYES'
ORDER BY index_name, column_position;

-- Triggers d'une table
SELECT trigger_name, triggering_event, status
FROM user_triggers
WHERE table_name = 'EMPLOYES';
```

### 12.3 Supprimer une Table

```sql
-- Suppression simple
DROP TABLE employes;

-- Avec suppression en cascade des contraintes FK
DROP TABLE departements CASCADE CONSTRAINTS;

-- Suppression avec purge immédiate (pas de corbeille)
DROP TABLE employes PURGE;

-- Restauration depuis la corbeille (Flashback Drop) 🟢 11g+
FLASHBACK TABLE employes TO BEFORE DROP;
```

---

## 13. Différences Majeures 11g vs 19c : Récapitulatif

| Fonctionnalité | Oracle 11g | Oracle 19c |
|----------------|------------|------------|
| **Auto-increment** | Séquence + Trigger | IDENTITY columns (12c+) |
| **VARCHAR2 max** | 4000 bytes | 4000 (ou 32767 avec EXTENDED) |
| **Colonnes invisibles** | ❌ Non | ✅ Oui (12c+) |
| **JSON natif** | ❌ Non (XML uniquement) | ✅ Oui (12c+) |
| **INTERVAL partitioning** | ❌ Manuel | ✅ Automatique (12c+) |
| **Compound triggers** | ✅ Oui (11g R2+) | ✅ Oui |
| **Function-based index** | ✅ Oui | ✅ Oui |
| **Bitmap index** | ✅ Oui (EE) | ✅ Oui (EE) |
| **Real-Time Statistics** | ❌ Non | ✅ Oui (12c+) |
| **Automatic Indexing** | ❌ Non | ✅ Oui (19c+) |

---

## 14. Checklist de Création de Table

**Avant de créer une table, vérifier :**

1. ✅ **Normalisation** : La table respecte-t-elle au moins la 3NF ?
2. ✅ **Clé primaire** : Identifiant unique défini ? (IDENTITY en 19c, Séquence en 11g)
3. ✅ **Types de données** : Les plus restrictifs possibles ?
4. ✅ **NOT NULL** : Sur toutes les colonnes obligatoires ?
5. ✅ **Contraintes** : CHECK, UNIQUE, FK correctement définies ?
6. ✅ **Nommage** : Convention respectée (contraintes nommées) ?
7. ✅ **Index** : Sur FK et colonnes fréquemment recherchées ?
8. ✅ **Triggers** : Audit, validation, historisation nécessaires ?
9. ✅ **Partitionnement** : Nécessaire pour volumétrie/performances ?
10. ✅ **Audit** : Colonnes de traçabilité (cree_le, modifie_par) ?
11. ✅ **Documentation** : Commentaires sur table/colonnes ?

```sql
-- Ajout de commentaires (documentation inline)
COMMENT ON TABLE employes IS 'Référentiel des employés actifs et inactifs';
COMMENT ON COLUMN employes.actif IS 'Y=Actif, N=Inactif (suppression logique)';
```

---

## 15. Ressources et Documentation

### Documentation Oracle officielle

- **Oracle 11g Documentation** : https://docs.oracle.com/cd/E11882_01/nav/portal_11.htm
- **Oracle 19c Documentation** : https://docs.oracle.com/en/database/oracle/oracle-database/19/
- **SQL Language Reference 11g** : https://docs.oracle.com/cd/E11882_01/server.112/e41084/toc.htm
- **SQL Language Reference 19c** : https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/

### Livres recommandés

- **"Expert Oracle Database Architecture"** par Thomas Kyte
- **"Oracle PL/SQL Programming"** par Steven Feuerstein
- **"Oracle Database 12c Performance Tuning"** (applicable 11g/19c)

### Outils recommandés

- **SQL Developer** : IDE officiel Oracle (gratuit)
- **Toad for Oracle** : Alternative commerciale puissante
- **DBeaver** : Open source, multi-base

### Commandes utiles

```sql
-- Générer DDL d'une table existante (11g et 19c)
SELECT DBMS_METADATA.GET_DDL('TABLE', 'EMPLOYES') FROM DUAL;

-- Générer DDL des index
SELECT DBMS_METADATA.GET_DDL('INDEX', 'IDX_EMP_NOM') FROM DUAL;

-- Générer DDL des triggers
SELECT DBMS_METADATA.GET_DDL('TRIGGER', 'TRG_EMPLOYES_INSERT') FROM DUAL;

-- Statistiques d'une table
SELECT num_rows, blocks, avg_row_len 
FROM user_tables 
WHERE table_name = 'EMPLOYES';

-- Taille d'une table en Mo
SELECT segment_name, ROUND(bytes/1024/1024, 2) AS size_mb
FROM user_segments
WHERE segment_name = 'EMPLOYES' AND segment_type = 'TABLE';
```

---

## Conclusion

La création de tables Oracle est la fondation de votre base de données. Comprendre les différences entre **Oracle 11g** et **Oracle 19c** est essentiel pour travailler efficacement dans votre entreprise.

**Points clés à retenir :**

1. **Oracle 11g** : Nécessite séquences + triggers pour auto-increment
2. **Oracle 19c** : IDENTITY columns, JSON, partitionnement automatique
3. **Index** : Toujours sur FK, stratégie selon cardinalité (B-Tree vs Bitmap)
4. **Triggers** : Audit, validation, historisation (row-level vs statement-level)
5. **Performance** : Statistiques, index couvrants, compound triggers (19c)

**Prochaines étapes pour approfondir :**
- Maîtriser le tuning SQL (EXPLAIN PLAN, plans d'exécution)
- Approfondir PL/SQL (procédures stockées, packages)
- Explorer les fonctionnalités avancées (partitionnement, Real Application Clusters)
- Étudier les stratégies de backup/recovery
- Pratiquer la migration 11g → 19c

**Ressources pratiques pour votre entreprise :**
- Scripts de migration 11g → 19c
- Compatibilité backward (code 19c qui fonctionne en 11g)
- Monitoring performances (AWR reports, ASH)

Oracle reste un SGBD extrêmement robuste et performant pour les applications d'entreprise critiques ! 🚀
