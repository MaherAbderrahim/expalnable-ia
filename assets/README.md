# Assets (ressources graphiques)

## Contenu

| Fichier | Description | Auteur | Licence |
|---|---|---|---|
| `task-taxonomy.svg` | Les trois tâches prostatiques — détection au niveau du patient, au niveau de la lésion, et grading — et ce que chacune permet d'expliquer | Original, écrit pour ce projet | Identique à celle du projet (libre à réutiliser avec attribution) |
| `xai-levels.svg` | Les quatre niveaux d'explicabilité prostatique, de la carte de chaleur au compte rendu PI-RADS jusqu'à l'explication intrinsèque | Original, écrit pour ce projet | Identique à celle du projet (libre à réutiliser avec attribution) |
| `evidence-chain.svg` | La chaîne de vérification employée pour chaque affirmation, et le piège Git LFS | Original, écrit pour ce projet | Identique à celle du projet (libre à réutiliser avec attribution) |

## Déclaration de licences — à lire avant d'ajouter quoi que ce soit

**Tous les fichiers de ce dossier ont été créés de zéro pour ce projet.** Aucune image tierce, capture d'écran, logo, figure de publication, photographie ou bribe de jeu de données n'a été copiée, recadrée, adaptée ou incluse ici.

C'était un choix délibéré, pas un oubli. La tentation naturelle aurait été d'ajouter une figure d'un article, une capture d'un examen IRM de PI-CAI, ou une image de lame H&E de PANDA. Chacune de ces ressources a un statut de licence qu'il faudrait dégager au préalable :

- **Données PI-CAI** — sous **CC BY-NC 4.0** (non commercial). Toute imagerie redistribuée sans accord violerait la licence, et le projet cherche précisément à ne pas le faire.
- **Données PANDA** — sous **CC BY-SA-NC 4.0** : non commercial **et** partage à l'identique. Les images de biopsie sont derivatives de l'ensemble ; les rediffuser imposerait la même licence à l'ensemble du projet.
- **Données PROSTATEx** — CC BY 3.0 sur TCIA, ce qui autoriserait la réutilisation, mais la collection est **dépréciée** et incluse dans PI-CAI : l'utiliser séparément créerait un chevauchement de patients.
- **Figures d'articles** — chacune relève du droit d'auteur de son éditeur ou de ses auteurs. Une capture d'écran n'est pas une quotation ; l'utiliser sans autorisation est une contrefaçon caractérisée, et la licence arXiv n'accorde aucun droit de redistribution. L'article de Hamm et al. (*Radiology*) est de plus sous une licence exclusive.

Les données de recherche (`research-data.json`) ne contiennent que des **métadonnées** : tailles de jeux de données, nombres de patients et de classes, identifiants de licence, URL et tailles en octets de fichiers vérifiés. Aucune donnée image n'est redistribuée, aucun échantillon de jeu de données n'est stocké dans ce dépôt, et **aucune donnée n'est sous licence non commerciale n'est redistribuée**.

**Si vous devez ensuite montrer de vraies images IRM ou des lames H&E dans votre propre présentation**, la voie correcte et simple est : téléchargez-les vous-même depuis la source, sous les conditions de la source, et citez le jeu de données. Pour PI-CAI, c'est un téléchargement Zenodo avec une citation CC BY-NC 4.0 ; pour PANDA, c'est un téléchargement Kaggle avec les règles de la compétition. Les deux sont gratuits et vous laissent dans une position de licence propre — et, dans le cas de PI-CAI, l'obligation d'attribuer correctement.

**Si vous ajoutez une image dans ce dossier plus tard**, notez ici : son URL source, sa licence, la date de récupération, et si une attribution est requise. Une image intraçable dans un dossier d'assets est un passif.
