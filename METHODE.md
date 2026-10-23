# Méthodologie du projet SVR + WSO

**Réalisé par : BADDOU Halima et EL GOUDE Meriem**  
**Formation : Master Intelligence Artificielle (IAA)**  
**Établissement : Université Ibn Zohr — Faculté des Sciences Appliquées de Ouarzazate**

Cette description est alignée sur le code et les sorties enregistrées du notebook `Projet_WSO_SVR.ipynb` joint. La vérification porte sur la lecture du notebook ; aucune nouvelle exécution complète n'a été effectuée lors de cette correction. Le notebook n'a pas été modifié par cette correction du document. Les correctifs de fonctionnement proposés figurent à la section 10.

## 1. Objectif et choix du formulaire

- **Méthode :** adaptation pédagogique de White Shark Optimizer (WSO).
- **Application :** prédiction des besoins de chauffage des bâtiments.
- **Problème :** optimisation continue de trois hyperparamètres par minimisation de l'erreur de validation croisée.
- **Modèle :** régression à vecteurs de support (SVR) à noyau RBF.

Nous comparons un SVR initial, un SVR réglé par WSO et un SVR réglé par recherche aléatoire. L'objectif est d'évaluer ces configurations dans un protocole défini, sans prétendre proposer un nouvel algorithme ou démontrer la supériorité générale de WSO.

## 2. Données, chargement et exploration

Le jeu **Energy Efficiency** de Tsanas et Xifara contient 768 configurations simulées de bâtiments, huit variables explicatives et deux sorties. Source officielle : [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency), DOI : [10.24432/C51307](https://doi.org/10.24432/C51307), licence CC BY 4.0.

| Colonne | Signification | Utilisation |
|---|---|---|
| X1 | Compacité relative | Entrée numérique |
| X2 | Surface totale | Entrée numérique |
| X3 | Surface des murs | Entrée numérique |
| X4 | Surface du toit | Entrée numérique |
| X5 | Hauteur totale | Entrée numérique |
| X6 | Orientation | Entrée catégorielle |
| X7 | Surface vitrée relative | Entrée numérique |
| X8 | Répartition du vitrage | Entrée catégorielle |
| Y1 | Besoin de chauffage | Cible à prédire |
| Y2 | Besoin de refroidissement | Exclue des entrées |

Y2 est exclue parce que le scénario étudié utilise uniquement les caractéristiques du bâtiment. Les cibles restent dans leur échelle d'origine ; aucune unité physique non vérifiée n'est ajoutée.

Dans le notebook joint, le téléchargement est **systématique** : le programme récupère l'archive UCI, ouvre `ENB2012_data.xlsx` et charge le tableau dans `df`. Il ne vérifie pas l'existence d'un CSV local. Une cellule distincte crée le dossier `data` et exporte le tableau.

L'exploration affiche les premières lignes, les types de colonnes, les valeurs manquantes et les statistiques descriptives. Les sorties enregistrées indiquent 768 lignes, dix colonnes et aucune valeur manquante. Deux assertions vérifient le nombre de lignes et l'absence de valeurs manquantes dans X1 à X8 et Y1. Il n'y a pas d'imputation ni de suppression de lignes dans le code.

**Particularité de l'export actuel :** `df.to_csv("data/energy_efficiency.csv")` écrit aussi l'index Pandas. Cela ne change pas les entrées de cette exécution, qui utilisent directement `df[features]`, mais ajoute une colonne technique au CSV. L'export conseillé est donné à la section 10.

## 3. Séparation, prétraitement et validation

La séparation aléatoire utilise `test_size=0.20` et `random_state=42` : 614 lignes sont réservées au développement et 154 au test final.

Sur les 614 lignes, nous utilisons `KFold(n_splits=3, shuffle=True, random_state=42)`. Les indices des trois plis sont calculés une fois et réutilisés pour toutes les configurations.

Le prétraitement contient :

- un `StandardScaler` pour X1, X2, X3, X4, X5 et X7 ;
- un `OneHotEncoder(handle_unknown='ignore')` pour X6 et X8 ;
- un SVR à noyau RBF placé après ces transformations dans une `Pipeline`.

Chaque pli réapprend les transformations uniquement sur sa partie d'entraînement. La cible n'est pas standardisée. Cette organisation évite une fuite d'information par le prétraitement vers la validation ou le test. Elle ne supprime pas la proximité éventuelle entre les géométries des différents sous-ensembles.

Le modèle initial utilise `C=1.0`, `gamma='scale'` et `epsilon=0.1`. L'option `scale` détermine gamma à partir des entrées transformées de l'entraînement ; ce n'est pas une constante manuelle.

## 4. Fonction objectif et espace de recherche

La fonction objectif est la **moyenne des RMSE des trois plis** :

\[
J(\theta)=\frac{1}{3}\sum_{k=1}^{3}\operatorname{RMSE}_k(\theta),
\qquad \theta=(C,\gamma,\varepsilon).
\]

Ce n'est pas la racine de la moyenne des MSE. Scikit-Learn renvoie des RMSE négatives avec `neg_root_mean_squared_error` ; le code change le signe de leur moyenne pour obtenir l'erreur positive à minimiser.

Les **positions normalisées** des requins, et non directement les hyperparamètres, appartiennent à `[0,1]^3`. Le décodage est :

\[
\theta_j=10^{\ell_j+z_j(h_j-\ell_j)},\quad
\ell=(-1,-4,-3),\quad h=(3,0,0).
\]

| Hyperparamètre | Domaine réel | Rôle |
|---|---|---|
| C | [0,1 ; 1000] | Pénalisation des erreurs du SVR |
| gamma | [0,0001 ; 1] | Portée du noyau RBF |
| epsilon | [0,001 ; 1] | Tolérance de la perte du SVR |

L'échelle logarithmique couvre plusieurs ordres de grandeur. WSO choisit ces trois réglages ; Scikit-Learn apprend les coefficients internes du SVR.



### Initialisation et déplacement

Nous initialisons 12 positions uniformes dans `[0,1]^3` et des vitesses nulles. Chaque position est évaluée ; la meilleure est conservée. Pour 15 itérations, le code utilise :

- `tau = 4.125` et `mu = 2 / abs(2 - tau - sqrt(tau**2 - 4*tau))` ;
- `p1 = 1.5 + exp(-(4*t/T)**2)` ;
- `p2 = 0.5 + exp(-(4*t/T)**2)` ;
- une attraction vers la meilleure position globale et vers un requin tiré au hasard dans la population ; ce tirage peut sélectionner le requin lui-même ;
- une fréquence indépendante par requin, uniforme dans `[0.07, 0.75]` ;
- un candidat égal à l'ancienne position augmentée de la vitesse divisée par cette fréquence.

### Branche de maintien

Pour chaque requin, un tirage est comparé à :

```python
movement = 1 / (6.25 + np.exp((T / 2 - t) / 100))
```

Si le tirage est inférieur à ce seuil, `candidates[stay] = positions[stay]` **remplace le candidat par l'ancienne position**. Avec T=15, la probabilité est proche de 14 % par requin et par itération ; ce n'est pas un pourcentage observé de la population à chaque cycle.

Le maintien ne remet pas la vitesse à zéro. De plus, le suivi collectif peut ensuite modifier le candidat d'indice 0 : un requin sélectionné pour le maintien n'est donc pas nécessairement évalué à sa position précédente après toutes les étapes. Pour les autres requins maintenus, la position est réévaluée sans déplacement. Aucun cache n'évite ces appels répétés.

### Suivi collectif et bornage

Après une première projection sur `[0,1]^3`, le code utilise :

```python
school = abs(1 - np.exp(-0.0005 * t / T))
if i < school * population:
    ...
```

Avec N=12, T=15 et t allant de 1 à 15, `school * N` reste strictement entre 0 et 1. **Seul le requin d'indice 0** est donc concerné. Cette condition est indicielle, et non une probabilité indépendante de suivre le groupe.

Pour ce requin, le programme calcule une position voisine de la meilleure solution, puis :

```python
candidates[i] = np.clip((candidates[i] + nearby) / divisor, 0, 1)
```

où `divisor = 2 * maximum(aléa, 1e-12)` est un vecteur. Cette formule peut dépasser les bornes, surtout lorsque le dénominateur est petit. La projection ramène alors les coordonnées dans le domaine. **La fréquence de saturation n'a pas été mesurée : il serait incorrect d'affirmer qu'elle se produit « presque toujours ».**

### Acceptation et arrêt

Chaque candidat final est évalué, même s'il répète une position. Il remplace sa position courante uniquement si son erreur est strictement inférieure. La meilleure solution globale est mise à jour pendant la boucle d'évaluation, et les nouvelles vitesses sont conservées même en cas de rejet du candidat.

L'algorithme s'arrête après 15 itérations, sans arrêt anticipé. L'historique contient 16 meilleurs coûts : l'initialisation puis les 15 cycles. Sa non-croissance vient de la conservation du meilleur coût ; elle ne prouve pas l'optimalité globale.

## 6. Comparaisons et budget

| Élément | WSO | Recherche aléatoire |
|---|---:|---:|
| Graine de recherche | 42 | 43 |
| Évaluations initiales | 12 | Sans étape séparée |
| Évaluations totales | 12 + 12 × 15 = 192 | 192 |
| Plis par évaluation | 3 | 3 |
| Ajustements internes du SVR | 576 | 576 |

Les recherches utilisent le même espace logarithmique, les mêmes plis et la même fonction objectif. L'égalité porte sur le nombre d'appels, pas sur le nombre de configurations distinctes ni sur la durée.

Le SVR initial nécessite trois ajustements de validation supplémentaires. Les trois configurations finales sont ensuite entraînées chacune sur les 614 lignes : le protocole comporte donc 1158 ajustements de SVR au total, si toutes les étapes s'achèvent normalement.

Les temps enregistrés couvrent la recherche de chaque méthode, pas le chargement ni les trois entraînements finaux. Les deux procédures évaluent des hyperparamètres différents : expliquer une différence de temps par C ou epsilon reste une hypothèse sans mesure complémentaire.

## 7. Test final et résultats visibles

Les paramètres sont fixés à partir de la validation avant de calculer les prédictions des trois modèles sur le même test. Le test ne sert pas à sélectionner les hyperparamètres dans le code fourni. Il est consulté lors d'une phase finale qui évalue chaque modèle ; il ne s'agit pas d'un unique appel pour les trois modèles.

Les sorties enregistrées dans le notebook joint affichent :

| Modèle | RMSE CV | RMSE test | MAE test | R² test |
|---|---:|---:|---:|---:|
| SVR initial | 2,706755 | 2,663562 | 1,657852 | 0,931935 |
| SVR + WSO | 0,940060 | 0,511822 | 0,368634 | 0,997487 |
| SVR + recherche aléatoire | 0,981786 | 0,705743 | 0,508007 | 0,995222 |

Ces valeurs sont lues dans les sorties conservées, sans nouvelle exécution lors de cette correction. Elles doivent être reliées aux exports de la même session pour la version finale du rapport. Les sorties d'un notebook peuvent persister après modification de ses cellules et ne prouvent pas à elles seules une exécution complète depuis un environnement vide.

Paramètres WSO affichés : `C=1000.0`, `gamma=0.0683623293740392`, `epsilon=0.006219548396136771`.

La trace affiche une RMSE CV arrondie à 0,9683 des itérations 1 à 13, puis 0,9401 aux itérations 14 et 15. La valeur exacte d'initialisation n'est pas imprimée dans cette trace ; elle doit être lue dans l'export `convergence.csv`, sans estimation à partir de la figure.

## 8. Traçabilité et interprétation

Le code prévoit les exports `comparaison.csv`, `predictions_test.csv`, `convergence.csv`, `experience.json` et `graphiques.png`. Les métadonnées comprennent les versions de NumPy, Pandas et Scikit-Learn, les indices du découpage, les graines, les paramètres et les durées.

 
## 9. Limites et perspectives

- Une seule graine par recherche et un seul découpage ne permettent pas d'estimer la variabilité des performances.
- Les observations sont simulées ; aucune validation sur des bâtiments réels n'est effectuée.
- Des configurations proches peuvent se trouver des deux côtés du découpage. Cela ne démontre pas la généralisation à des familles de géométries nouvelles.
- Le paramètre C retenu atteint la borne supérieure de 1000. Les conclusions dépendent du domaine choisi ; il ne faut pas ajuster ce domaine à partir du test déjà consulté.
- La recherche utilise plusieurs fois les mêmes plis. Une validation imbriquée et des répétitions prévues à l'avance renforceraient l'étude.
- La branche de maintien réévalue des configurations, et le suivi collectif est limité à l'indice 0 avec ces constantes. Ces propriétés doivent rester explicites.




## 11. Sources

1. Tsanas, A. et Xifara, A. (2012). *Energy Efficiency*. UCI Machine Learning Repository. https://doi.org/10.24432/C51307
2. Braik, M., Hammouri, A., Atwan, J., Al-Betar, M. A. et Awadallah, M. A. (2022). *White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems*. Knowledge-Based Systems, 243, 108457. https://doi.org/10.1016/j.knosys.2022.108457
3. Braik, M. *White Shark Optimizer (WSO)*, MATLAB Central File Exchange. https://www.mathworks.com/matlabcentral/fileexchange/107365-white-shark-optimizer-wso
4. Documentation MEALPY, source de `mealpy.swarm_based.WSO`. https://mealpy.readthedocs.io/en/latest/_modules/mealpy/swarm_based/WSO.html


