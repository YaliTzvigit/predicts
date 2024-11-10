# Prediction sur ANGERS / PARIS SG

# PREDICTIONS SUR ANGERS / PSG 

import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

# Données d'entraînement
data = {
    'victoires_Angers': [2, 3, 1, 0, 2],  # Victoires d'Angers
    'victoires_PSG': [3, 1, 4, 5, 2],     # Victoires du PSG
    'moy_buts_marques_Angers': [1.6, 2.0, 1.2, 0.8, 1.5],  # Moyenne buts marqués par Angers
    'moy_buts_marques_PSG': [2.0, 2.0, 2.5, 3.0, 1.8],     # Moyenne buts marqués par PSG
    'moy_buts_encaisse_Angers': [1.4, 1.0, 1.8, 2.2, 1.6],  # Moyenne buts encaissés par Angers
    'moy_buts_encaisse_PSG': [1.0, 1.4, 0.8, 0.6, 1.2],     # Moyenne buts encaissés par PSG
}

# Résultats des matchs (1 = victoire Angers, 0 = nul, -1 = défaite Angers)
resultats = [1, -1, -1, -1, 0]

# Création du DataFrame
df = pd.DataFrame(data)

# Préparation des données
X = df
y = pd.Series(resultats)

# Diviser les données en ensemble d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
predictions = model.predict(X_test)

# Affichage des résultats
print(f"Prédictions sur l'ensemble de test : {predictions}")
print(f"Précision du modèle : {accuracy_score(y_test, predictions):.2f}")

# Nouveau match à prédire
nouveau_match = pd.DataFrame({
    'victoires_Angers': [2],
    'victoires_PSG': [4],
    'moy_buts_marques_Angers': [1.5],
    'moy_buts_marques_PSG': [2.8],
    'moy_buts_encaisse_Angers': [1.7],
    'moy_buts_encaisse_PSG': [0.9]
})

# Prédiction pour le nouveau match
prediction_nouveau = model.predict(nouveau_match)
probabilites = model.predict_proba(nouveau_match)

# Affichage des résultats pour le nouveau match
resultat_texte = {1: "Victoire d'Angers", 0: "Match nul", -1: "Défaite d'Angers"}
print(f"\nPrédiction pour le nouveau match : {resultat_texte[prediction_nouveau[0]]}")
print(f"Probabilités : Victoire Angers: {probabilites[0][2]:.2f}, Nul: {probabilites[0][1]:.2f}, Victoire PSG: {probabilites[0][0]:.2f}")