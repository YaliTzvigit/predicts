



import pandas as pd
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.multiclass import OneVsRestClassifier # type: ignore 

# Données d'entraînement avec les résultats des 5 derniers matchs
data = {
    'resultats_Atl': [0, 1, 1, 1, 1],  
    'resultats_udi': [1, -1, 1, -1, -1],     
    'buts_marqués_Atl': [0, 6, 2, 3, 2],  
    'buts_marqués_udi': [1, 0, 2, 2, 0],     
    'buts_encassés_Atl': [0, 1, 0, 0, 0],  
    'buts_encassés_udi': [0, 1, 0, 3, 2],     
}

# Résultats des matchs (1 = victoire Atl, 0 = nul, -1 = défaite Atl)
resultats = [1, -1, 0, 1, 1]  # 3 classes pour la prédiction

df = pd.DataFrame(data)

df['moy_buts_marqués_Atl'] = df['buts_marqués_Atl'].mean()  
df['moy_buts_marqués_udi'] = df['buts_marqués_udi'].mean()        
df['moy_buts_encassés_Atl'] = df['buts_encassés_Atl'].mean()  
df['moy_buts_encassés_udi'] = df['buts_encassés_udi'].mean()        

# Préparation des données pour l'entraînement du modèle
X = df[['resultats_Atl', 'resultats_udi', 'buts_marqués_Atl', 'buts_marqués_udi', 
        'buts_encassés_Atl', 'buts_encassés_udi', 'moy_buts_marqués_Atl', 
        'moy_buts_marqués_udi', 'moy_buts_encassés_Atl', 'moy_buts_encassés_udi']]
y = pd.Series(resultats)

# Diviser les données en ensemble d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle avec multi_class pour prédire 3 catégories
model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
predictions = model.predict(X_test)

# Affichage des résultats
print(f"Prédictions sur l'ensemble de test : {predictions}")
print(f"Précision du modèle : {accuracy_score(y_test, predictions):.2f}")

# Nouveau match à prédire
nouveau_match = pd.DataFrame({

    'resultats_Atl': [sum([0, 1, 1, 1, 1]) / 5],  
    'resultats_udi': [sum([1, -1, 1, -1, -1]) / 5],     
    'buts_marqués_Atl': [2.6],  
    'buts_marqués_udi': [1.0],     
    'buts_encassés_Atl': [0.2],  
    'buts_encassés_udi': [1.2],     
    'moy_buts_marqués_Atl': [df['moy_buts_marqués_Atl'].mean()],
    'moy_buts_marqués_udi': [df['moy_buts_marqués_udi'].mean()],
    'moy_buts_encassés_Atl': [df['moy_buts_encassés_Atl'].mean()],
    'moy_buts_encassés_udi': [df['moy_buts_encassés_udi'].mean()]
})

# Prédiction pour le nouveau match
prediction_nouveau = model.predict(nouveau_match)
probabilites = model.predict_proba(nouveau_match)

# Affichage des probabilités pour les trois issues
prob_victoire_atl = round(probabilites[0][1] * 100)  # Victoire Atl    
prob_victoire_udi = round(probabilites[0][0] * 100)  # Victoire Udinese

resultat_texte = {1: "Victoire d'Atl", 0: "Match nul", -1: "Défaite d'Atl"}

print(f"\nPrédiction pour le nouveau match : {prediction_nouveau[0]}")
print(f"Probabilités : Victoire Atl: {prob_victoire_atl}%, Victoire Udinese: {prob_victoire_udi}%")
