##############################################################################
# Copyright (C) 2024                                                         #
#                                                                            #
# CC BY-NC-SA 4.0                                                            #
#                                                                            #
# Canonical URL https://creativecommons.org/licenses/by-nc-sa/4.0/           #
# Attribution-NonCommercial-ShareAlike 4.0 International CC BY-NC-SA 4.0     #
#                                                                            #
# Prof. Elaine Cecilia Gatto | Prof. Ricardo Cerri | Prof. Mauri Ferrandin   #
#                                                                            #
# Federal University of São Carlos - UFSCar - https://www2.ufscar.br         #
# Campus São Carlos - Computer Department - DC - https://site.dc.ufscar.br   #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br - Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
# You are free to:                                                           #
#     Share — copy and redistribute the material in any medium or format     #
#     Adapt — remix, transform, and build upon the material                  #
#     The licensor cannot revoke these freedoms as long as you follow the    #
#       license terms.                                                       #
#                                                                            #
# Under the following terms:                                                 #
#   Attribution — You must give appropriate credit , provide a link to the   #
#     license, and indicate if changes were made . You may do so in any      #
#     reasonable manner, but not in any way that suggests the licensor       #
#     endorses you or your use.                                              #
#   NonCommercial — You may not use the material for commercial purposes     #
#   ShareAlike — If you remix, transform, or build upon the material, you    #
#     must distribute your contributions under the same license as the       #
#     original.                                                              #
#   No additional restrictions — You may not apply legal terms or            #
#     technological measures that legally restrict others from doing         #
#     anything the license permits.                                          #
#                                                                            #
##############################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from ECC import ECC


# 1. Generate a multi-label classification dataset (simulated)
X, Y = make_multilabel_classification(n_samples=1000, n_features=20, 
                                      n_classes=5, n_labels=2, 
                                      random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=42)

# Convert Y_train and Y_test into DataFrame for ease of use
Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test)

# 2. Initialize the ECC model with a RandomForestClassifier as the base model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Using 1 core to avoid parallelization
ecc = ECC(model=model, n_chains=10, n_jobs=1) 

# 3. Fit the ECC model
ecc.fit(X_train, Y_train)

# 4. Predict probabilities on the test set
probas = ecc.predict_proba(X_test)

# 5. Make binary predictions on the test set using a threshold of 0.5
binary_preds = ecc.predict(X_test, threshold=0.5)

# 6. Make binary predictions using the cardinality of the training set as threshold
cardinality_preds = ecc.predict_cardinality(X_test, Y_train)

# 7. Save predictions to CSV for later use
ecc.save_predictions(binary_preds, 'binary_predictions.csv')
ecc.save_predictions(probas, 'probability_predictions.csv')
ecc.save_predictions(cardinality_preds, 'cardinality_predictions.csv')

# 8. Get the execution time of each chain and the total training time
chain_times, total_time = ecc.get_chain_times()
print(f"\nExecution time of each chain: {chain_times}")
print(f"\nTotal training time: {total_time:.2f} seconds")

# 9. Check the size of the models for each chain in bytes
model_sizes = ecc.model_size()
print(f"Model sizes for each chain (in bytes): {model_sizes}")

# 13. mozel sizes com 
model_sizes_2 = ecc.model_size_joblib()
print(f"Model sizes for each chain: {model_sizes_2}")

# 10. Display the first 5 binary predictions
print("First 5 binary predictions:")
print(binary_preds[:5])

# 11. Display the first 5 cardinality-based predictions
print("First 5 cardinality-based predictions:")
print(cardinality_preds[:5])

# 12. Display the first 5 predicted probabilities
print("First 5 predicted probabilities:")
print(probas[:5])