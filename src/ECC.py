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
from sklearn.base import clone
from sklearn.multioutput import ClassifierChain
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
import time
import sys
import joblib
import os
from joblib import Parallel, delayed

class ECC:
    def __init__(self, model, n_chains=10, n_jobs=1):
        """
        Initialize the Ensemble of Classifier Chains (ECC).

        Parameters:
        - model: Base model to be used in each classifier chain (e.g., RandomForest, SVC, etc.).
        - n_chains: Number of chains in the ensemble (default: 10).
        - n_jobs: Number of parallel jobs (default: 1). If 1, no parallelization will be used.
        
        Example of usage:
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        
        # Creating a multi-label dataset
        X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)
        
        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()
        
        # Initialize the ECC model with 10 chains and 1 job (no parallelization)
        ecc_model = ECC(model=base_model, n_chains=10, n_jobs=1)
        
        # Train the model (assuming there's a fit function implemented)
        ecc_model.fit(X, y)
        
        # Predict (assuming there's a predict function implemented)
        predictions = ecc_model.predict(X)
        
        """
        self.model = model
        self.n_chains = n_chains
        self.n_jobs = n_jobs
        self.chains = None
        self.chain_train_times = []  # To store the training time of each chain
        self.train_time_total = 0    # Total training time for the ensemble
        self.test_time_total = 0     # Total testing time for predictions

    #=========================================================#
    #                                                         #
    #=========================================================#
    def fit(self, X, Y):
        """
        Train the ensemble of classifier chains.

        Parameters:
        - X: Feature matrix (DataFrame or array-like).
        - Y: Multi-label targets (DataFrame or array-like).

        Returns:
        - None
        
        Example of usage:

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone
        import time

        # Creating a multi-label dataset
        X, Y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains (for example) and 1 job (no parallelization)
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model using the fit method
        ecc_model.fit(X, Y)
        
        # After training, you can inspect the training times for each chain
        print(ecc_model.chain_train_times)
        print(f"Total training time: {ecc_model.train_time_total:.2f} seconds")
        
        """
        if len(X) != len(Y):
            raise ValueError("The number of samples in X and Y must be the same.")
        
        # Initialize chains with random order and independent models
        self.chains = [ClassifierChain(clone(self.model), order="random") for _ in range(self.n_chains)]
        
        # Start total training time
        start_time_total = time.time()

        for chain in self.chains:
            start_time_chain = time.time()
            chain.fit(X, Y)  # Fit each chain independently
            end_time_chain = time.time()
            self.chain_train_times.append(end_time_chain - start_time_chain)  # Save time for this chain
        
        # Save total training time
        self.train_time_total = time.time() - start_time_total


    #=========================================================#
    #                                                         #
    #=========================================================#
    def predict_proba(self, X):
        """
        Compute the label probabilities for the input data.

        Parameters:
        - X: Input feature matrix (DataFrame or array-like).

        Returns:
        - Array of probabilities for each label.
        
        Example of usage:

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone
        from joblib import Parallel, delayed
        import time
        import numpy as np

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)
        X_test, _ = make_multilabel_classification(n_samples=10, n_features=20, n_classes=5, random_state=43)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains and 2 jobs for parallelization
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=2)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Predict probabilities on new test data
        probabilities = ecc_model.predict_proba(X_test)

        # Display the probabilities for each label
        print(probabilities)
        print(f"Total prediction time: {ecc_model.test_time_total:.2f} seconds")
        """
        if self.chains is None:
            raise NotFittedError('Model has not been fitted yet.')
        
        # Start total test time
        start_time_total = time.time()

        # Parallelized prediction probabilities
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(chain.predict_proba)(X) for chain in self.chains)
        
        # Save total test time
        self.test_time_total = time.time() - start_time_total

        return np.mean(predictions, axis=0)

 
    #=========================================================#
    #                                                         #
    #=========================================================#
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions based on the probability threshold.

        Parameters:
        - X: Input feature matrix (DataFrame or array-like).
        - threshold: Probability threshold to convert probabilities to binary (default: 0.5).

        Returns:
        - Array of binary predictions for each label.
        
        Example of usage:

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone
        import numpy as np

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)
        X_test, _ = make_multilabel_classification(n_samples=10, n_features=20, n_classes=5, random_state=43)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Predict binary labels with a threshold of 0.5
        predictions = ecc_model.predict(X_test, threshold=0.5)

        # Display the binary predictions for each label
        print(predictions)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


    #=========================================================#
    #                                                         #
    #=========================================================#
    def predict_cardinality(self, X, Y_train):
        """
        Make binary predictions using a threshold based on the label cardinality from the training set.

        Parameters:
        - X: Input feature matrix (DataFrame or array-like).
        - Y_train: Training multi-label target (to calculate cardinality).

        Returns:
        - Array of binary predictions for each label based on cardinality.
        
        Example of usage:

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone
        import numpy as np

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)
        X_test, _ = make_multilabel_classification(n_samples=10, n_features=20, n_classes=5, random_state=43)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Predict binary labels using the label cardinality from the training set
        predictions = ecc_model.predict_cardinality(X_test, Y_train)

        # Display the binary predictions based on label cardinality
        print(predictions)
        """
        cardinality = Y_train.sum(axis=1).mean() / Y_train.shape[1]
        probas = self.predict_proba(X)
        return (probas >= cardinality).astype(int)



    #=========================================================#
    #                                                         #
    #=========================================================#
    def model_size(self):
        """
        Return the size (in bytes) of each chain's model using sys.getsizeof.

        Returns:
        - List of sizes in bytes for each chain.
        
        Example of usage:

        import sys
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Get the size of each model in the ensemble
        model_sizes = ecc_model.model_size()

        # Display the size of each chain's model in bytes
        print(model_sizes)
        """
        sizes = [sys.getsizeof(chain) for chain in self.chains]
        return sizes



    #=========================================================#
    #                                                         #
    #=========================================================#
    def model_size_joblib(self, filename='ecc_model.pkl'):
        """
        Save the model to a file and return its size (in bytes).

        Parameters:
        - filename: Name of the file to save the model (default: 'ecc_model.pkl').

        Returns:
        - Size of the saved model file in bytes.
        
        Example of usage:

        import joblib
        import os
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Save the model to a file and get the size of the saved file in bytes
        model_file_size = ecc_model.model_size_joblib(filename='ecc_model.pkl')

        # Display the size of the saved model file in bytes
        print(f"Model file size: {model_file_size} bytes")
        """
        joblib.dump(self, filename)
        file_size = os.path.getsize(filename)
        return file_size



    #=========================================================#
    #                                                         #
    #=========================================================#
    def save_predictions(self, predictions, filename):
        """
        Save predictions to a CSV file.

        Parameters:
        - predictions: Predictions to save (binary or probabilistic).
        - filename: Name of the file to save the predictions (CSV format).

        Returns:
        - None
        
        Example of usage:

        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_multilabel_classification
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone

        # Creating a multi-label dataset
        X_train, Y_train = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, random_state=42)
        X_test, _ = make_multilabel_classification(n_samples=10, n_features=20, n_classes=5, random_state=43)

        # Base model: RandomForestClassifier
        base_model = RandomForestClassifier()

        # Initialize the ECC model with 5 chains
        ecc_model = ECC(model=base_model, n_chains=5, n_jobs=1)

        # Train the ECC model
        ecc_model.fit(X_train, Y_train)

        # Predict probabilities on new test data
        predictions = ecc_model.predict_proba(X_test)

        # Save the predictions to a CSV file
        ecc_model.save_predictions(predictions, 'predictions.csv')

        # Confirm the file has been created (Optional)
        import os
        if os.path.exists('predictions.csv'):
            print("Predictions saved to 'predictions.csv'")
        """
        import pandas as pd

        df = pd.DataFrame(predictions)
        df.to_csv(filename, index=False)
