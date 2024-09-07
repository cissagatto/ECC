# ECC - Ensemble Classifier Chains

Welcome to the ECC GitHub repository! This project aims to improve the implementation of the Ensemble of Classifier Chains method for multi-label classification problems. Below, you'll find instructions on how to get started with ECC and an example of how to use it.

## How to Cite

```plaintext
@misc{ECC2024,
  author = {Elaine Cecília Gatto},
  title = {ECC: A python implementation for Ensemble of Classifier Chains},  
  year = {2024},  
  doi = {},
  url = {https://github.com/cissagatto/ECC}
}
```

## Overview

ECC is a method that extends the Classifier Chains approach by incorporating an ensemble of base classifiers. This implementation leverages `scikit-learn` for the base model and provides additional functionalities for efficient training, prediction, and model evaluation.

## Installation

To start with `ECC` you must install the required packages. You can do this using `pip`:

```bash
pip install numpy pandas scikit-learn
```


## Usage

Here is a step-by-step example of how to use ECC with a `RandomForestClassifier`:

```python
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

# 10. Check the size of the models for each chain in bytes
model_sizes_2 = ecc.model_size_joblib()
print(f"Model sizes for each chain: {model_sizes_2}")

# 11. Display the first 5 binary predictions
print("First 5 binary predictions:")
print(binary_preds[:5])

# 12. Display the first 5 cardinality-based predictions
print("First 5 cardinality-based predictions:")
print(cardinality_preds[:5])

# 13. Display the first 5 predicted probabilities
print("First 5 predicted probabilities:")
print(probas[:5])
```

## Features

- **Multi-label Classification:** Handles multi-label problems effectively.
- **Model Customization:** Use any `scikit-learn` classifier as the base model.
- **Ensemble Method:** Combines predictions from multiple chains for improved performance.
- **Efficient Training:** Allows parallel processing and provides timing information.
- **Prediction Saving:** Save predictions to CSV files for further analysis.

## 📚 **Contributing**

If you'd like to contribute to ECC's development, feel free to open an issue or submit a pull request. We welcome contributions that enhance ECC's functionality and performance.

## 📧 **Contact**

For any questions or support, please contact:
- **Prof. Elaine Cecilia Gatto** (elainececiliagatto@gmail.com)
  

## Acknowledgment
- This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001.
- This study was partly financed by the Conselho Nacional de Desenvolvimento Científico e Tecnológico - Brasil (CNPQ) - Process number 200371/2022-3.
- The authors also thank the Brazilian research agency FAPESP for financial support.


# Links

| [Site](https://sites.google.com/view/professor-cissa-gatto) | [Post-Graduate Program in Computer Science](http://ppgcc.dc.ufscar.br/pt-br) | [Computer Department](https://site.dc.ufscar.br/) |  [Biomal](http://www.biomal.ufscar.br/) | [CNPQ](https://www.gov.br/cnpq/pt-br) | [Ku Leuven](https://kulak.kuleuven.be/) | [Embarcados](https://www.embarcados.com.br/author/cissa/) | [Read Prensa](https://prensa.li/@cissa.gatto/) | [Linkedin Company](https://www.linkedin.com/company/27241216) | [Linkedin Profile](https://www.linkedin.com/in/elainececiliagatto/) | [Instagram](https://www.instagram.com/cissagatto) | [Facebook](https://www.facebook.com/cissagatto) | [Twitter](https://twitter.com/cissagatto) | [Twitch](https://www.twitch.tv/cissagatto) | [Youtube](https://www.youtube.com/CissaGatto) |

----
Happy coding! 🚀
