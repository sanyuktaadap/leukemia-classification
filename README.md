# Deep Learning Based B-ALL Classification

Using Perioheral Blood Smear Images to diagnose Breast Cancer as benign or malignant, and further classify three stages of malignant B-cell development in the bone marrow.

- To set up data for ML, split data using [this](./run_setup.py) script
- Create dataset class with [this](./dataset.py) script
- Create model class with [this](./model.py) script
- Use [this](./metrics.py) to import functions that calculate metrics
- Use [utils](./utils.py) to import function to start training, validation and testing

#### Train-Val-Test Scripts
- [Training and Validation](./train.py)
- [Testing](./test.py)
