# FSGAMLR

Feature selection by Genetic Algorithm-Multiple Linear Regression implemented in Python

## Dependencies

* Python
* numpy
* matplotlib
* scikit-learn

## Usage

```python
import fsgamlr
import pandas as pd
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

X = dataset.data
y = dataset.target

fsgamlr = fsgamlr.GeneticAlgorithm(X, y, 
                                   max_features=5, 
                                   population_size=100,
                                   n_generation=50)

result = fsgamlr.optimize(verbose=0)
fsgamlr.plot_result()
```