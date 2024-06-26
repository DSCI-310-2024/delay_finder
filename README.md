# delay_finder

[![codecov](https://codecov.io/gh/DSCI-310-2024/delay_finder/graph/badge.svg?token=uO4Pe7Dg7w)](https://codecov.io/gh/DSCI-310-2024/delay_finder)

Authors: Siddharth Balodi, Charles Benkard, Mikel Ibarra Gallardo, and Stephanie Ta.

`delay_finder` is DSCI 310 Group 17's Python package for analyzing, predicting, and visualizing data related to airline delay.

It has functions to read CSV files into a dataframe, filter dataframe columns, replace a value in a dataframe, split data into 80/20 testing and training sets, save a model as a pickle object and file, and make a histogram visualizing a numeric feature.

This package builds off of [pandas](https://github.com/pandas-dev/pandas), [sci-kit learn](https://github.com/scikit-learn/scikit-learn), [altair](https://github.com/altair-viz/altair) packages and Python's [pickle](https://github.com/python/cpython/blob/main/Lib/pickle.py) library and makes workflow more efficient in analyzing, predicting, and visualizing data related to airline delay.

## Installation

```bash
$ pip install delay_finder
```

## Usage

`delay_finder` has multiple functions that can be used to analyze and visualize airline delay data.

Here are usage examples of two of our functions:
``` {python}
import pandas as pd
from delay_finder.filter_columns import filter_columns
from delay_finder.replace_value import replace_value

# Read in example data
df = pd.read_csv('candy_example_data.csv')

# Filter the example data to only have columns 'candy' and 'amount'
filtered_df = filter_columns(df, ['candy', 'amount'])

# Replace a value in a column of the example data, specifically 4 with 11 in the 'amount' column.
df_replace_kitkat_amount = replace_value(df, 'amount', 4, 11)
```

For usage examples of each function, please navigate to this [file in our repository](https://github.com/DSCI-310-2024/delay_finder/blob/main/docs/example.ipynb).

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`delay_finder` was created by Siddharth Balodi, Charles Benkard, Mikel Ibarra Gallardo, and Stephanie Ta. The code is licensed under the terms of the MIT license.  
The usage examples file in licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).
For more information, please see the license file.

## Credits

`delay_finder` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).  
The documentation and ci-cd workflow was adapted from Daniel Chen's [pycounts-dan](https://github.com/chendaniely/pycounts-dan) repository.
