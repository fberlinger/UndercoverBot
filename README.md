# Distributed Robot Fish

## Requirements

- Python 3.6
- Jupyter 1.0
- Numpy
- Scipy
- Matplotlib
- (PIP _not mandatory but recommended_)

## Installation

Either install Jupyter, Numpy, Scipy, and Matplotlib via PIP:

```
git clone https://code.harvard.edu/frl487/cs262-final && cd cs262-final
pip install -r ./requirements.txt
```

Or manually via https://jupyter.org/install and https://scipy.org/install.html

## Run

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open one of the following files ending in `.ipynb`:

- `Cohesion.ipynb`
- `Information Propagation.ipynb`
- `Hop Counts.ipynb`
- `Leader Election.ipynb`
- `Collective Sampling and Search.ipynb`

Please run each cell individually! **Warning**: Using `Run All` will not work
as the experiments start several threads for every fish and the code execution
async, hence, Jupyter Notebook runs new cells to quickly before others finished.
