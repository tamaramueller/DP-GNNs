# DP-GNNs
## Differentially private graph neural networks (GNNs) for whole-graph classification tasks.

This repo contains code to train graph neural networks for graph classification tasks with differential privacy (DP). Check out our [paper](https://arxiv.org/pdf/2202.02575.pdf) for the details about the methods.

### Repo Structure

**the ``src`` folder:**
- **experiments**: The experiments folder contains the python scripts to run different experiments (e.g. ``synthetic_dataset.py`` runs the experiment on the synthetic dataset)
- **utils**: the utils folder contains necessary utilities. The ``config.py`` file contains some general settings for the setup, ``training_utils.py`` contains utilities for training, ``utils.py`` some general utils, and ``generat_synthetic_dataset.py`` creates the synthetic dataset.
- **datasets**: The datasets folder contians code to construct the datasets
- **models**: The models folder contains the code to define the Graph Neural Network (GNN) models

**the base folder:**
In the base folder you can find a ``yaml`` file to construct the environment

### Running an Experiment
In order to run the first experiment on the synthetic dataset, the following steps are needed:

1. Install the environment:

``conda env create --file=environment.yaml``

2. Run the follwing code for the default non-DP run on the **synthetic** dataset:

``python src/experiments/synthetic_dataset.py``

3. By altering the parameters you can change the settings of the run. E.g. to run in in a DP setting with a maximum privacy budget of $\varepsilon=5$, run the following code:

``python src/experiments/synthetic_dataset.py --dp 1 --max_epsilon 5``
