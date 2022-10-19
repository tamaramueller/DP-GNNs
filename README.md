# DP-GNNs
## Differentially private graph neural networks (GNNs) for whole-graph classification tasks.

This repo contains code to train graph neural networks for graph classification tasks with differential privacy (DP). Check out our [paper](https://arxiv.org/pdf/2202.02575.pdf) for the details about the methods.

### Abstract

Graph Neural Networks (GNNs) have established themselves as the state-of-the-art models for
many machine learning applications such as the analysis of social networks, protein interactions
and molecules. Several among these datasets contain privacy-sensitive data. Machine learning with
differential privacy is a promising technique to allow deriving insight from sensitive data while
offering formal guarantees of privacy protection. However, the differentially private training of
GNNs has so far remained under-explored due to the challenges presented by the intrinsic structural
connectivity of graphs. In this work, we introduce differential privacy for graph-level classification,
one of the key applications of machine learning on graphs. Our method is applicable to deep learning
on multi-graph datasets and relies on differentially private stochastic gradient descent (DP-SGD). We
show results on a variety of synthetic and public datasets and evaluate the impact of different GNN
architectures and training hyperparameters on model performance for differentially private graph
classification. Finally, we apply explainability techniques to assess whether similar representations
are learned in the private and non-private settings and establish robust baselines for future work in
this area.

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

``conda env create --file=environment.yml``

2. Run the follwing code for the default non-DP run on the **synthetic** dataset:

``python3 src/experiments/synthetic_dataset.py``

3. By altering the parameters you can change the settings of the run. E.g. to run in in a DP setting with a maximum privacy budget of $\varepsilon=5$, run the following code:

``python3 src/experiments/synthetic_dataset.py --dp 1 --max_epsilon 5``

### Citing
If you use this work, please cite the following paper:

```
@article{mueller2022differentially,
  title={Differentially Private Graph Classification with GNNs},
  author={Mueller, Tamara T and Paetzold, Johannes C and Prabhakar, Chinmay and Usynin, Dmitrii and Rueckert, Daniel and Kaissis, Georgios},
  journal={arXiv preprint arXiv:2202.02575},
  year={2022}
}
```
