# DP-GNNs
## Differentially private graph neural networks (GNNs) for whole-graph classification tasks.

This repo contains code to train graph neural networks for graph classification tasks with differential privacy (DP). Check out our [paper](https://ieeexplore.ieee.org/abstract/document/9980390) for the details about the methods.

### Abstract

Graph Neural Networks (GNNs) have established themselves as state-of-the-art for many machine learning applications such as the analysis of social and medical networks. Several among these datasets contain privacy-sensitive data. Machine learning with differential privacy is a promising technique to allow deriving insight from sensitive data while offering formal guarantees of privacy protection. However, the differentially private training of GNNs has so far remained under-explored due to the challenges presented by the intrinsic structural connectivity of graphs. In this work, we introduce a framework for differential private graph-level classification. Our method is applicable to graph deep learning on multi-graph datasets and relies on differentially private stochastic gradient descent (DP-SGD). We show results on a variety of datasets and evaluate the impact of different GNN architectures and training hyperparameters on model performance for differentially private graph classification, as well as the scalability of the method on a large medical dataset. Our experiments show that DP-SGD can be applied to graph classification tasks with reasonable utility losses. Furthermore, we apply explainability techniques to assess whether similar representations are learned in the private and non-private settings. Our results can also function as robust baselines for future work in this area.

### Repo Structure

**the ``src`` folder:**
- **experiments**: The experiments folder contains the Python scripts to run different experiments (e.g. ``synthetic_dataset.py`` runs the experiment on the synthetic dataset)
- **utils**: the utils folder contains necessary utilities. The ``config.py`` file contains some general settings for the setup, ``training_utils.py`` contains utilities for training, ``utils.py`` some general utils, and ``generat_synthetic_dataset.py`` creates the synthetic dataset.
- **datasets**: The datasets folder contains code to construct the datasets
- **models**: The models folder contains the code to define the Graph Neural Network (GNN) models

**the base folder:**
In the base folder you can find a ``yaml`` file to construct the environment.

### Running an Experiment
In order to run the first experiment on the synthetic dataset, the following steps are needed:

1. Install the environment:

``conda env create --file=environment.yml``

2. Run the following code for the default non-DP run on the **synthetic** dataset:

``python3 src/experiments/synthetic_dataset.py``

3. By altering the parameters, you can change the run settings. E.g. to run in a DP setting with a maximum privacy budget of $\varepsilon=5$, run the following code:

``python3 src/experiments/synthetic_dataset.py --dp 1 --max_epsilon 5``

### Citing
If you use this work, please cite the following paper:

```
@article{mueller2022differentially,
  title={Differentially private graph neural networks for whole-graph classification},
  author={Mueller, Tamara T and Paetzold, Johannes C and Prabhakar, Chinmay and Usynin, Dmitrii and Rueckert, Daniel and Kaissis, Georgios},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={6},
  pages={7308--7318},
  year={2022},
  publisher={IEEE}
}
```
