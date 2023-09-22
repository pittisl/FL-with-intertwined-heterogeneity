# Tackling the staleness in Federated learning with intertwined data and device heterogeneity 

## Introduction
Implementation of our work Tackling the staleness in Federated learning with intertwined data and device heterogeneity with pytorch

Experiments are conducted in two scenarios (fixed and streaming data)

For the fixed data scenario, fixed.py partitions the non-iid data in a naive way (assign data in a specific class to a client) and fixed_dirichlet.py uses dirichlet distribution to sample non-iid client data. Two datasets Mnist and Cifar-10 are available.

For the fixed data scenario, you can run the experiment with streaming_fed.py. The dataset we use is Mnist and SVHN.

The implementation details about the gradient inversion can be found in the inversefed folder. The functions about client data sampling and local training can be found in sampling.py and update.py.

## Parameter setting

The basic federated learning paremeters such as training epochs and the number of clients are set in option.py

Other parameters are set in the main files (fixed.py streaming_fed.py fixed_dirichlet.py):

**DC**: method for delay compensate

dc = 1: gradient inversion based estimation

dc = 2: direct aggregation with staleness

dc = 3: first order compensation in DC-ASGD

dc = 4: weighted aggregation with staleness

dc = 0: FL without staleness

**GI_interation**: number of iterations in gradient inversion when using gradient inversion based estimation

**num_image_rec**: the size of D_rec

**problematic_class**: selected class which is affected by staleness most

**delay**: staleness of the problematic clients

 


## Requirments

* Python3
* Pytorch
* argparse
* torchvision




