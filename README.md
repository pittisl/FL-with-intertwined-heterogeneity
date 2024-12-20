# Tackling the staleness in Federated learning with intertwined data and device heterogeneity 

## Introduction
Implementation of our work ([Tackling the staleness in Federated learning with intertwined data and device heterogeneity](https://arxiv.org/abs/2309.13536)) with pytorch. 

This work aim to mitigate staleness in FL scenarios where data and device heterogeneities are intertwined by converting a stale model update into a unstale one. The basic idea is to estimate the distributions of clients' local training data from their uploaded stale model updates, and use these estimations to compute unstale client model updates. In this way, our approach does not require any auxiliary dataset nor the clients' local models to be fully trained, and does not incur any additional computation or communication overhead at client devices.

We compared our approach with the existing FL strategies in two scenarios (fixed and variant data). 

## Requirments

* Python3
* Pytorch
* argparse
* torchvision


## General Usage

For the fixed data scenario, `fixed.py` partitions the non-iid data in a naive way (assign data in a specific class to a client):
```
python fixed.py
```
and `fixed_dirichlet.py` uses dirichlet distribution to sample non-iid client data:
```
python fixed_dirichlet.py
```

For the variant data scenario, you can run the experiment with `streaming_fed.py`. The dataset we use is Mnist and SVHN.
```
python streaming_fed.py
```

A folder named `data` will be automatically created, and the datasets will be downloaded into this folder. The supported datasets can be found in `sampling.py` and supported models can be found in `models.py`. When the training is finished, the per-epoch accuracy will be stored in the `save` folder.

The implementation details about the gradient inversion can be found in the inversefed folder. The functions about client data sampling and local training can be found in `sampling.py` and `update.py`.

## Parameter setting

The basic federated learning paremeters such as training epochs and the number of clients are set in `option.py`

Other parameters are set in the main files (`fixed.py` `streaming_fed.py` `fixed_dirichlet.py`):

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

 



## Acknowledgments

The code for gradient inversion in the inversefed folder is adapted from ([invertinggradients](https://github.com/JonasGeiping/invertinggradients)).  
We are grateful to the authors for their work and contributions.

If you use this repository, make sure to also review and comply with the licensing terms of the original project.

 ## Citation
 ```
@article{wang2023tackling,
  title={Tackling the Unlimited Staleness in Federated Learning with Intertwined Data and Device Heterogeneities},
  author={Wang, Haoming and Gao, Wei},
  journal={arXiv preprint arXiv:2309.13536},
  year={2023}
}
 ```


