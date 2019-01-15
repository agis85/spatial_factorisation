# Spatial Factorisation

Code for the paper [Factorised spatial representation learning: application in semi-supervised myocardial segmentation].

The main files are:

* sdnet.py: the model implementation
* sdnet_trainer.py: code related to running an experiment

Data loaders are stored in the _loaders_ package. To define a new loader, extend class `base_loader.Loader`, and add initialisation in loader_factory.py. The data folder location can be specified in parameters.py.

The main method is in main.py and arguments can be passed at runtime. For example, an experiment can be run with:
```
python main.py --dataset acdc --split 0 --ul_mix 1 --l_mix 0.5
```

`--split` defines the cross validation data split, `--ul_mix` the percentage of unlabelled data, and `--l_mix` the percentage of labelled images. These proportions are calculated by comparing with the total number of labelled images in the dataset.

[Factorised spatial representation learning: application in semi-supervised myocardial segmentation]: https://link.springer.com/chapter/10.1007/978-3-030-00934-2_55
