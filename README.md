# CNN-SQA - Speech Quality Assessment Model

A Convolutional Neural Network (CNN) model to address the automatic non-reference speech quality assessment problem. It was trained, validated and tested through public databases and its accuracy was evaluated against classical models such as PESQ, ViSQOL and P.563.

## Technologies
* Python 2.7.15
* Tensorflow 1.11.0

<!-- ## Setup
```
# update/upgrade
apt get update && apt get upgrade

# install nvidia driver
apt install nvidia-driver-<version>

``` -->

## Configuration
A `config.py` file is used to define the model execution settings. Be carefully, specifing an invalid parameter or a nonexistent path could result in an unexpected behavior. The default parameters are defined in the `config.py` file, so to run in the default mode it is needed just call the train file.

## How to run
```
# install dependencies
$ pip install -U scikit-learn

# to train
$ python src/train.py
```

## Running tests
```
# install dependencies
$ pip install mock

# run tests
$ cd cnn_sqa
$ python -m unittest discover . "*_test.py"
```

## How to cite

* https://link.springer.com/article/10.1007/s00521-021-05767-4#citeas

## Acknowledgements
* The authors would like to thank the support of NVIDIA Corporation with the donation of the Titan XP GPU used for this research.

## Todo
* Update Python to 3.x.x
* Update Tensorflow to 2.x.x
* Add setup section
* Adjust packages