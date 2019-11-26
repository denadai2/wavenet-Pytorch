# Original WaveNet - Pytorch

I started implementing the base architecture of [WaveNet](https://arxiv.org/abs/1609.03499) for Pytorch 1.3.x. 
This implementation is based on:

* [TensorFlow WaveNet](https://github.com/ibab/tensorflow-wavenet)
* [PyTorch Template](https://github.com/victoresque/pytorch-template)

## TODO
- [ ] Local conditioning
- [x] Global conditioning
- [ ] Dataset Example

## Run

    python3 train.py --config config.json
