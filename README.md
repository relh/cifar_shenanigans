# cifar_shenanigans

Run `python main.py --setting default` for normal cifar100 ResNet18.

Run `python main.py --setting cerebus` for a 3-way split in every ResBlock
(now SwitchBlock) between ReLU, Sigmoid, and Tanh.

You might need `pip install tqdm` plus a few more.
