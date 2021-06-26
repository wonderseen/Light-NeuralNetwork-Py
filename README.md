Re-inventing the wheel of light neural network system (pure python).

## Features maintain:
1. Basic Tensor computation, but in low speed which is due to the way Python interprets.
2. Forward and Backward processing of individual element, e.g., Dense, Sigmoid, etc.
3. A Threading dataloader.
4. Some basic initializers.
5. As an extra efforts, I turned a python-MLP into numpy-MLP, further explored a c-style interpreter JIT (built in numba-JIT) aiming for numpy acceleration, but no effeciency was seen as expected.

## TODO:
1. Writing-Reading Single-Threading dataloader -> Multi-threading loader for buffer reading.
2. Make a supplement for missing activations, BN, LN, etc.
3. Reproduce an Optimizers class.
4. Remanage all Classes in a Generic Programing way.