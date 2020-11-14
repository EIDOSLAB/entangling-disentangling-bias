# EnD: Entangling and Disentangling deep representations for bias correction

## Prerequisites

```
pip3 install -r requirements.txt
```

## Running

**Note**: Anonymous mode for https://wandb.ai seems not to be working at the moment. An account is suggested

To create the sweep (hyperparameter search):

```
wandb sweep sweeps/biased_mnist_abs.yaml
```

To change the value of rho, edit sweeps/biased_mnist_abs.yaml accordingly.
We recommend searching alpha and beta in the range [0;1] for rho=0.990,0.995,0.997 and
[0;50] for rho=0.999.

To launch one or more agents

```
wandb agent <sweep_id>
```

We recommend at least 20 runs, to let the hyperparameters search converge.

## Visualizing results

From wandb.ai, and to see final results:

```
python3 results.py <sweep_id>
```

Example output:

```
    valid_acc  biased_test_acc  unbiased_test_acc     state        id
51   0.839778           0.9997             0.8501  finished  wtb1t9kr
55   0.825611           0.9996             0.8361  finished  u5oslxfp
51   0.821111           0.9999             0.8249  finished  j8wj5jyg
. . .
. . .
------- SUMMARY FOR EnD-cvpr21/Biased MNIST - rho 0.997 - ABS - valid -------
biased_test_acc: 99.97 ± 0.01
unbiased_test_acc: 83.70 ± 1.03

```

## Running a local instance

```
python3 EnD_rebias_mnist_sweep.py --local --alpha 0.1 --beta 0.1
```

More CLI arguments can be found in **configs.py**
