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

To launch one or more agend

```
wandb agent <sweep_id>
```

## Visualizing results

From wandb.ai, and to see final results:

```
python3 results.py <sweep_id>
```

## Running a local instance

```
python3 EnD_rebias_mnist_sweep.py --local --alpha 0.1 --beta 0.1
```

More CLI arguments can be found in **configs.py**
