


## Dependencies

This code is modified based on [CVNets](https://github.com/apple/ml-cvnets). Please follow the original repository's instructions to install the necessary dependencies.

## Data Preparation

- **Pretraining**: The ImageNet-1k dataset should be prepared first, consisting of `train` and `val` directories.
- **Fine-tuning**: Organize all fine-tuning datasets in the ImageNet format under `$finetuneDataDir`.

## Pretraining on ImageNet-1k

Before starting pretraining, make sure to modify the `root_train` and `root_val` paths in all `.yaml` files under `./scripts`. Then, run the following scripts for pretraining:

### Pretraining Scripts

```bash
sh ./scripts/run_vit-extiny-linear.sh
sh ./scripts/run_vit-extiny-roll.sh

sh ./scripts/run_vit-micro-linear.sh
sh ./scripts/run_vit-micro-roll.sh

sh ./scripts/run_vit-tiny-linear.sh
sh ./scripts/run_vit-tiny-roll.sh
```

The results of pretraining can be found under the `./formal_res` directory.

## Fine-tuning

Before starting fine-tuning, modify the `$finetuneDataDir` variable in all `.sh` scripts under the `./scripts/finetune` directory to point to the location of the fine-tuning datasets. Also, ensure that `$setname` corresponds to each fine-tuning dataset.

### Fine-tuning Scripts

```bash
sh ./scripts/finetune/finetune_vit-extiny-linear.sh
sh ./scripts/finetune/finetune_vit-extiny-roll.sh

sh ./scripts/finetune/finetune_vit-micro-linear.sh
sh ./scripts/finetune/finetune_vit-micro-roll.sh

sh ./scripts/finetune/finetune_vit-tiny-linear.sh
sh ./scripts/finetune/finetune_vit-tiny-roll.sh
```

The results of fine-tuning can be found in the `./formal_res/finetuning` directory.
