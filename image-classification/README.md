## Dependencies

These codes are modified based on [CVNets](https://github.com/apple/ml-cvnets), so please install the dependencies according to its instructions.


## Data preparation

- For pretraining, the ImageNet-1k dataset should be prepared first, which consists of `train` and `val` directories.
- For finetuning, all datasets should be organized as ImageNet format under `$finetuneDataDir`

## Pretaining on ImageNet-1k
first modify the `root_train` and `root_train` paths of all `.yaml` files, then run the following scripts for pretraining


```bash
sh ./scripts/run_vit-extiny-linear.sh
sh ./scripts/run_vit-extiny-roll.sh

sh ./scripts/run_vit-micro-linear.sh
sh ./scripts/run_vit-micro-roll.sh

sh ./scripts/run_vit-tiny-linear.sh
sh ./scripts/run_vit-tiny-roll.sh


```

the results can be found under `./formal_res`



## Finetuning

first modify the variables `$finetuneDataDir` in all `.sh` scripts under `./scripts/finetune` to be the directory of the finetune datasets, and make sure `$setname` correspinds to each finetuning datasets, then run the following scripts

```bash
sh ./scripts/finetune/fintune_vit-extiny-linear.sh
sh ./scripts/finetune/fintune_vit-extiny-roll.sh

sh ./scripts/finetune/fintune_vit-micro-linear.sh
sh ./scripts/finetune/fintune_vit-micro-roll.sh

sh ./scripts/finetune/fintune_vit-tiny-linear.sh
sh ./scripts/finetune/fintune_vit-tiny-roll.sh

```

the results can be found in `./formal_res/finetuning`
