

## Dependencies

To run the project, you need the following dependencies:

- **Python**: 3.9.0
- **Transformers**: 4.51.3
- **Datasets**: 3.5.1
- **Torch**: 2.7.0


## Pretraining

To begin pretraining the model, run the following script:

```bash
sh pretrain-micro.sh
```

- The pretrained model will be saved under the `./models` directory.
- The best evaluation loss is logged at the end of `./models/$EXP_NAME/train_log.txt`.

## Fine-tuning

To fine-tune the pretrained model, use the following script:

```bash
sh finetune-micro.sh
```

- The fine-tuned model will be saved under the `./finetune-models` directory.
- The best evaluation accuracy is logged at the end of `./finetune-models/$EXP_NAME/train_log.txt`.
