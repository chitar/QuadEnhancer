## Dependencies

This code is modified from the commonsense reasoning task of [DoRA](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning). Please follow the instructions in the original repository to install the required dependencies and prepare the data.


## Finetuning

To fine-tune the LLaMA models, you need to run the corresponding scripts with different rank settings.

```bash
# rank r=32
sh llama_7B_LoRA_SAW.sh 32 64 ./llama_models/roll_r32 0 roll 1

sh llama2_7B_LoRA_SAW.sh 32 64 ./llama2_models/roll_r32 0 roll 1

sh llama3_8B_LoRA_SAW.sh 32 64 ./llama3_models/roll_r32 0 roll 1

# rank r=16
sh llama_7B_LoRA_SAW.sh 16 32 ./llama_models/roll_r16 0 roll 1

sh llama2_7B_LoRA_SAW.sh 16 32 ./llama2_models/roll_r16 0 roll 1

sh llama3_8B_LoRA_SAW.sh 16 32 ./llama3_models/roll_r16 0 roll 1



```