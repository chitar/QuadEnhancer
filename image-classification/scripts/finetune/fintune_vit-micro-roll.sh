epoch=100
pretrained_model=".formal_res/vit-micro-roll/checkpoint_ema_best.pt"
finetuneDataDir="./finetuneData"

for setname in caltech_c102 cifar_c10 cifar_c100 flowers_c102 food_c101 pet_c37

do

n_class=$(echo "$setname" | grep -oE '[0-9]+')
run_label="fintune-vit-micro-roll-$setname"

PYTHONPATH=. cvnets-train \
 --dataset.root-train                      "$finetuneDataDir/$setname/train"           \
 --dataset.root-val                         "$finetuneDataDir/$setname/val"            \
\
 --common.run-label                     $run_label                                                    \
 --scheduler.max-epochs               $epoch                                                                  \
 --model.classification.pretrained   $pretrained_model                                               \
 --model.classification.n-classes      $n_class                                                                                 \
\
 --common.config-file scripts/finetune/finetune-vit-tiny.yaml \
 --common.results-loc formal_res/finetuning \
 --saw_method roll --saw_k 1  \
 | tee    "./formal_res/finetuning/$run_label.log"

done

