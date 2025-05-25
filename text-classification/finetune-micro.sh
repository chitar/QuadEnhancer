

epoch=10

for model in GPT2-Micro16 GPT2-Micro32
do
for setname in imdb yelp_polarity  ag_news  glue,cola  glue,sst2   emotion

do

echo $model-$setname
python finetune.py \
  --saw_method $1 \
  --saw_k $2 \
  --epoch $epoch \
  --model $model \
  --dataset $setname
done
done
