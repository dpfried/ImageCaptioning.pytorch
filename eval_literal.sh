#!/bin/bash

#model_dir="models/updown"
model_dir=$1

split=$2
beam_size=$3

if [ -z "$beam_size" ]
then
  beam_size=5
fi

id="literal_${split}_bs-${beam_size}"

model_path=${model_dir}/model-best.pth
output_path=${model_path}_${id}.out

python -u tools/eval.py \
	--id "blargh" \
  --force 1 \
  --dump_images 0 \
	--verbose_captions 0 \
	--verbose_beam 0 \
  --save_verbose_predictions 1 \
  --split $split \
  --num_images 5000 \
  --model ${model_path} \
  --infos_path ${model_dir}/infos*-best.pkl \
  --language_eval 1 \
  --beam_size $beam_size \
  --diversity_lambda 0.0 \
  | tee $output_path

  #| tee expts/${id}
