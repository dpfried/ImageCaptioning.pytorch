#!/bin/bash

model_dir="models/updown"

split=$1
beam_size=$2

id="literal_${split}_bs-${beam_size}"

python -u tools/eval.py \
	--id $id \
  --force 1 \
  --dump_images 0 \
	--verbose_captions 0 \
	--verbose_beam 0 \
  --save_verbose_predictions 1 \
  --split $split \
  --num_images 5000 \
  --model ${model_dir}/model-best.pth \
  --infos_path ${model_dir}/infos_tds-best.pkl \
  --language_eval 1 \
  --beam_size $beam_size \
  | tee expts/${id}
