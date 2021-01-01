model_dir="models/updown"

# val or test
split=$1

# number of candidate captions to rescore
n_candidates=$2

# weight lambda in log p_s0 * lambda  + log p_s1 * (1 - lambda)
s0_weight=$3

# number of additional images
n_distractors=$4
if [ -z $n_distractors ]
then
	n_distractors=5
else
  shift
fi

shift 3;

distractor_split="train"

candidate_gen="bs"

id="pragmatic_${split}_cand-${candidate_gen}-${n_candidates}_s0-weight-${s0_weight}"

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
	--pragmatic_inference 1 \
	--pragmatic_distractors $n_distractors \
	--pragmatic_distractor_split $distractor_split \
	--pragmatic_s0_weight $s0_weight \
	--sample_n_method $candidate_gen \
	--sample_n $n_candidates \
  | tee expts/${id}
