# Reward-Augmented Decoding

This repository contains the code for EMNLP 2023 paper [Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model](https://openreview.net/forum?id=I13VHLJjLO).

## Important !
After you've cloned the repo, create a new conda environment for RAD and activate it
```
cd RAD
conda env create -f environment.yml
conda activate rad_env
```

Build the project
```
pip install -e .
```

If you want to try our toxicity and sentiment reward models, make sure you have `gdown` installed, and run
```
cd reward_modeling
gdown https://storage.googleapis.com/rad_release/saved_models.zip
unzip saved_models.zip && rm saved_models.zip
```
To train a toxicity reward model with `jigsaw_unintended_bias` dataset, you have to download it manually from Kaggle: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data. Then, specify the dataset path `jigsaw_dir: PATH/TO/JIGSAW` in `reward_modeling/configs/config_rm.yaml`.

## Train Your Own Reward Model
Add custom dataset to `utils/get_one_dataset`, make sure it has only two attributes, "text" and "labels." Then, go to `reward_modeling/`, specify training details in `reward_modeling/configs/config_rm.yaml`. For example, to train a reward model for sentiment steering task, run the following code
```
python trainer_rm.py \
	--configs rm_sentiment gpt2-small \
	--wandb_entity WANDB_ID
```
To disable wandb, set `log_wandb` to `false` in `config_rm.yaml`.

## Sentiment
To run sentiment-controlled generation experiment, run the command
```
OUTPUT_DIR=outputs/
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50
INVERSE=False

python eval_sentiment.py \
	$OUTPUT_DIR \
	--batch_size $BATCH_SIZE \
	--lm $LANGUAGE_MODEL \
	--topk $TOPK \
	--beta $BETA \
	--inverse $INVERSE
```
Specify prompt type by assigning `DATASET` to one of `[negative, neutral, positive]`. You can adjust steering direction by setting `inverse` to either `True` or `False` --- for `inverse=True`, RAD steers generation toward lower reward.
Specify `--test True` to run only 100 examples.


## Toxicity
Add your **Perspective API KEY** to `utils/perspective_api.py` and adjust the `QUOTA_IN_QPS` according to your quota. Current `RateLimiter` is set for 1QPS, which is not optimal. Perspective API increases quota to 100QPS upon [request](https://developers.perspectiveapi.com/s/request-quota-increase?language=en_US).

To run detoxification experiment, run the command
```
OUTPUT_DIR=outputs/
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50
INVERSE=True

python eval_toxicity.py \
	$OUTPUT_DIR \
	--batch_size $BATCH_SIZE \
	--lm $LANGUAGE_MODEL \
	--topk $TOPK \
	--beta $BETA \
	--inverse $INVERSE
```
Here, we set `inverse=True` to make RAD generate text with low toxicity. 


## Custom Task
For a custom task, finetune a task-specific reward model with `reward_modeling/trainer_rm.py`. Use `generate.py` to perform reward augmented decoding and follow `eval_sentiment.py` to evaluate performance.


## Reminder
- Run `pip install -e .` everytime you made changes to the sub-modules.
- The code supports multi-gpu decoding by hosting a copy of the reward model on each gpu and evaluate rewards separately.
- In case `RewardAugmentedLogitsProcessor` doesn't function properly, try initiating `RewardAugmentedDecoder` with `efficient=False` to run decoding without tracking `past_key_values`.
- We suggest running the code on linux server with `Ubuntu-v20.04` to reproduce exact experiment results.


## Citation
```
@inproceedings{
anonymous2023rewardaugmented,
title={Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model},
author={Anonymous},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=I13VHLJjLO}
}
```