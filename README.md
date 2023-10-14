


# Reward-Augmented Decoding

This repository contains the code for EMNLP 2023 paper [Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model](https://openreview.net/forum?id=I13VHLJjLO).

## Important !
Create a conda environment for RAD by running
```
bash install_env.sh
conda activate rad_env
```

Build the project,
```
cd RAD
pip install -e .
```

Download crucial components---datasets and trained reward model.
```

```


## Sentiment
To run sentiment-controlled generation experiment, run the command
```
OUTPUT_DIR=outputs/
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50
DATASET=negative

python eval_sentiment.py \
$OUTPUT_DIR \
--batch_size $BATCH_SIZE \
--lm $LANGUAGE_MODEL \
--topk $TOPK \
--beta $BETA \
--dataset $DATASET \
```
Specify prompt type by assigning `DATASET` to one of `[negative, neutral, positive]`. Pick steering direction by setting `TOWARD_POSITIVE` to either `True` or `False`.


## Toxicity
Add your Perspective API KEY to `utils/perspective_api.py` and adjust the `QUOTA_IN_QPS` according to your quota. Current `RateLimiter` is set for 1QPS, which is not optimal. Perspective API increases quota to 100QPS upon [request](https://developers.perspectiveapi.com/s/request-quota-increase?language=en_US).

To run detoxification experiment, run the command
```
OUTPUT_DIR=outputs/
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50

python eval_toxicity.py \
$OUTPUT_DIR \
--batch_size $BATCH_SIZE \
--lm $LANGUAGE_MODEL \
--topk $TOPK \
--beta $BETA \
```
Specify `--test True` to run only 100 examples.


## Custom Task
For a custom task, finetune a task-specific reward model using `reward_modeling/trainer_rm.py`. Use `generate.py` to perform reward augmented decoding and follow `eval_sentiment.py` to evaluate performance.


## Reminder
- Run `pip install -e .` everytime you made changes to the sub-modules.
- The code supports multi-gpu decoding by hosting a copy of the reward model on each gpu and evaluate rewards separately.
- In case `RewardAugmentedLogitsProcessor` doesn't function properly, try initiating `RewardAugmentedDecoder` with `efficient=False` to run decoding without tracking `past_key_values`.
- We suggest running the code on linux server with `Ubuntu-v20.04` to reproduce experiment results.


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