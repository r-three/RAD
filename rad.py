from transformers import (
    LogitsProcessorList,
    TopPLogitsWarper,
)
from utils.logits_processor import (
    RewardAugmentedLogitsProcessor,
    RewardAugmentedLogitsProcessorNoPkv
)


class RewardAugmentedDecoder():
    
    def __init__(self, language_model, lm_tokenizer, reward_model, rm_tokenizer, 
                 max_length, num_gpus=4, inverse=False, efficient=True):
        self._lm = language_model
        self._lm_tokenizer = lm_tokenizer
        self._rm = reward_model
        self._rm_tokenizer = rm_tokenizer
        self._max_length = max_length
        self._num_gpus = num_gpus
        self._inverse = inverse
        self._efficient = efficient

    def sample(
            self, 
            prompts,
            max_new_tokens=20,
            topk=20,
            num_return_sequences=25, 
            method="linear",
            beta=30,
            return_continuation_only=True,
            data_container=None
        ):
        input_ids = self._lm_tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length-max_new_tokens,
        ).to('cuda')
        
        # dry run
        if not self._rm:
            outputs = self._lm.generate(
                **input_ids,
                # min_new_tokens=2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
                num_return_sequences=num_return_sequences,
            )
        else:
            if self._efficient:
                logits_processor = LogitsProcessorList([
                    # TopPLogitsWarper(top_p=0.9),
                    RewardAugmentedLogitsProcessor(
                        self._lm_tokenizer,
                        self._rm_tokenizer,
                        self._rm,
                        topk=topk,
                        method=method,
                        beta=beta,
                        num_gpus=self._num_gpus,
                        inverse=self._inverse,
                        data_container=data_container
                    ),
                ])
                
            else:
                logits_processor = LogitsProcessorList([
                    # TopPLogitsWarper(top_p=0.9),
                    RewardAugmentedLogitsProcessorNoPkv(
                        self._lm_tokenizer,
                        self._rm_tokenizer,
                        self._rm,
                        topk=topk,
                        method=method,
                        beta=beta,
                        inverse=self._inverse,
                    ),
                ])
                
            outputs = self._lm.generate(
                **input_ids,
                logits_processor=logits_processor,
                # min_new_tokens=2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
                num_return_sequences=num_return_sequences,
            )
            
        if return_continuation_only:
            input_length = len(input_ids.input_ids[0])
            outputs = outputs[:, input_length:]          # remove prompt
            
        ret = self._lm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ret = [ret[i:i+num_return_sequences] for i in range(0, len(ret), num_return_sequences)]
        
        return ret
