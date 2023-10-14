import torch
import numpy as np
from transformers import LogitsProcessor
import time
from queue import Queue
from threading import Thread
import copy


class RewardAugmentedLogitsProcessor(LogitsProcessor):
    '''
        This class is used to process logits of the language model at every timestep.
        It will load a copy of reward model on each GPU and take care of past_key_values.
    '''
    
    def __init__(self, lm_tokenizer, rm_tokenizer, reward_model, topk=20, 
                 method="linear", beta=30, num_gpus=4, inverse=False, data_container=None):
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = rm_tokenizer
        self._reward_model = reward_model
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse
        self._num_gpus = num_gpus
        self._past_key_values = [None]*self._num_gpus
        self._previous_input_ids_to_topk_idx = {}           # (batch, dict{input_id: topk_idx}), get last non-zero inputid
        self._step = 0
        self._attention_mask = [None]*self._num_gpus        # (batch x topk, sequence_length)
        self._reward_models = []
        self._data_container = data_container
        for i in range(self._num_gpus):
            model_copy = copy.deepcopy(self._reward_model)
            model_copy = model_copy.to(f'cuda:{i}')
            self._reward_models.append(model_copy)
            

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        '''
            past_key_values:
                Tuple of length config.n_layers, each containing tuples of tensors of shape
                (batch_size, num_heads, sequence_length, embed_size_per_head).
        '''
        def process_prompts(q: Queue[int]):
            gpu_id = q.get()
            batch_size = scores.shape[0]
            rows_per_gpu = int(np.ceil(batch_size * self._topk / self._num_gpus))
            start = gpu_id * rows_per_gpu
            end = min(start+rows_per_gpu, batch_size*self._topk)

            input_prompts_partition = input_prompts[start: end]
            past_key_values_part, attention_mask_part = self.get_past_key_values(input_prompts_partition, gpu_id, max_prompt_length)
            # on different devices
            self._past_key_values[gpu_id] = past_key_values_part
            self._attention_mask[gpu_id] = attention_mask_part
            q.task_done()

        def do_normal_task(q: Queue[int]):
            gpu_id = q.get()
            batch_size = scores.shape[0]
            rows_per_gpu = int(np.ceil(batch_size * self._topk / self._num_gpus))
            start = gpu_id * rows_per_gpu
            end = min(start+rows_per_gpu, batch_size*self._topk)

            candidate_tokens_partition = candidate_tokens[start: end]
            reward_scores_part, self._past_key_values[gpu_id], self._attention_mask[gpu_id] = self.get_reward(
                candidate_tokens_partition, self._past_key_values[gpu_id], self._attention_mask[gpu_id], gpu_id, max_candidate_length
            )
            reward_scores[gpu_id] = reward_scores_part.to('cuda')
            q.task_done()

        with torch.inference_mode():
            topk_scores, topk_ids = torch.topk(scores, self._topk, dim=-1)           # (batch, topk,)
            reward_scores = [None]*self._num_gpus
            last_selected_topk_indices = []
            max_prompt_length = -1
            max_candidate_length = -1

            # prepare pkv and attn_mask
            if self._step == 0:
                '''
                    1. repeat prompt topk times
                    2. get prompt pkv and attn_mask
                '''
                input_prompts = self._lm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                max_prompt_length = self._rm_tokenizer.batch_encode_plus(
                    input_prompts,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.shape[1]
                
                input_prompts = [element for element in input_prompts for i in range(self._topk)]       # (batch x topk, )

                q = Queue()
                for i in range(self._num_gpus):
                    q.put(i)
                for i in range(self._num_gpus):
                    worker = Thread(target=process_prompts, args=(q,))
                    worker.start()
                q.join()

            else:
                '''
                    1. use dict to find which token is chosen in last step
                    2. select that pkv and broadcast, select that attn_mask and broadcast
                '''
                for i, (input_ids_i, input_ids_to_topk_idx_dict_i), in enumerate(zip(input_ids, self._previous_input_ids_to_topk_idx)):
                    # skip if eos is being generated
                    if input_ids_i[-1]==self._lm_tokenizer.eos_token_id:
                        last_selected_topk_indices.append(-1)
                        continue
                    last_selected_topk_idx = input_ids_to_topk_idx_dict_i[input_ids_i[-1].item()]
                    last_selected_topk_indices.append(last_selected_topk_idx)
                    batch_size = scores.shape[0]

                    # for example i, update its pkv and attn_mask on corresponding gpu(s)
                    rows_per_gpu = int(np.ceil(batch_size * self._topk / self._num_gpus))
                    start, end = i*self._topk, (i+1)*self._topk-1

                    start_gpu, end_gpu = start//rows_per_gpu, end//rows_per_gpu
                    start_idx, end_idx = start%rows_per_gpu, end%rows_per_gpu

                    selected_token_gpu = (start+last_selected_topk_idx)//rows_per_gpu
                    selected_token_idx = (start+last_selected_topk_idx)%rows_per_gpu

                    while start_gpu < end_gpu:
                        rows = self._attention_mask[start_gpu].shape[0] # rows might be different from rows_per_gpu since the last gpu might have less rows
                        self._attention_mask[start_gpu][start_idx:, :] = self._attention_mask[selected_token_gpu][selected_token_idx, :].repeat(
                            rows-start_idx, 1)
                        if start_gpu==selected_token_gpu:
                            for layer_kv in self._past_key_values[start_gpu]:
                                for e in layer_kv:
                                    e[start_idx:, :, :, :] = e[selected_token_idx, :, :, :].unsqueeze(0).repeat(
                                        rows-start_idx, 1, 1, 1)
                        else:
                            for layer_kv,layer_kv_selected in zip(self._past_key_values[start_gpu], self._past_key_values[selected_token_gpu]):
                                for e, e_selected in zip(layer_kv, layer_kv_selected):
                                    e[start_idx:, :, :, :] = e_selected[selected_token_idx, :, :, :].unsqueeze(0).repeat(
                                        rows-start_idx, 1, 1, 1)
                        start_idx = 0
                        start_gpu += 1

                    self._attention_mask[start_gpu][start_idx:end_idx+1, :] = self._attention_mask[selected_token_gpu][selected_token_idx, :].repeat(
                        end_idx-start_idx+1, 1)
                    if start_gpu==selected_token_gpu:
                        for layer_kv in self._past_key_values[start_gpu]:
                            for e in layer_kv:
                                e[start_idx:end_idx+1, :, :, :] = e[selected_token_idx, :, :, :].unsqueeze(0).repeat(
                                    end_idx-start_idx+1, 1, 1, 1)
                    else:   # if selected token is not on the same machine with current token
                        for layer_kv,layer_kv_selected in zip(self._past_key_values[start_gpu], self._past_key_values[selected_token_gpu]):
                            for e, e_selected in zip(layer_kv, layer_kv_selected):
                                e[start_idx:end_idx+1, :, :, :] = e_selected[selected_token_idx, :, :, :].unsqueeze(0).repeat(
                                    end_idx-start_idx+1, 1, 1, 1)
                    
            # get candidate sequences reward
            batch_size = scores.shape[0]
            ids = topk_ids.reshape((batch_size*self._topk, 1))
            candidate_tokens = self._lm_tokenizer.batch_decode(ids, skip_special_tokens=True)

            max_candidate_length = self._rm_tokenizer.batch_encode_plus(
                candidate_tokens,
                return_tensors="pt",
                padding=True,
            ).input_ids.shape[1]

            q = Queue()
            for i in range(self._num_gpus):
                q.put(i)
            for i in range(self._num_gpus):
                worker = Thread(target=do_normal_task, args=(q,))
                worker.start()
            q.join()

            reward_scores = torch.cat(reward_scores, dim=0).reshape((-1, self._topk))

            if self._data_container is not None:
                if self._step==0:  # update cur_row on first step since last step is hard to track
                    self._data_container['cur_row'] += batch_size
                cur_row = self._data_container['cur_row']
                self._data_container['rewards'][cur_row-batch_size:cur_row, self._step, :] = reward_scores.cpu().numpy()        # (rows, topk)
                self._data_container['logits'][cur_row-batch_size:cur_row, self._step, :] = topk_scores.cpu().numpy()           # (rows, topk)
                if self._step!=0:
                    self._data_container['selected_indices'][cur_row-batch_size:cur_row, self._step-1] = np.array(last_selected_topk_indices)     # (rows, )

            for score, id, ts in zip(scores, topk_ids, reward_scores):
                score[id] = self.apply_function(score[id], ts)
                inverse_id = torch.tensor(np.setdiff1d(range(len(score.cpu().numpy())), id.cpu().numpy()), device='cuda')
                score[inverse_id] = -float("Inf")  # set all other scores to -inf
                
            # update step, pkv, attn_mask, and dict
            self._step+=1
            self._previous_input_ids_to_topk_idx = [
                {ids.item():pos for pos,ids in enumerate(topk_ids_i)} for topk_ids_i in topk_ids
            ]
            return scores
    
    def get_reward(self, candidate_texts, past_key_values, past_attention_mask, gpu, max_candidate_length):
        with torch.inference_mode():
            inputs = self._rm_tokenizer.batch_encode_plus(
                candidate_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_candidate_length,
            ).to(f'cuda:{gpu}')

            # attention_mask with <pad> in between. e.g. [13,1052,38,50256,50256,11,50256] => [1,1,1,0,0,1,0]
            attention_mask = torch.cat((past_attention_mask, inputs.attention_mask), dim=-1)        # (batch x topk, new_seq_length)
            position_ids = torch.cumsum(attention_mask, dim=-1)[:, past_attention_mask.shape[-1]:]  # cumsum the attention to get correct pos id for each new token
            reward_scores, past_key_values = self.helper(inputs.input_ids, attention_mask, position_ids, past_key_values, gpu)
            return reward_scores, past_key_values, attention_mask
    
    # helper method that calls reward model and returns reward scores
    def helper(self, input_ids, attention_mask, position_ids, past_key_values, gpu):
        reward_model = self._reward_models[gpu]
        _, reward_logits, past_key_values = reward_model(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         position_ids=position_ids,
                                                         labels=None,
                                                         use_cache=True,
                                                         past_key_values=past_key_values)
        # # save for future exploration
        # if reward_logits.shape[1] > 1:   # classification case
        #     # calculate expected value of reward, no matter 2 classes or 5 classes, normalized by ÷ (num_class-1)
        #     device = torch.device(f'cuda:{gpu}')
        #     probs = torch.softmax(reward_logits, dim=-1)                                                # e.g. [0.1, 0.1, 0.1, 0.1, 0.6]
        #     class_vals = torch.arange(reward_logits.shape[1], dtype=torch.float).unsqueeze(0).permute(1,0).to(device)      # e.g. [0,1,2,3,4]
        #     reward_scores = torch.matmul(probs, class_vals)/(reward_logits.shape[1]-1)                  # sum and divide by 4: 3.0/4.0=0.75
        #     return reward_scores, past_key_values
        # else:
        #     return reward_logits[:, 0], past_key_values
        return reward_logits[:, 0], past_key_values


    def get_past_key_values(self, contexts, gpu, max_prompt_length):
        with torch.inference_mode():
            reward_model = self._reward_models[gpu]
            input_ids = self._rm_tokenizer.batch_encode_plus(
                contexts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_prompt_length,
            ).to(f'cuda:{gpu}')
            _, _, past_key_values = reward_model(**input_ids, labels=None, use_cache=True)
            return past_key_values, input_ids.attention_mask

    def apply_function(self, original_score, reward_score):
        reward_score = torch.clamp(reward_score, min=0, max=1)
        if self._inverse:
            reward_score = 1-reward_score
        if self._method == "linear":
            return original_score + (reward_score*self._beta).to(original_score.dtype)
        else:
            raise ValueError(f"method {self._method} not supported")
    


class RewardAugmentedLogitsProcessorNoPkv(LogitsProcessor):
    
    def __init__(self, lm_tokenizer, rm_tokenizer, reward_model, topk=20, 
                 method="linear", beta=30, inverse=False):
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = rm_tokenizer
        self._reward_model = reward_model.to('cuda')
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _, topk_ids = torch.topk(scores, self._topk, dim=-1)                                    # (batch, topk,)
        input_ids_enflated = input_ids.unsqueeze(1).expand((-1, self._topk, -1))                # (batch, topk, seq_len)
        candidate_input_ids = torch.cat((input_ids_enflated, topk_ids.unsqueeze(-1)), dim=-1)   # (batch, topk, seq_len+1)
        candidate_input_ids_unroll = candidate_input_ids.reshape((
            candidate_input_ids.shape[0]*candidate_input_ids.shape[1], -1))         # (batch*topk, seq_len+1)
        candidate_input_texts = self._lm_tokenizer.batch_decode(candidate_input_ids_unroll, skip_special_tokens=True)
        
        # return reward scores
        reward_scores = self.get_reward(candidate_input_texts).reshape((input_ids.shape[0], -1))

        # apply function (topk_scores, logits)
        for score, id, rs in zip(scores, topk_ids, reward_scores):
            
            score[id] = self.apply_function(score[id], rs)
            inverse_id = torch.tensor(np.setdiff1d(range(len(score.cpu().numpy())), id.cpu().numpy()), device='cuda')
            score[inverse_id] = -float("Inf")  # set all other scores to -inf
        return scores
    
    def get_reward(self, candidate_texts):
        with torch.inference_mode():
            # tokenizer should be configured in RAD
            input_ids = self._rm_tokenizer.batch_encode_plus(
                candidate_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._rm_tokenizer.max_length,
            ).to('cuda')
            
            _, reward, _ = self._reward_model(**input_ids, labels=None)
            return reward
    
    def apply_function(self, original_score, reward_score):
        reward_score = torch.clamp(reward_score, min=0, max=1)
        if self._inverse:
            reward_score = 1-reward_score
        if self._method == "linear":
            return original_score + (reward_score*self._beta).to(original_score.dtype)
        else:
            raise ValueError(f"method {self._method} not supported")