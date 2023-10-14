from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2ForSequenceClassification
import torch
from torch import nn
from typing import Optional, Tuple


class GPT2RewardModel(nn.Module):
    def __init__(self, reward_model_name="gpt2", out_features=1, loss_fn="cumulative_mse"):
        super(GPT2RewardModel, self).__init__()
        model = GPT2LMHeadModel.from_pretrained(reward_model_name)
        # model = GPT2ForSequenceClassification.from_pretrained(reward_model_name)
        model.lm_head = nn.Linear(in_features=model.lm_head.in_features, out_features=out_features, bias=True)
        # model.score = nn.Linear(in_features=model.score.in_features, out_features=out_features, bias=True)
        model.config.use_cache = True
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features
        self.loss_fn = get_loss_fn(loss_fn)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs['logits']
        # find the last valid token's ids
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(logits.device)
        # use the last valid token's representation: (batch, max_length, out_features) => (batch, out_features)
        scores = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(scores, labels, logits, sequence_lengths+1)

        if use_cache:
            past_key_values = outputs['past_key_values']
            return loss, scores, past_key_values
        else:
            return loss, scores


def get_loss_fn(name):
    if name == "mse":
        def mse_loss_fn(scores, labels, logits, lengths):
            return nn.MSELoss()(scores, labels)
        
        loss_fn = mse_loss_fn

    elif name == "cross_entropy":
        def ce_loss_fn(scores, labels, logits, lengths):
            return nn.CrossEntropyLoss()(scores, labels)    # here score is logits[last_token_id]
        
        loss_fn = ce_loss_fn

    elif name == "cumulative_mse":
        def cumulative_mse_fn(scores, labels, logits, lengths):
            mse_loss = nn.MSELoss(reduction='none')
            losses = []
            for i in range(len(labels)):
                logit = logits[i, :lengths[i]].reshape(lengths[i], -1)                  # (lengths[i], out)
                label = labels[i].reshape(-1).repeat(lengths[i], 1).float()             # (lengths[i], out)
                loss = mse_loss(logit, label)                                            # (lengths[i], out)
                loss = torch.matmul(
                    loss.permute(1,0).float(),
                    torch.arange(start=1, end=lengths[i]+1, device=logits.device).float()
                )   # (out,)
                losses.append(2*torch.sum(loss)/(lengths[i]+1)/lengths[i])              # s = n(n+1)/2
            return torch.stack(losses).mean()
        
        loss_fn = cumulative_mse_fn
        
    elif name == "cumulative_ce":
        def cumulative_ce_fn(scores, labels, logits, lengths):
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            losses = []
            for i in range(len(labels)):
                logit = logits[i, :lengths[i]]  # (lengths[i], out_features)
                label = labels[i].repeat(lengths[i])  # (lengths[i],)
                # multiply ce_loss with a linearly increasing weight e.g. [1/5, 2/5, 3/5, 4/5, 5/5]
                loss = ce_loss(logit, label)*torch.arange(start=1, end=lengths[i]+1, device=logits.device)/lengths[i]
                # sum and multiply by 2/(1+n) to keep the expected loss the same as other methods
                losses.append(2*torch.sum(loss)/(lengths[i]+1))
            return torch.stack(losses).mean()
        
        loss_fn = cumulative_ce_fn
        
    else:
        raise ValueError(f"loss function name {name} not available")
    
    return loss_fn