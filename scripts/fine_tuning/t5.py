import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration

class MyT5(T5ForConditionalGeneration):
    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs):

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_past_key_value_states=decoder_past_key_value_states,
            use_cache=use_cache,
            labels=None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(outputs[0].view(-1, self.config.vocab_size),
                              labels.view(-1))
            decoder_attention_mask_float =decoder_attention_mask.float() 
            loss = torch.sum(losses * decoder_attention_mask_float.view(-1)) / torch.sum(decoder_attention_mask_float)
            return (loss,) + outputs
        return outputs

