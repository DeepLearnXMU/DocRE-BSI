import torch
import torch.nn.functional as F
import numpy as np
import pdb

def process_long_input(model, input_ids, attention_mask):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True
    )
    sequence_output = output['last_hidden_state']
    hidden_states = output["hidden_states"]
    return sequence_output, output["attentions"], hidden_states
