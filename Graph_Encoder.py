import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention,BertSelfOutput
from transformers import BertPreTrainedModel
import pdb
import math
import copy
from transformers.models.bert.modeling_bert import ACT2FN
from losses import ATLoss,Focal_loss

class Graph_Encoder(BertPreTrainedModel):
    def __init__(self, config, num_layers,edge_num=4,return_intermediate=True):
        super().__init__(config)
        self.config = copy.deepcopy(config)
        self.config.num_attention_heads = edge_num
        self.num_layers=num_layers

        self.entity_embeddings =  nn.ModuleList([nn.Embedding(50, config.hidden_size) for i in range(num_layers)])
        self.sentence_embeddings = nn.ModuleList([nn.Embedding(30, config.hidden_size) for i in range(num_layers)])

        self.layers = nn.ModuleList([Struct_SelfAttention(self.config) for i in range(num_layers)])

        self.MLP = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps),
            )

        # self.init_weights()


    def forward(
            self,
            hidden_states,
            mention_mask,
            structure_mask,
            Sentence_em=None,
            entity_id_em=None,
            cross_struct_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            Temp=None,
         ):
        Loss = 0.0
        for i, layer_module in enumerate(self.layers):
            hidden_states,loss = layer_module(hidden_states=hidden_states,
                                         attention_mask=mention_mask,
                                         structure_mask=structure_mask,
                                         Sentence_em = Sentence_em,
                                         entity_id_em = entity_id_em,
                                         layer_idx=i,
                                         cross_struct_mask=cross_struct_mask,  # 结构mask
                                         encoder_hidden_states=encoder_hidden_states,
                                         encoder_attention_mask=encoder_attention_mask
                                        )
            Loss += loss

        hidden_states1 = self.MLP(hidden_states)

        return hidden_states, hidden_states1, Loss / 20.0

class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.intermediate_act_fn = ACT2FN["gelu"]

        self.dense3 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.LayerNorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.LayerNorm1(hidden_states + input_tensor)

        hidden_states1 = self.dense2(hidden_states)
        hidden_states2 = self.intermediate_act_fn(hidden_states1)

        hidden_states3 = self.dense3(hidden_states2)
        hidden_states3 = self.dropout3(hidden_states3)
        hidden_states = self.LayerNorm3(hidden_states3 + hidden_states)
        return hidden_states

class Struct_SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.num_attention_heads = self.config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, int(3*self.attention_head_size))
        self.key = nn.Linear(config.hidden_size, int(3*self.attention_head_size))
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.query1 = nn.Sequential(
            nn.Linear(config.hidden_size, int(1*self.attention_head_size)),
            # nn.LeakyReLU(0.2),
            # nn.ReLU(inplace=True),
        )
        self.key1 = nn.Sequential(
            nn.Linear(config.hidden_size, int(1*self.attention_head_size)),
            # nn.LeakyReLU(0.2),
            # nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.Focal_loss = Focal_loss()

        self.GNN_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        self.Coss_attention = Cross_Attention(self.config)

        self.Output = Output(self.config)

    def transpose_for_scores(self, x):
        num_attention_heads = int(x.size(-1) /self.attention_head_size)
        new_x_shape = x.size()[:-1] + (num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if(len(list(x.size()))==4):
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(0, 3, 1, 2,4)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_mask=None,
        layer_idx=None,
        Sentence_em=None,
        entity_id_em=None,
        cross_struct_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # ================local context encoder===================
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        b, h, n, d = key_layer.size()
        attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_mask0 = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0  # mask pad
        attention_scores3 = attention_scores1 + attention_mask0

        Index = attention_mask.unsqueeze(1).unsqueeze(2).expand(b, 3, n, n)
        Index = Index * Index.transpose(-1, -2)
        self_del = torch.eye(n).to(structure_mask).unsqueeze(0).unsqueeze(1)
        Index = ((Index * (1.0 - self_del)) > 0.5)
        loss = self.Focal_loss(attention_scores3[Index], structure_mask[Index])

        attention_scores1 = attention_scores1 / math.sqrt(key_layer.size(-1))
        attention_mask0 = (1.0 - structure_mask) * -10000.0
        attention_scores1 = attention_scores1 + attention_mask0

        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # ================global context encoder===================
        hidden_states1 = self.LayerNorm(hidden_states + Sentence_em[layer_idx] + entity_id_em[layer_idx])
        query_layer = self.transpose_for_scores(self.query1(hidden_states1))
        key_layer = self.transpose_for_scores(self.key1(hidden_states1))
        attention_scores2 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(key_layer.size(-1))
        # mask pad
        attention_mask0 = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        attention_scores2 = attention_scores2 + attention_mask0

        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout(attention_probs2)

        attention_probs = torch.cat((attention_probs2,attention_probs1),dim=1)
        attention_probs = self.dropout(attention_probs)
        context_layer =  torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        GNN_out = self.GNN_output(context_layer)

        #Cross-attention
        Cross_out = self.Coss_attention(hidden_states=GNN_out,
                                        cross_struct_mask=cross_struct_mask,
                                        encoder_hidden_states=encoder_hidden_states[layer_idx],
                                        encoder_attention_mask=encoder_attention_mask
                                        )

        Out_put = self.Output(Cross_out,hidden_states)
        return Out_put,loss

class Cross_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.num_attention_heads = 4

        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size//4)
        self.key = nn.Linear(config.hidden_size, self.all_head_size//4)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.Output = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
        )
        self.Gate = nn.Sequential(
                nn.Linear(config.hidden_size*2, config.hidden_size),
                nn.Sigmoid(),
        )
        self.Gate[0].bias.data.fill_(-1.)

        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        attention_head_size = int(x.size(-1) / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        cross_struct_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(key_layer.size(-1))

        attention_mask = (1.0 - cross_struct_mask) * -10000.0
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.Output(context_layer)

        gate = self.Gate(torch.cat((hidden_states,context_layer),dim=-1))
        Out_put = self.LayerNorm(gate * context_layer + (1.0 - gate) * hidden_states)

        return Out_put





