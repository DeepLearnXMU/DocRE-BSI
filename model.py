import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import numpy as np
from Graph_Encoder import Graph_Encoder
from transformers import AutoConfig, AutoModel, AutoTokenizer
import math
import time
import copy
import pdb
from torch.nn import functional as F
from BERT import BertModel,BertPreTrainedModel
from Roberta import RobertaModel

def output_model_put(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def contrast_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss1 = (2 - 2 * (x * y.detach()).sum(dim=-1)).mean()
    loss2 = (2 - 2 * (x.detach() * y).sum(dim=-1)).mean()
    return 0.8 * loss1 + 0.2 * loss2


def contrast_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss1 = (2 - 2 * (x * y.detach()).sum(dim=-1)).mean()
    loss2 = (2 - 2 * (x.detach() * y).sum(dim=-1)).mean()
    return 0.8 * loss1 + 0.2 * loss2

def compute_kl_loss(p, q,T=2.0):
    p = p/T
    q = q/T
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q.detach(), dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p.detach(), dim=-1), reduction='none')
    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)
    loss = 0.8 * p_loss + 0.2 * q_loss
    return loss.mean()

def KL_loss(p, q, T=2.0):
    p = p/T
    q = q/T
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)
    loss = (p_loss + q_loss) / 2.0
    return loss.mean()

def Distillation_loss(p, q, labels, T_p=3.0, T_q=2.0, Weight=0.8):
    label = labels.clone()
    label[:, 0] = 1.0
    Temp = (label * T_p) + ((1 - label) * T_q)

    p = p/Temp
    q = q/Temp
    loss1 = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q.detach(), dim=-1))
    loss2 = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p.detach(), dim=-1))
    loss_kd = Weight * loss1 + (1.0-Weight) * loss2
    return loss_kd


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self,hidden_states):
        hidden_states1 = self.dense1(hidden_states)
        hidden_states2 = self.LayerNorm(hidden_states+hidden_states1)

        return hidden_states2

class Classifer(nn.Module):
    def __init__(self, args, config, emb_size=768):
        super().__init__()
        emb_size = config.hidden_size

        self.Head_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, emb_size),
            nn.Tanh(),
        )
        self.Tail_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, emb_size),
            nn.Tanh(),
        )

        self.MLP = MLP(config)

        self.Bilinear = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(emb_size, eps=config.layer_norm_eps),
            nn.Linear(emb_size, config.num_labels, bias=False),
        )

        self.hidden_size = config.hidden_size

    def forward(self,hs,ts,rs):
        hs_pair = self.Head_extractor(hs)
        ts_pair = self.Tail_extractor(ts)
        bl = torch.cat((hs_pair,ts_pair),dim=-1)
        logits = self.Bilinear(bl)

        return logits


class Discriminator(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size = 128
        self.Dis_struct_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, 3),
        )
        self.Dis_struct_out = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, 3),
        )

    def forward(self, Node1, Node2, Node3):
        Logits_n1 = self.Dis_struct_in(Node1)
        Logits_n2 = self.Dis_struct_in(Node2)
        Logits_n3 = self.Dis_struct_in(Node3)
        Logits_n = [Logits_n1, Logits_n2, Logits_n3]

        return Logits_n



class DocREModel(nn.Module):
    def __init__(self, args,config,emb_size=768, block_size=64, num_labels=-1,used_cross_attnetion=False):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.emb_size = self.config.hidden_size
        self.num_labels=num_labels
        config.output_hidden_states = True
        self.encoder = BertModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        self.Decoder = Graph_Encoder(config=self.config, num_layers=args.num_decoder_layers)
        self.loss_fnt = ATLoss()
        self.hidden_size = config.hidden_size
        emb_size = config.hidden_size

        self.BCE = nn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.Head_extractor = nn.Sequential(
            nn.Linear(config.hidden_size * 3, emb_size),
            nn.Tanh(),
        )
        self.Tail_extractor = nn.Sequential(
            nn.Linear(config.hidden_size * 3, emb_size),
            nn.Tanh(),
        )
        self.Bilinear = nn.Sequential(
            nn.Linear(emb_size * block_size, emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(emb_size, eps=config.layer_norm_eps),
            nn.Linear(emb_size, self.config.num_labels, bias=False),
        )
        self.Classifer = nn.ModuleList([Classifer(args, config) for i in range(2)])
        self.Dis_model = Discriminator(config)

        self.MASK_token = nn.parameter.Parameter(nn.init.xavier_uniform_(
            torch.empty(1, self.config.hidden_size)).float())

        self.emb_size = emb_size
        self.block_size = block_size

        self.Step = 0
        self.Total = 1
        # self.init_weights()

    def init_weights(self):
        for m in [self.Head_extractor,self.Tail_extractor,self.Bilinear,self.Decoder]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def init_para(self):
        for m in [self.head_extractor, self.tail_extractor, self.bilinear,self.Combine, self.Relation_classifier]:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)


    def Encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]

        sequence_output, attention, hidden_states = process_long_input(self.encoder, input_ids, attention_mask)
        return sequence_output, attention,hidden_states

    def get_mention_rep(self, sequence_output, attention, attention_mask, entity_pos, entity_type, Sentence_index):
        mention, Node_type, mention_entity_type, mention_entity_id, mention_sentence_id = [], [], [], [], []
        Entity_attention = []
        entity2mention = []
        Sentence_index_matrix = []
        batch_size, doc_len, _ = sequence_output.size()
        Sentence_em, Sentence_em1 = [], []

        Max_met_num = -1
        for i in range(batch_size):
            mention.append([sequence_output[i][0]])
            Sentence_em.append([self.Decoder.sentence_embeddings[0].weight[-1]])
            Sentence_em1.append([self.Decoder.sentence_embeddings[1].weight[-1]])

            Node_type.append([1])
            mention_entity_type.append([7])
            mention_entity_id.append([48])
            mention_sentence_id.append([29])

            mention_indx = 1
            entity2mention.append([])
            entity_atts = []
            for j, e in enumerate(entity_pos[i]):
                e_att = []
                entity2mention[-1].append([])
                for start, end, sentence_id in e:
                    mention[-1].append((sequence_output[i][start + 1] + sequence_output[i][end]) / 2.0)
                    Sentence_em[-1].append(self.Decoder.sentence_embeddings[0].weight[sentence_id + 1])
                    Sentence_em1[-1].append(self.Decoder.sentence_embeddings[1].weight[sentence_id + 1])
                    e_att.append((attention[i, :, start + 1] + attention[i, :, end]) / 2.0)

                    Node_type[-1].append(2)
                    mention_entity_type[-1].append(entity_type[i][j])
                    mention_entity_id[-1].append(j + 1)
                    mention_sentence_id[-1].append(sentence_id + 1)

                    entity2mention[-1][-1].append(mention_indx)
                    mention_indx += 1

                if len(e_att) > 1:
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_att = e_att[0]
                entity_atts.append(e_att)

            entity_atts = torch.stack(entity_atts, dim=0)
            Entity_attention.append(entity_atts)

            for e in entity2mention[-1]:
                e_emb = []
                s_emb, s_emb1 = [], []
                mention_sentence_id[-1].append([])
                for j in e:
                    e_emb.append(mention[-1][j])
                    s_emb.append(Sentence_em[-1][j])
                    s_emb1.append(Sentence_em1[-1][j])
                    mention_sentence_id[-1][-1].append(mention_sentence_id[-1][j])
                if (len(e_emb) > 1):
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    s_emb = torch.stack(s_emb, dim=0).mean(0)
                    s_emb1 = torch.stack(s_emb1, dim=0).mean(0)
                else:
                    e_emb = e_emb[0]
                    s_emb = s_emb[0]
                    s_emb1 = s_emb1[0]

                mention[-1].append(e_emb)
                Sentence_em[-1].append(s_emb)
                Sentence_em1[-1].append(s_emb1)

                Node_type[-1].append(3)
                mention_entity_type[-1].append(mention_entity_type[-1][e[0]])
                mention_entity_id[-1].append(mention_entity_id[-1][e[0]])

                mention_indx += 1

            Sentence_matrix = torch.zeros((len(Sentence_index[i]) + 1, doc_len)).to(sequence_output.device)
            for j, (start_sent, end_sent) in enumerate(Sentence_index[i]):
                Sentence_matrix[j + 1][start_sent:end_sent] = 1

            Sentence_index_matrix.append(Sentence_matrix)

            if (Max_met_num < mention_indx):
                Max_met_num = mention_indx

        mention_mask = []
        for i in range(batch_size):
            origin_len = len(mention[i])
            extence = Max_met_num - origin_len
            mention_mask.append([1] * origin_len + [0] * extence)

            Node_type[i].extend([0] * extence)
            mention_entity_type[i].extend([0] * extence)
            mention_entity_id[i].extend([0] * extence)
            mention_sentence_id[i].extend([0] * extence)

            for j in range(extence):
                mention[i].append(torch.zeros(self.config.hidden_size).to(sequence_output.device))
                Sentence_em[i].append(torch.zeros(self.config.hidden_size).to(sequence_output.device))
                Sentence_em1[i].append(torch.zeros(self.config.hidden_size).to(sequence_output.device))

            mention[i] = torch.stack(mention[i], dim=0)
            Sentence_em[i] = torch.stack(Sentence_em[i], dim=0)
            Sentence_em1[i] = torch.stack(Sentence_em1[i], dim=0)

        mention = torch.stack(mention, dim=0)
        Sentence_em = torch.stack(Sentence_em, dim=0)
        Sentence_em1 = torch.stack(Sentence_em1, dim=0)

        structure_mask = []
        for b in range(len(entity_pos)):
            structure_mask_Temp = np.zeros((3, Max_met_num, Max_met_num), dtype='float')
            for i in range(Max_met_num):
                if mention_mask[b][i] < 0.5:
                    break
                else:
                    for j in range(Max_met_num):
                        if mention_mask[b][j] < 0.5:
                            break
                        if i == j:
                            structure_mask_Temp[0][i][j] = 1
                            structure_mask_Temp[1][i][j] = 1
                            structure_mask_Temp[2][i][j] = 1
                        else:
                            if (i == 0):
                                structure_mask_Temp[0][i][j] = 1
                                structure_mask_Temp[0][j][i] = 1
                            else:
                                if (Node_type[b][i] == 3):
                                    if (Node_type[b][j] == 3 and len(
                                            set(mention_sentence_id[b][i]) & set(mention_sentence_id[b][j])) > 0):
                                        structure_mask_Temp[1][i][j] = 1
                                        structure_mask_Temp[1][j][i] = 1
                                    if (Node_type[b][j] == 2 and mention_entity_id[b][i] == mention_entity_id[b][j]):
                                        structure_mask_Temp[2][i][j] = 1
                                        structure_mask_Temp[2][j][i] = 1

                                if (Node_type[b][i] == 2):
                                    if (Node_type[b][j] == 2 and mention_entity_id[b][i] == mention_entity_id[b][j]):
                                        structure_mask_Temp[2][i][j] = 1
                                        structure_mask_Temp[2][j][i] = 1
                                    if (Node_type[b][j] == 2 and mention_sentence_id[b][i] == mention_sentence_id[b][j]
                                            and mention_entity_id[b][i] != mention_entity_id[b][j]):
                                        structure_mask_Temp[1][i][j] = 1
                                        structure_mask_Temp[1][j][i] = 1


            structure_mask.append(structure_mask_Temp.tolist())

        mention_mask = torch.tensor(mention_mask, dtype=torch.float).to(sequence_output)
        Node_type = torch.tensor(Node_type, dtype=torch.float).to(sequence_output)
        structure_mask = torch.tensor(structure_mask, dtype=torch.float).to(sequence_output)
        structure_mask = structure_mask.bool().float()

        Node_type = torch.tensor(Node_type, dtype=torch.long).to(sequence_output.device)
        mention_entity_type = torch.tensor(mention_entity_type, dtype=torch.long).to(sequence_output.device)
        mention_entity_id = torch.tensor(mention_entity_id, dtype=torch.long).to(sequence_output.device)

        batch_size, Max_node_num, _ = mention.size()
        Max_doc_len = attention_mask.size()[-1]
        mask1 = attention_mask.unsqueeze(1)
        mask1 = mask1.repeat(1, Max_node_num, 1)
        mask1 = mask1.unsqueeze(1)

        mask2 = []
        for i in range(batch_size):
            mention_sentence_id[i][0] = 0
            temp = []
            for j in mention_sentence_id[i]:
                if isinstance(j, int):
                    temp.append(Sentence_index_matrix[i][j])
                else:
                    temp.append(Sentence_index_matrix[i][j].sum(0))

            temp = torch.stack(temp, dim=0)
            mask2.append(temp)
            mention_sentence_id[i][0] = 29

            mask2[-1][0] = attention_mask[i]

        mask2 = torch.stack(mask2, dim=0)
        mask2 = mask2.unsqueeze(1)
        cross_struct_mask = torch.cat([mask2, mask2, mask1, mask1], 1)

        cross_struct_mask = (cross_struct_mask > 0).to(attention_mask)

        Sentence_em = [Sentence_em, Sentence_em1]

        mention_feature = {
            "mention": mention,
            "Sentence_em": Sentence_em,
            "mention_mask": mention_mask,
            "structure_mask": structure_mask,
            "cross_struct_mask": cross_struct_mask,
            "mention_entity_type": mention_entity_type,
            "Node_type": Node_type,
            "mention_entity_id": mention_entity_id,
            "mention_sentence_id": mention_sentence_id,
            "entity2mention": entity2mention,
            "Entity_attention": Entity_attention
        }
        return mention_feature

    def get_entity_pair(
            self,
            mention_output,
            encoder_out,
            mention_feature,
            hts,
            encoder_hidden_states,
    ):
        Entity_attention = mention_feature["Entity_attention"]
        entity2mention=mention_feature["entity2mention"]
        mention_entity_id=mention_feature["mention_entity_id"]
        mention_sentence_id=mention_feature["mention_sentence_id"]
        mention_mask =mention_feature["mention_mask"]
        Node_type = mention_feature["Node_type"]
        mentions = mention_feature["mention"]

        hss, tss, rss, rss1, Entity_pairs, hts_tensor = [], [], [], [], [], []
        hss_m, tss_m, rss_m=[],[],[]
        batch_size,Max_met_num,_=mention_output.size()

        for i in range(batch_size):
            Entity_Index = (Node_type[i] == 3.0)
            entity_embs = mentions[i][Entity_Index]
            entity_embs1 = mention_output[i][Entity_Index]

            ht_i = torch.LongTensor(hts[i]).to(mention_output.device)
            hts_tensor.append(ht_i)

            Entity = []
            for k,e in enumerate(entity2mention[i]):
                e_emb = []
                for j in e:
                    e_emb.append(mention_output[i][j])
                if (len(e_emb) > 1):
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                else:
                    e_emb = e_emb[0]
                Entity.append(e_emb)
            Entity = torch.stack(Entity, dim=0)

            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hs1 = torch.index_select(entity_embs1, 0, ht_i[:, 0])
            ts1 = torch.index_select(entity_embs1, 0, ht_i[:, 1])

            hs2 = torch.index_select(Entity, 0, ht_i[:, 0])
            ts2 = torch.index_select(Entity, 0, ht_i[:, 1])

            doc_rep = mention_output[i][0][None, :].expand(hs.size()[0], self.hidden_size)

            hs2 = (hs2 + doc_rep)/2.0
            ts2 = (ts2 + doc_rep)/2.0

            h_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 0])
            t_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", encoder_out[i], ht_att)

            hs_pair = self.Head_extractor(torch.cat([hs1,hs2, rs], dim=1))
            ts_pair = self.Tail_extractor(torch.cat([ts1,ts2, rs], dim=1))
            b1 = hs_pair.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts_pair.view(-1, self.emb_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            rs = self.Bilinear(bl)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)

        return hss,tss,rss


    def middle_entity(self,sequence_output,attention,entity_pos,hts,index=0):
        batch_size, doc_len, _ = sequence_output.size()
        logits, Nodes= [], []
        Max_met_num=-1
        for i in range(batch_size):
            mention = [sequence_output[i][0]]
            mention_indx = 1
            entity2mention = []
            entity_atts = []
            for j,e in enumerate(entity_pos[i]):
                e_att=[]
                entity2mention.append([])
                for start,end,sentence_id in e:
                    mention.append((sequence_output[i][start + 1] + sequence_output[i][end]) / 2.0)
                    e_att.append((attention[i, :, start + 1] + attention[i, :, end]) / 2.0)

                    entity2mention[-1].append(mention_indx)
                    mention_indx+=1
                if len(e_att) > 1:
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_att=e_att[0]
                entity_atts.append(e_att)

            entity_atts = torch.stack(entity_atts, dim=0)

            Entity = []
            for e in entity2mention:
                e_emb = []
                for j in e:
                    e_emb.append(mention[j])

                if(len(e_emb)>1):
                    e_emb = torch.logsumexp(torch.stack(e_emb,dim=0), dim=0)
                else:
                    e_emb = e_emb[0]

                Entity.append(e_emb)
                mention.append(e_emb)

            Nodes.append(self.Classifer[index].MLP(torch.stack(mention, dim=0)))
            Entity1 = self.Classifer[index].MLP(torch.stack(Entity, dim=0))

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(Entity1, 0, ht_i[:, 0])
            ts = torch.index_select(Entity1, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            logit = self.Classifer[index](hs,ts,rs)

            logits.append(logit)

        logits = torch.cat(logits, dim=0)
        Nodes = torch.cat(Nodes, dim=0)
        return logits,Nodes


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                entity_type=None,
                sample_index=None,
                Sentence_index=None,
                position_ids=None,
                token_type_ids=None
                ):

        sequence_output, attention, hidden_states = self.Encode(input_ids, attention_mask)
        sequence_output = (hidden_states[-2] + hidden_states[-1]) / 2.0
        attention1 = (attention[-3] + attention[-4]) / 2.0
        hidden_states1 = [sequence_output,sequence_output]

        mention_feature = self.get_mention_rep(sequence_output, attention1, attention_mask, entity_pos, entity_type,Sentence_index)
        entity_id_em = self.Decoder.entity_embeddings[0](mention_feature["mention_entity_id"])
        entity_id_em1 = self.Decoder.entity_embeddings[1](mention_feature["mention_entity_id"])
        entity_id_em = [entity_id_em,entity_id_em1]


        decoder_output,decoder_output0,Loss =  self.Decoder(
            hidden_states= mention_feature["mention"],
            mention_mask=mention_feature["mention_mask"],
            structure_mask=mention_feature["structure_mask"],
            cross_struct_mask=mention_feature["cross_struct_mask"],
            Sentence_em = mention_feature["Sentence_em"],
            entity_id_em = entity_id_em,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        hss,tss,logits = self.get_entity_pair(
            decoder_output,
            sequence_output,
            mention_feature,
            hts,
            attention_mask
        )

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.clone().float(), labels.clone().float(), P_weight=2.0)

            Index = (mention_feature["mention_mask"]>0.5)
            logits1, Nodes1 = self.middle_entity(hidden_states[-1], attention[-3], entity_pos, hts, index=0)
            logits2, Nodes2 = self.middle_entity(hidden_states[-7], attention[-7], entity_pos, hts, index=1)

            Node_Input = mention_feature["mention"].clone()
            Mask_Index = (mention_feature["Node_type"] == 2).float()

            Mask_Index = Mask_Index - F.dropout(Mask_Index, p=0.2) * (1 - 0.2)
            Mask_Index = (Mask_Index > 0.5)
            Node_Input[Mask_Index] = self.MASK_token

            decoder_output1, decoder_output11, Loss1 = self.Decoder(
                hidden_states=Node_Input,
                mention_mask=mention_feature["mention_mask"],
                structure_mask=mention_feature["structure_mask"],
                cross_struct_mask=mention_feature["cross_struct_mask"],
                Sentence_em=mention_feature["Sentence_em"],
                entity_id_em=entity_id_em,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
            )

            hss, tss, logits11 = self.get_entity_pair(
                decoder_output1,
                sequence_output,
                mention_feature,
                hts,
                attention_mask
            )
            Loss += Loss1
            loss += self.loss_fnt(logits11.clone().float(), labels.clone().float(), P_weight=2.0)
            Loss2 = KL_loss(logits11, logits)

            loss21 = 0.3 * self.loss_fnt(logits1.float(), labels.clone().float()) \
                     + 0.7 * KL_loss(logits1, logits)
            loss22 = 0.3 * self.loss_fnt(logits2.float(), labels.clone().float()) \
                     + 0.7 * KL_loss(logits2, logits)
            loss2 = (loss21 + loss22) / 2.0

            Logits_n = self.Dis_model(decoder_output[Index].detach(), Nodes1, Nodes2)

            Label = torch.zeros((Logits_n[0].size(0),)).to(Logits_n[0]).long()
            loss6, loss7 = 0, 0
            for k, logit in enumerate(Logits_n):
                loss6 += self.CrossEntropyLoss(logit, Label + k)
                loss7 += self.CrossEntropyLoss(logit, Label)

            Labels1 = F.normalize(mention_feature["mention"][Mask_Index].detach(), dim=-1, p=2)
            Nodes_1 = F.normalize(decoder_output11[Mask_Index], dim=-1, p=2)
            loss8 = (1 - (Labels1 * Nodes_1).sum(dim=-1)).mean() / 100.0

            loss61 = loss6 / loss6.item()
            loss71 = loss7 / loss7.item()

            Loss_s = loss61 / 100.0
            Loss_m1 = 10 * loss + loss8 + Loss2 + loss2
            Loss_m = 10 * (Loss_m1 / Loss_m1.item()) + loss71 / 100.0
            loss_list = [loss, Loss, loss8, Loss2, loss2, loss6, loss7]

            output = (Loss_m, Loss_s, loss_list) + output

        return output
