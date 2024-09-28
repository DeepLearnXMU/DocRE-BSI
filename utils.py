import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    if max_len%2!=0:
        max_len+=1

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]

    entity_pos = [f["entity_pos"] for f in batch]
    Sentence_index=[f["Sentence_index"] for f in batch]
    hts = [f["hts"] for f in batch]
    entity_type=[f["entity_type"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    position = [i for i in range(max_len)]
    position_ids = torch.tensor([position for _ in batch], dtype=torch.long)

    type_id = [0 for i in range(max_len)]
    token_type_ids = torch.tensor([type_id for _ in batch], dtype=torch.long)

    output = (input_ids, input_mask, labels, entity_pos, hts,entity_type,Sentence_index,position_ids,token_type_ids)
    return output