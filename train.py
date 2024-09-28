import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import pdb
import time
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast as autocast, GradScaler
import pandas as pd
from scipy import stats

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(args, model, train_features, dev_features, test_features,epoch_start=0):
    def Set_optimizer():
        new_layer = ["Bilinear","Tail_extractor","Head_extractor","Relation_layer",'Decoder',"Classifer","MASK_token"]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder']

        grouped_params = [
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in new_layer)],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in new_layer)],
                'weight_decay': 0.0,
                'lr': args.learning_rate
            }
        ]

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == 'AdamW':
            optimizer = AdamW(grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            raise Exception("Invalid optimizer.")

        return optimizer

    def Struct_optimizer():
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n, p in model.Dis_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in model.Dis_model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            },
        ]

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(grouped_params, lr=args.decoder_lr, eps=args.adam_epsilon)
        elif args.optimizer == 'AdamW':
            optimizer = AdamW(grouped_params, lr=args.decoder_lr, eps=args.adam_epsilon)
        else:
            raise Exception("Invalid optimizer.")

        return optimizer

    def finetune(features, optimizer, optimizer_D, num_epoch, num_steps,epoch_start=0):
        best_score = 63.0
        features_long, features_short = [], []
        for feature in features:
            if (len(feature["input_ids"]) > 350):
                features_long.append(feature)
            else:
                features_short.append(feature)
        train_dataloader = DataLoader(features, batch_size=6, shuffle=True,collate_fn=collate_fn, drop_last=True, num_workers=5)
        train_iterator = range(epoch_start, int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        model.Total = total_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)

        total_steps = total_steps // 6
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler_D = get_linear_schedule_with_warmup(optimizer_D, num_warmup_steps=warmup_steps,num_training_steps=total_steps)

        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        scaler = GradScaler()
        for epoch in tqdm(train_iterator, desc="Epoch"):
            tr_loss, tr_loss1 = 0, [0, 0, 0, 0, 0, 0, 0, 0, 0]
            tr_step_num = 0
            model.zero_grad()
            train_t = 0
            torch.cuda.empty_cache()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train")):
                model.Step += 1
                model.train()
                sample_index = torch.LongTensor(list(range(len(batch[2]))))
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          "entity_type": batch[5],
                          "sample_index": sample_index,
                          "Sentence_index": batch[6],
                          "position_ids": batch[-2].to(args.device),
                          "token_type_ids": batch[-1].to(args.device),
                          }
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss1 = outputs[1]
                    loss2 = outputs[2]

                if args.n_gpu > 1:
                    loss = loss.mean()
                    loss1 = loss1.mean()
                    loss2 = [l.mean() for l in loss2]

                loss = loss / args.gradient_accumulation_steps
                loss1 = loss1 / args.gradient_accumulation_steps

                # ==========updata model==============
                scaler.scale(loss).backward(retain_graph=True)
                scaler.unscale_(optimizer)
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                    num_steps += 1

                # ==========update discriminator==============
                if (step % 6 == 0):
                    optimizer_D.zero_grad()
                    scaler.scale(loss1).backward()
                    scaler.unscale_(optimizer_D)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler_D.step()
                    scaler.step(optimizer_D)
                    scaler.update()
                    model.zero_grad()

                tr_loss += loss.item()
                for i in range(len(loss2)):
                    tr_loss1[i] += loss2[i].item()

                tr_step_num += 1
                del loss, loss1, loss2
                del outputs

            if (epoch > num_epoch - 2):
                X = input()

            if ((epoch + 1) % 1 == 0):
                dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                print("\n*********** Eval results : %d ***********" % (epoch + 1))
                dev_output["train_losss"] = tr_loss / tr_step_num
                dev_output["train_losss1"] = [float('{:.6f}'.format(l / tr_step_num)) for l in tr_loss1]
                for key in sorted(dev_output.keys()):
                    print("\t", key, "=", str(dev_output[key]))

                if dev_score > best_score or epoch > num_epoch - 2:
                    best_score = dev_score

                    if args.save_path != "":
                        save_path = args.save_path + str(epoch + 1) + "_" + str(round(best_score, 2)) + ".bin"
                        print("\n\tSave Model to ", save_path)
                        torch.save(model.state_dict(), save_path)

        return num_steps

    optimizer = Set_optimizer()
    optimizer_D = Struct_optimizer()

    if args.n_gpu > 1:
        model = DataParallel(model)

    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, optimizer_D, args.num_train_epochs, num_steps,epoch_start)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in tqdm(dataloader, desc="dev"):
        model.eval()
        sample_index = torch.LongTensor(list(range(len(batch[2]))))
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  "entity_type": batch[5],
                  "sample_index": sample_index,
                  "Sentence_index": batch[6],
                  "position_ids": batch[-2].to(args.device),
                  "token_type_ids": batch[-1].to(args.device)
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.detach().cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    best_f1, best_f1_ign, p, r, pre, tol = 0, 0, 0, 0, 0, 0
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, p, r, pre, tol = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_P:": p * 100,
        tag + "_R:": r * 100,
    }
    return best_f1*100, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        sample_index = torch.LongTensor(list(range(len(batch[2]))))
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  "entity_type": batch[5],
                  "sample_index": sample_index,
                  "Sentence_index": batch[6],
                  "position_ids": batch[-2].to(args.device),
                  "token_type_ids": batch[-1].to(args.device)
                  }
        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument('--decoder_lr', type=float, default=4e-5)
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_decoder_layers', type=int, default=2)


    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

    n_gpu = 0
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
    print("device:", device, "\t", "n_gpu:", n_gpu)

    args.n_gpu = n_gpu
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    load_train_file = os.path.join(args.data_dir, "load_"+args.train_file)
    load_dev_file = os.path.join(args.data_dir, "load_"+args.dev_file)
    load_test_file = os.path.join(args.data_dir, "load_"+args.test_file)

    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length,save_file=load_train_file)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length,save_file=load_dev_file)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length,save_file=load_test_file)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(args,config,num_labels=args.num_labels)
    model.encoder.expand_position()
    model = model.to(args.device)

    load_path=args.load_path
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    file_list = sorted(os.listdir(load_path), reverse=True)

    if len(file_list) == 0 or True:
        train(args, model, train_features, dev_features, test_features)
    else:
        load_file=load_path+"XXXXX"
        print("=============load model from %s================"%(load_file))
        epoch = int(file_list[0].split("_")[0])
        model.load_state_dict(torch.load(load_file))
        train(args, model, train_features, dev_features, test_features,epoch)



if __name__ == "__main__":
    main()
