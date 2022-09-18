import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import json

from datetime import datetime
import random
import shutil
import time
from graphks_data import collate_fn, GraphKSDataset, GraphKSBatcher, get_batch_loader
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from graphks_model import GraphKSModel
import torch.backends.cudnn as cudnn
from utils import *


def main(args):

    print("#########\nexp_name: {}\n{}\n###########".format(args.exp_name, 
        '\n'.join(f'{k}={v}' for k, v in vars(args).items())))

    if args.dataset not in ["wow"]:
        raise Exception("dataset[%s] not supported"%args.dataset)

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    logger = get_log(args.exp_name, is_test=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda", torch.cuda.is_available())

    my_dataset = GraphKSDataset
    my_collate_fn = collate_fn


    test_seen_dataset = my_dataset(args.data_dir, "test_seen", args)
    args.edge_size = test_seen_dataset.get_edge_vocab_size()
    test_unseen_dataset = my_dataset(args.data_dir, "test_unseen", args)
    train_dataset = my_dataset(args.data_dir, "train", args)
    valid_seen_dataset = my_dataset(args.data_dir, "valid_seen", args)
    valid_unseen_dataset = my_dataset(args.data_dir, "valid_unseen", args)

    test_seen_loader = get_batch_loader(test_seen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size,
                                    num_workers=args.workers, is_test=True)
    test_unseen_loader = get_batch_loader(test_unseen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size, 
                                    num_workers=args.workers, is_test=True)
    train_loader = get_batch_loader(train_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size,
                                    num_workers=args.workers, is_test=True)
    valid_seen_loader = get_batch_loader(valid_seen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size,
                                    num_workers=args.workers, is_test=True)
    valid_unseen_loader = get_batch_loader(valid_unseen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size, 
                                    num_workers=args.workers, is_test=True)

    # Batcher
    dis_batcher = GraphKSBatcher(args.bert_truncate, args.bert_config)

    model = GraphKSModel(args)
    model.transformer.resize_token_embeddings(len(dis_batcher.tokenizer))



    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.to(device)

    if args.resume:
        model_path = args.resume
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {} val_acc_seen {} val_acc_unseen {})"
                  .format(model_path, checkpoint['epoch'], checkpoint['acc_val_seen'], checkpoint['acc_val_unseen']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    cudnn.benchmark = True

    acc, seen_logit, topics_logits = valid(test_seen_loader, model, dis_batcher, "test_seen", logger, args)
    folder = args.exp_name
    out_file = os.path.join(folder, "test_seen_fin_select.jsonl")
    my_write(out_file, seen_logit)
    out_file = os.path.join(folder, "test_seen_logits.jsonl")
    my_write(out_file, seen_logit, mode=0)
    out_file = os.path.join(folder, "test_seen_topics_logits.jsonl")
    my_write(out_file, topics_logits, mode=0)

    acc_unseen, unseen_logit, topics_logits = valid(test_unseen_loader, model, dis_batcher, "test_unseen", logger, args)
    folder = args.exp_name
    out_file = os.path.join(folder, "test_unseen_fin_select.jsonl")
    my_write(out_file, unseen_logit)
    out_file = os.path.join(folder, "test_unseen_logits.jsonl")
    my_write(out_file, unseen_logit, mode=0)
    out_file = os.path.join(folder, "test_unseen_topics_logits.jsonl")
    my_write(out_file, topics_logits, mode=0)

    acc, seen_logit, topics_logits = valid(valid_seen_loader, model, dis_batcher, "valid_seen", logger, args)
    folder = args.exp_name
    out_file = os.path.join(folder, "valid_seen_fin_select.jsonl")
    my_write(out_file, seen_logit)
    out_file = os.path.join(folder, "valid_seen_logits.jsonl")
    my_write(out_file, seen_logit, mode=0)
    out_file = os.path.join(folder, "valid_seen_topics_logits.jsonl")
    my_write(out_file, topics_logits, mode=0)

    acc_unseen, unseen_logit, topics_logits = valid(valid_unseen_loader, model, dis_batcher, "valid_unseen", logger, args)
    folder = args.exp_name
    out_file = os.path.join(folder, "valid_unseen_fin_select.jsonl")
    my_write(out_file, unseen_logit)
    out_file = os.path.join(folder, "valid_unseen_logits.jsonl")
    my_write(out_file, unseen_logit, mode=0)
    out_file = os.path.join(folder, "valid_unseen_topics_logits.jsonl")
    my_write(out_file, topics_logits, mode=0)


    acc, train_logit, topics_logits = valid(train_loader, model, dis_batcher, "train", logger, args)
    folder = args.exp_name
    out_file = os.path.join(folder, "train_fin_select.jsonl")
    my_write(out_file, train_logit)
    out_file = os.path.join(folder, "train_logits.jsonl")
    my_write(out_file, train_logit, mode=0)
    out_file = os.path.join(folder, "train_topics_logits.jsonl")
    my_write(out_file, topics_logits, mode=0)


@torch.no_grad()
def valid(valid_loader, model, dis_batcher, split, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    topT = AverageMeter('Acc@Topic', ':6.2f')
    topK = AverageMeter('Acc@Know', ':6.2f')
    list_ = [batch_time, data_time, topT, topK]
    progress = ProgressMeter(
        len(valid_loader),
        list_,
        prefix="Test %s: " % split)
    model.eval()
    res_knowl = []
    res_topic = []

    end = time.time()
    for i, batch_data in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        knowledges, histories, topics, topics_knowledges = batch_data[0], batch_data[1], batch_data[4], batch_data[5]
        dis_args = dis_batcher(*batch_data)
        inputs_args = list(dis_args)
        topic_tar, knowl_tar = inputs_args[-2], inputs_args[-1]
        inputs_args = inputs_args[:-2]

        knowl_node_indicate = inputs_args[7]
        knowl_to_topic = inputs_args[8]
        
        inputs_args += [topic_tar, knowl_tar]
        # for return cross
        topics_logits, knowl_logits, _, _ = model(*inputs_args)


        # acc
        tacc = accuracy(topics_logits, topic_tar[:, 0])
        kacc = accuracy(knowl_logits, knowl_tar[:, 0])
        
        topT.update(tacc[0].item(), len(histories))
        topK.update(kacc[0].item(), len(histories))


        # save
        logits = knowl_logits
        logits = F.softmax(logits, dim=-1)
        logits_np = logits.cpu().detach().numpy() #[b, n_k]


        for bi in range(logits_np.shape[0]):
            fin_logits = logits_np[bi].tolist()
            fin_logits = list(map(float, fin_logits))
            tar = knowl_tar[bi, 0].item()
            tar, fin_logits = recover_knowledge_order(knowledges[bi], topics[bi], topics_knowledges[bi], tar, fin_logits)
            res_knowl.append({"target": tar, "prob": fin_logits})

        logits = F.softmax(topics_logits, dim=-1)
        logits_np = logits.cpu().detach().numpy() #[b, n_k]
        for bi in range(logits_np.shape[0]):
            fin_logits = logits_np[bi].tolist()
            fin_logits = list(map(float, fin_logits))
            tar = topic_tar[bi, 0].item()
            res_topic.append({"target": tar, "prob": fin_logits})


        batch_time.update(time.time() - end)
        end = time.time()
        if i % (args.print_freq)== 0:
            progress.display(i, logger)


    print("Acc@Topic: %.3f" % (topT.avg))
    print("Acc@Know: %.3f" % (topK.avg))

    return topK.avg, res_knowl, res_topic

def my_write(path, data, mode=1):
    if mode == 1:
        # just write top1 index
        _data = []
        for i, item in enumerate(data):
            logits = item["prob"]
            lnp = np.array(logits)
            ind = int(np.argmax(lnp))
            _data.append({"target": item["target"], "pred": ind})
    else:
        # write all
        _data = data
    with open(path, "w", encoding='utf-8') as wfp:
        for written_object in _data:
            wfp.write("%s\n"%(json.dumps(written_object)))

def recover_knowledge_order(origin_knowledges, topics, topics_knowledges, target, logits):
    new_order_knowledges = []
    for topic in topics:
        new_order_knowledges += topics_knowledges[topic]
    origin_target = origin_knowledges.index(new_order_knowledges[target])
    origin_logits = [logits[new_order_knowledges.index(k)] for k in origin_knowledges]
    return origin_target, origin_logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for GraphKS')

    parser.add_argument('--dataset', type=str, default='wow')

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_scale', type=float, default=100.0)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # files
    parser.add_argument('--data_dir', type=str, default='wizard_of_wikipedia/data/')

    # training scheme
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    # save
    parser.add_argument('--exp_name', type=str, default='output_models/test')
    parser.add_argument('--log', type=str, default='wizard_of_wikipedia/log')
    # model
    parser.add_argument('--bert_config', type=str, default='pretrain-models/bert_base_uncased')
    parser.add_argument('--bert_truncate', type=int, default=128)  # for bert
    parser.add_argument('--edge_hidden_size', type=int, default=64)
    parser.add_argument('--gat_alpha', type=float, default=0.2)
    parser.add_argument('--gat_hid_dim', type=int, default=2048)
    parser.add_argument('--gat_header', type=int, default=8)
    parser.add_argument('--gat_layer', type=int, default=1)
    parser.add_argument('--encoder_out_dim', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)
