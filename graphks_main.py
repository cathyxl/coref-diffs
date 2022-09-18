import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

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

    print("#################\nexp_name: {}\n{}\n#################".format(
        args.exp_name, '\n'.join(f'{k}={v}' for k, v in vars(args).items())))

    if args.dataset not in ["wow"]:
        raise Exception("dataset[%s] not supported"%args.dataset)

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    logger = get_log(args.exp_name)
    print("cuda available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_dataset = GraphKSDataset
    my_collate_fn = collate_fn

    train_dataset = my_dataset(args.data_dir, "train", args)
    args.edge_size = train_dataset.get_edge_vocab_size()
    train_loader = get_batch_loader(train_dataset, collate_fn=my_collate_fn, batch_size=args.train_batch_size,
                                    num_workers=args.workers, is_test=False)

    valid_seen_dataset = my_dataset(args.data_dir, "valid_seen", args)
    valid_seen_loader = get_batch_loader(valid_seen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size,
                                    num_workers=args.workers, is_test=True)
    valid_unseen_dataset = my_dataset(args.data_dir, "valid_unseen", args)
    valid_unseen_loader = get_batch_loader(valid_unseen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size, 
                                    num_workers=args.workers, is_test=True)

    test_seen_dataset = my_dataset(args.data_dir, "test_seen", args)
    test_seen_loader = get_batch_loader(test_seen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size,
                                    num_workers=args.workers, is_test=True)
    test_unseen_dataset = my_dataset(args.data_dir, "test_unseen", args)
    test_unseen_loader = get_batch_loader(test_unseen_dataset, collate_fn=my_collate_fn, batch_size=args.eval_batch_size, 
                                    num_workers=args.workers, is_test=True)
        
    dis_batcher = GraphKSBatcher(args.bert_truncate, args.bert_config)

    model = GraphKSModel(args)
    model.transformer.resize_token_embeddings(len(dis_batcher.tokenizer))

    optimizer = AdamW([{'params':model.parameters()}],
                        lr=args.lr)
    num_training_steps = math.ceil(float(len(train_dataset)) / args.train_batch_size) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

    print("optimizer config: \n{}".format(optimizer))


    # loss function
    cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if "scheduler" in checkpoint and scheduler != None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        print("lr: {}".format(optimizer.param_groups[0]["lr"]))
        train(train_loader, model, dis_batcher, optimizer, scheduler, cls_criterion, epoch, logger, args)

        valid_acc = valid(valid_seen_loader, model, dis_batcher, "valid seen", logger, args)
        valid_acc_unseen = valid(valid_unseen_loader, model, dis_batcher, "valid unseen", logger, args)
        test_acc = valid(test_seen_loader, model, dis_batcher, "test seen", logger, args)
        test_acc_unseen = valid(test_unseen_loader, model, dis_batcher, "test unseen", logger, args)
        is_best = valid_acc > best_acc
        if is_best:
            best_acc = valid_acc
        print("best_acc", best_acc)
        checkpoint_data = {
            'epoch': epoch + 1,
            'acc_val_seen': valid_acc,
            'acc_val_unseen': valid_acc_unseen,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if scheduler is not None:
            checkpoint_data["scheduler"] = scheduler.state_dict()
        save_checkpoint(checkpoint_data, is_best=is_best, folder=args.exp_name, filename='checkpoint.pth.tar')


def train(train_loader, model, dis_batcher, optimizer, scheduler, cls_criterion, epoch, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses1 = AverageMeter('LossTopic', ':.4e')
    losses2 = AverageMeter('LossKnow', ':.4e')
    topT = AverageMeter('Acc@Topic', ':6.2f')
    topK = AverageMeter('Acc@Know', ':6.2f')
    list_ = [batch_time, data_time, losses, losses1, losses2]

    losses4 = AverageMeter('LossHis', ':.4e')
    list_.append(losses4)

    list_ += [topT, topK]
    progress = ProgressMeter(
        len(train_loader),
        list_,
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, batch_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        histories = batch_data[1]
        dis_args = dis_batcher(*batch_data)
        inputs_args = list(dis_args)
        topic_tar, knowl_tar = inputs_args[-2], inputs_args[-1]
        inputs_args = inputs_args[:-2]

        # forward
        inputs_args += [topic_tar, knowl_tar]
        topics_logits, knowl_logits, his_topics_logits, his_knowl_logits = model(*inputs_args)

        # acc
        tacc = accuracy(topics_logits, topic_tar[:, 0])
        kacc = accuracy(knowl_logits, knowl_tar[:, 0])
        
        # loss
        tcls_loss = cls_criterion(topics_logits, topic_tar[:, 0])
        kcls_loss = cls_criterion(knowl_logits, knowl_tar[:, 0])
        loss = tcls_loss + kcls_loss

        # add history loss
        hisloss = 0.0
        for hi in range(len(his_topics_logits)):
            hisloss += cls_criterion(his_topics_logits[hi], topic_tar[:, hi+1])
            hisloss += cls_criterion(his_knowl_logits[hi], knowl_tar[:, hi+1])
        hisloss /= len(his_topics_logits)
        hisloss *= 0.5
        loss += hisloss
        losses4.update(hisloss.item(), len(histories))

        losses.update(loss.item(), len(histories))
        losses1.update(tcls_loss.item(), len(histories))
        losses2.update(kcls_loss.item(), len(histories))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        
        topT.update(tacc[0].item(), len(histories))
        topK.update(kacc[0].item(), len(histories))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i, logger)


@torch.no_grad()
def valid(valid_loader, model, dis_batcher, split, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    topT = AverageMeter('Acc@Topic', ':6.2f')
    topK = AverageMeter('Acc@Know', ':6.2f')
    list_ = [batch_time, data_time, topT, topK]
    test_len = len(valid_loader)
    progress = ProgressMeter(
        test_len,
        list_,
        prefix="Test %s: " % split)
    model.eval()
    end = time.time()
    for i, batch_data in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        histories = batch_data[1]
        dis_args = dis_batcher(*batch_data)
        inputs_args = list(dis_args)
        topic_tar, knowl_tar = inputs_args[-2], inputs_args[-1]
        inputs_args = inputs_args[:-2]

        knowl_to_topic = inputs_args[8]
        knowl_node_indicate = inputs_args[7]

        inputs_args += [topic_tar, knowl_tar]
        topics_logits, knowl_logits, _, _ = model(*inputs_args)

        # acc
        tacc = accuracy(topics_logits, topic_tar[:, 0])
        kacc = accuracy(knowl_logits, knowl_tar[:, 0])
        
        topT.update(tacc[0].item(), len(histories))
        topK.update(kacc[0].item(), len(histories))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % (args.print_freq*5)== 0 or i == (test_len-1):
            progress.display(i, logger)


    print("Acc@Topic: %.3f" % (topT.avg))
    print("Acc@Know: %.3f" % (topK.avg))
    return topK.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for GraphKS')

    parser.add_argument('--dataset', type=str, default='wow')

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
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
