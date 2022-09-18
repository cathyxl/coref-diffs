import os
import torch
from torch.nn import functional as F
import json
import re
import numpy as np
from torch.utils.data import DataLoader, Dataset

from itertools import cycle
from transformers import BertTokenizer, GPT2Tokenizer
import transformers
import random
import pickle

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
    segment_ids=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    res = [input_ids[:, keep_column_mask]]
    if attention_mask != None:
        res.append(attention_mask[:, keep_column_mask])
    if segment_ids != None:
        res.append(segment_ids[:, keep_column_mask])
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def collate_fn(batch):
    knowledges        = [item[0] for item in batch]
    histories         = [item[1] for item in batch]
    users             = [item[2] for item in batch]
    responses         = [item[3] for item in batch]
    topics            = [item[4] for item in batch]
    topics_knowledges = [item[5] for item in batch]
    topics_rela       = [item[6] for item in batch]
    knowl_topic_rela  = [item[7] for item in batch]
    knowl_order_rela  = [item[8] for item in batch]
    ents              = [item[9] for item in batch]
    topic_tar         = [item[10] for item in batch] # list([b, his_depth+1])
    knowl_tar         = [item[11] for item in batch] # list([b, his_depth+1])
    return knowledges, histories, users, responses, topics, topics_knowledges, topics_rela, knowl_topic_rela, knowl_order_rela, ents, topic_tar, knowl_tar

class GraphKSDataset(Dataset):
    def __init__(self, data_dir, split, args):
        # load data
        data_path = "{}/dialog/{}.jsonl".format(data_dir, split)
        print("loading data from", data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self._data = [json.loads(line) for line in f.readlines()]

        graph_dir = "topic_relations"

        misc_data_path = "{}/{}/{}.jsonl".format(data_dir, graph_dir, split)
        print("loading graph from", misc_data_path)
        with open(misc_data_path, 'r', encoding='utf-8') as f:
            self._misc_data = [json.loads(line) for line in f.readlines()]


        graph_dir = "knowled_relations"
        extra_knowl_rela_path = "{}/{}/coref_{}.jsonl".format(data_dir, graph_dir, split)
        print("loading extra knowledge relations from", extra_knowl_rela_path)
        with open(extra_knowl_rela_path, 'r', encoding='utf-8') as f:
            self._extra_knowl_rela = [json.loads(line) for line in f.readlines()]

        edge_vocab_path = "{}/{}/edge_vocab.txt".format(data_dir, graph_dir)
        print("loading edge_vocab from", edge_vocab_path)
        with open(edge_vocab_path, 'r', encoding='utf-8') as f:
            self._edge_vocab = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
            self._edge_vocab = {v:i for i, v in enumerate(self._edge_vocab)}

        self._n_data = len(self._data)
        self.args = args
        print(split, "dataset size", self._n_data)

    def get_edge_vocab_size(self):
        return len(self._edge_vocab)

    def parse_knowledge_sentence(self, string):
        split1 = string.split(" __topic__ ")
        if len(split1) < 2:
            raise Exception("{} Missing information".format(string))
        sent_order = int(split1[0].strip())
        split2 = split1[1].split(" __knowledge__ ")
        topic = split2[0]
        if len(split2) < 2:
            # if knowledge sentence is null，using topic sentence instead
            knowledge = topic
        else:
            knowledge = split2[1]
        return sent_order, topic, knowledge

    def __len__(self):
        return self._n_data

    def __getitem__(self, data_i):
        knowledges = self._data[data_i]['knowledge']
        history = self._data[data_i]['history'][::-1] # ["bla bla...", "bla bla.."]
        response = self._data[data_i]['response'] # text
        history_idx = self._data[data_i]['history_idx'][::-1]
        user = self._data[data_i]['user'][::-1]
        history_topics_extraction = self._misc_data[data_i]['history_topics_extraction'][::-1]
        if len(history_topics_extraction) > len(history):
            history_topics_extraction = history_topics_extraction[:len(history)]

        topics = self._misc_data[data_i]['topics'] # ["blue", "royle blue(film)"]
        topics_rela = self._misc_data[data_i]['topics_ngram_rela'] # [[0, 1, "relation xxx"]]
        topics_rela = []
        topics_rela += self._misc_data[data_i]['topics_wiki_rela']

        ents = [] # ["blue", "blue and royle blue", ""]
        for item in history_topics_extraction:
            ent_str_list = []
            for ent in item:
                ent_str_list += [topics[idx] for idx in ent['topic_index']]
            if len(ent_str_list) > 0:
                ents.append(" and ".join(ent_str_list))
            else:
                ents.append("")

        try:
            topics_rela = [[item[0], item[1], self._edge_vocab[item[2]]] for item in topics_rela] # [[0, 1, 3],...]
        except Exception as e:
            raise Exception("Err: topics relation not in edge vocab")

        # 提取knowledge与topic的关系
        knowledges_text_order = {}
        knowl_topic_rela = [] # [[0, 1, 3],...]
        knowl_order_rela = [] # [[0, 1, 3],...]

        knowl_tar = [0,] + history_idx
        topic_tar = [-1,] * len(knowl_tar)
        for i in range(len(knowledges)):
            k = knowledges[i]
            sent_order, topic, knowledges[i] = self.parse_knowledge_sentence(k)
            topic_index = topics.index(topic)
            for j in range(len(knowl_tar)):
                if i == knowl_tar[j]:
                    topic_tar[j] = topic_index
            knowl_index = i
            try:
                rela_index = self._edge_vocab["the %dth sentence of"%sent_order]
            except Exception as e:
                raise Exception("Err: knowledges relation not in edge vocab")
            knowl_topic_rela.append([knowl_index, topic_index, rela_index])

            if topic not in knowledges_text_order:
                knowledges_text_order[topic] = []
            knowledges_text_order[topic].append([sent_order, knowledges[i]])
        topics_knowledges = {}
        topics_knowledges_order = {}
        for topic in knowledges_text_order:
            knowledges_text_order[topic] = sorted(knowledges_text_order[topic], key=lambda x: x[0])
            topics_knowledges[topic] = [x[1] for x in knowledges_text_order[topic]]
            topics_knowledges_order[topic] = [x[0] for x in knowledges_text_order[topic]]
            max_order = max(topics_knowledges_order[topic])
            if max_order == 0:
                max_order = 1
                min_order = 0
            else:
                if len(topics_knowledges_order[topic]) > 2:
                    min_order = min(topics_knowledges_order[topic][1:])
                else:
                    min_order = 0
            topics_knowledges_order[topic] = [(x, max_order, min_order, float(x) / max_order) for x in topics_knowledges_order[topic]]

        # get new knowledge order
        new_order_knowledges = []
        for topic in topics:
            new_order_knowledges += topics_knowledges[topic]

        for kr in knowl_topic_rela:
            k = knowledges[kr[0]]
            kindex = new_order_knowledges.index(k)
            kr[0] = kindex

        # change the kg target index
        for ki in range(len(knowl_tar)):
            if knowl_tar[ki] != -1:
                k = knowledges[knowl_tar[ki]]
                knowl_tar[ki] = new_order_knowledges.index(k)

        for rela in self._extra_knowl_rela[data_i]["knowledge_relations"]:
            k1_index = rela[0]
            k2_index = rela[1]
            knowl_order_rela.append([new_order_knowledges.index(knowledges[k1_index]), new_order_knowledges.index(knowledges[k2_index]), self._edge_vocab[rela[2]]])
                

        return (knowledges, history, user, response, topics, topics_knowledges, topics_rela, knowl_topic_rela, knowl_order_rela, ents, topic_tar, knowl_tar)


class GraphKSBatcher:

    def __init__(self, block_size, bert_config):
        self.block_size = 512
        self.tokenizer = BertTokenizer.from_pretrained(bert_config, do_lower_case=True)
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.self_relation_id = 1
        # add special token
        self.cls_token = "[cls]"
        self.usr_token = "[usr]"
        self.bot_token = "[bot]"
        self.ent_token = "[ent]"
        
        self.tokenizer.add_tokens([self.cls_token], special_tokens=True)
        self.tokenizer.add_tokens([self.usr_token], special_tokens=True)
        self.tokenizer.add_tokens([self.bot_token], special_tokens=True)
        self.tokenizer.add_tokens([self.ent_token], special_tokens=True)
        self.cls_token_id = len(self.tokenizer) - 4
        self.usr_token_id = len(self.tokenizer) - 3
        self.bot_token_id = len(self.tokenizer) - 2
        self.ent_token_id = len(self.tokenizer) - 1

    def tokenize(self, text, text_pair=None, max_length=64):
        if transformers.__version__.startswith("2."):
            return self.tokenizer.encode_plus(text,
                              text_pair=text_pair,
                              add_special_tokens=True,
                              return_attention_mask=True,
                              return_token_type_ids=True,
                              pad_to_max_length=True,
                              max_length=max_length)
        else:
            return self.tokenizer.encode_plus(text,
                              text_pair=text_pair,
                              add_special_tokens=True,
                              return_attention_mask=True,
                              return_token_type_ids=True,
                              truncation=True,
                              padding="max_length",
                              max_length=max_length)

    def gather_special_tokens(self, input_ids, special_tokens_ids):
        """
        return [(token_id, offset), ...]
        """
        return [(id_, i) for i, id_ in enumerate(input_ids) if id_ in special_tokens_ids]

    def get_context_input(self, his, usr, ent):
        cxt_str = ""
        for hi in range(len(his)):
            if usr[hi] == 0:
                role = self.usr_token
            else:
                role = self.bot_token
            cxt_str += "{} {} {} {} ".format(role, his[hi], self.ent_token, ent[hi])
        if len(cxt_str) > 0:
            cxt_str = cxt_str[:-1]
        return cxt_str

    def get_passage_input(self, top, kno, kept=1.0):
        passage_text = top
        for k in kno:
            k_text = k
            if kept < 1:
                k_tokens = k_text.split(" ")
                k_text = " ".join(k_tokens[:int(len(k_tokens)*kept)])
            passage_text += " {} {}".format(self.cls_token, k_text)
        return passage_text

    def __call__(self, knowledges, histories, users, responses, topics, topics_knowledges, 
        topics_rela, knowl_topic_rela, knowl_order_rela, ents, topic_tar, knowl_tar):
        input_ids = [] # [bs, topics_len, seq_len](torch.long)
        segment_ids = [] # [bs, topics_len, seq_len](torch.long)
        attention_masks = [] # [bs, topics_len, seq_len](torch.bool)
        candidate_offsets = [] # [bs, k_len, 2](torch.long)
        is_his = [] # [bs, topics_len, his_depth, seq_len](torch.bool), one hot

        graph_adj = [] # [bs, node_len, node_len](torch.long)
        topic_node_indicate = [] # [bs, node_len](torch.long) 1 for valid, 2 for pad, 0 for non-topic
        knowl_node_indicate = [] # [bs, node_len](torch.long) 1 for valid, 2 for pad, 0 for non-knowledge
        knowl_to_topic = [] # [bs, k_len, topics_len](torch.long)

        max_knowledge = 0
        max_topic = 0
        his_depth = 0
        for his, kg, topic in zip(histories, knowledges, topics):
            max_knowledge = max(max_knowledge, len(kg))
            max_topic = max(max_topic, len(topic))
            his_depth = max(his_depth, len(his))
        node_len = max_topic + max_knowledge

        empty_input_id = empty_segment_id = empty_attention_mask = [self.pad_id] * self.block_size
        empty_his_idx = [0] * his_depth

        for his, usr, kg, response, topic, topic_knowledges, topic_r, knowl_topic_r, knowl_order_r, ent in zip(histories, users, knowledges, responses, 
            topics, topics_knowledges, topics_rela, knowl_topic_rela, knowl_order_rela, ents):
            cxt_str = self.get_context_input(his, usr, ent)
            kg_len = len(kg)
            topic_len = len(topic)
            ids = []
            type_ids = []
            att_masks = []
            is_his_ = []
            candidate_offsets_ = [] # [kl, 2]
            for ti, top in enumerate(topic):
                k_per_t = len(topic_knowledges[top])
                passage_text = self.get_passage_input(top, topic_knowledges[top])
                passage_input = self.tokenize(cxt_str, passage_text, max_length=self.block_size)
                # knowledge sentence offset
                special_tokens = self.gather_special_tokens(passage_input["input_ids"], [self.cls_token_id])
                offsets = [[ti, st[1]] for st in special_tokens]
                if len(offsets) < k_per_t:
                    passage_text = self.get_passage_input(top, topic_knowledges[top], kept=0.9)
                    passage_input = self.tokenize(cxt_str, passage_text, max_length=self.block_size)
                    # knowledge sentence offset
                    special_tokens = self.gather_special_tokens(passage_input["input_ids"], [self.cls_token_id])
                    offsets = [[ti, st[1]] for st in special_tokens]
                    if len(offsets) < k_per_t:
                        print("passage knowledge overflow, total knowledges {} found {}\ncxt {}\npassage {}\n{}\n{}".format(
                            k_per_t, len(offsets), cxt_str, passage_text, len(cxt_str.split(" ")), len(passage_text.split(" "))))
                elif len(offsets) > k_per_t:
                    raise Exception("offsets length out of size, total knowledges {} found {}\ncxt {}\npassage {}\n{}\n{}".format(
                        k_per_t, len(offsets), cxt_str, passage_text, len(cxt_str.split(" ")), len(passage_text.split(" "))))
                candidate_offsets_ += offsets
                ids.append(passage_input["input_ids"])
                type_ids.append(passage_input["token_type_ids"])
                att_masks.append(passage_input["attention_mask"])
                # his offset
                special_tokens = self.gather_special_tokens(ids[-1], 
                    [self.usr_token_id, self.bot_token_id])
                his_offsets = [st[1] for st in special_tokens]
                # padding
                his_offsets += [his_offsets[0],] * (his_depth - len(his_offsets))
                is_his_.append(his_offsets)

            ids += [empty_input_id] * (max_topic - topic_len)
            type_ids += [empty_segment_id] * (max_topic - topic_len)
            att_masks += [empty_attention_mask] * (max_topic - topic_len)
            is_his_ += [empty_his_idx] * (max_topic - len(is_his_))
            candidate_offsets_ += [[max_topic-1, 0]] * (max_knowledge - len(candidate_offsets_))
            input_ids.append(ids)
            segment_ids.append(type_ids)
            attention_masks.append(att_masks)
            is_his.append(is_his_)
            candidate_offsets.append(candidate_offsets_)
            topic_node_indicate.append([1]*topic_len + [2]*(max_topic-topic_len) + [0]*max_knowledge)
            knowl_node_indicate.append([0]*max_topic + [1]*kg_len + [2]*(max_knowledge-kg_len))


            # graph
            adj = np.zeros((node_len, node_len), dtype=np.int)
            for r in topic_r:
                adj[r[0]][r[1]] = r[2]
                adj[r[1]][r[0]] = r[2]
            for r in knowl_topic_r:
                # knowledge node在topic node后面
                k_index = r[0] + max_topic
                t_index = r[1]
                adj[k_index][t_index] = r[2]
                adj[t_index][k_index] = r[2]
            # add knowledge to knowledge relation, undirectional
            for r in knowl_order_r:
                src_knowl_index = r[0] + max_topic
                dst_knowl_index = r[1] + max_topic
                adj[dst_knowl_index][src_knowl_index] = r[2]
            # add self-relation to graph
            for gi in range(adj.shape[0]):
                adj[gi][gi] = self.self_relation_id

            graph_adj.append(adj)

        # batch trim
        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda() # [bs, node_len, l]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).cuda() # [bs, node_len, l]
        attention_masks = torch.tensor(attention_masks, dtype=torch.bool).cuda() # [bs, node_len, l]
        bsz, _, l  = input_ids.size()
        input_ids = input_ids.view(bsz*max_topic, l)
        segment_ids = segment_ids.view(bsz*max_topic, l)
        attention_masks = attention_masks.view(bsz*max_topic, l)
        input_ids, attention_masks, segment_ids = trim_batch(input_ids, self.pad_id, 
            attention_mask=attention_masks, segment_ids=segment_ids)
        input_ids = input_ids.view(bsz, max_topic, -1)
        segment_ids = segment_ids.view(bsz, max_topic, -1)
        attention_masks = attention_masks.view(bsz, max_topic, -1)
        topic_node_indicate = torch.tensor(topic_node_indicate, dtype=torch.long).cuda() # [bs, node_len]
        knowl_node_indicate = torch.tensor(knowl_node_indicate, dtype=torch.long).cuda() # [bs, node_len]
        graph_adj = torch.tensor(graph_adj, dtype=torch.long).cuda() # [bs, node_len, node_len]

        # contruct knowledge2topic transition matrix
        knowl_to_topic = graph_adj[:, max_topic:max_topic+max_knowledge, :max_topic]
        knowl_to_topic = (knowl_to_topic > 0).long()


        # topic_tar[bs, his_depth+1], knowl_tar[bs, his_depth+1], 因为0是current gt
        topic_tar = [tar + [-1] * (his_depth + 1 - len(tar)) for tar in topic_tar]
        knowl_tar = [tar + [-1] * (his_depth + 1 - len(tar)) for tar in knowl_tar]
        topic_tar = torch.tensor(topic_tar, dtype=torch.long).cuda()
        knowl_tar = torch.tensor(knowl_tar, dtype=torch.long).cuda()

        is_his = torch.tensor(is_his, dtype=torch.long).cuda()
        is_his = F.one_hot(is_his, num_classes=input_ids.shape[2]).bool()

        candidate_offsets = torch.tensor(candidate_offsets, dtype=torch.long).cuda()

        returns = [input_ids, segment_ids, attention_masks, candidate_offsets, is_his, graph_adj, topic_node_indicate, knowl_node_indicate, knowl_to_topic, topic_tar, knowl_tar]

        return tuple(returns)

def get_batch_loader(dataset, collate_fn, batch_size=2, num_workers=0, is_test=True):
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(not is_test), num_workers=num_workers, collate_fn=collate_fn
    )
    return loader
