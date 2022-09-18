import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import numpy as np
import math
import os
import random
from transformers import BertModel, BertConfig
from graph_encoder import RGATEncoderLayer


class GraphKSModel(nn.Module):
    def __init__(self, args):
        print("init", self.__class__.__name__)
        super(GraphKSModel, self).__init__()
        edge_size = args.edge_size
        self.edge_hidden_size = args.edge_hidden_size
        self.edge_embedding = nn.Embedding(edge_size, self.edge_hidden_size)
        self.transformer = BertModel.from_pretrained(args.bert_config)
        config = BertConfig.from_pretrained(args.bert_config)
        if args.encoder_out_dim > 0:
            self.encoder_out_dim = args.encoder_out_dim
            self.encoder_trans = nn.Sequential(
                nn.Linear(config.hidden_size, args.encoder_out_dim),
            )
        else:
            self.encoder_out_dim = config.hidden_size
        self.gat_hid_dim = args.gat_hid_dim
        self.dropout = args.dropout
        self.gat_header = args.gat_header

        self.topic_cls_layer = nn.Sequential(
            nn.Linear(self.encoder_out_dim*2, 1),
        )
        self.knowl_cls_layer = nn.Sequential(
            nn.Linear(self.encoder_out_dim*4+self.edge_hidden_size, self.encoder_out_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_out_dim, 1),
        )


        self.his_topic_cls_layer = nn.Sequential(
            nn.Linear(self.encoder_out_dim, 1),
        )
        self.his_knowl_cls_layer = nn.Sequential(
            nn.Linear(self.encoder_out_dim*2+self.edge_hidden_size, self.encoder_out_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_out_dim, 1),
        )

        # gnn, res
        l = [RGATEncoderLayer(self.encoder_out_dim, self.edge_hidden_size,
                self.gat_hid_dim, self.dropout, args.gat_alpha,
                self.gat_header, ofeat=self.encoder_out_dim),]
        l.append(RGATEncoderLayer(self.encoder_out_dim*2, self.edge_hidden_size,
                self.gat_hid_dim, self.dropout, args.gat_alpha,
                self.gat_header, ofeat=self.encoder_out_dim))
        self.graph_encoder = nn.ModuleList(l)
        self.graph_node_trans = nn.Sequential(
            nn.Linear(self.encoder_out_dim*2, self.encoder_out_dim),
            nn.ReLU(),
        )


        # gru
        self.gru1 = nn.GRU(self.encoder_out_dim*2, self.encoder_out_dim, 2)
        self.gru1_hidden0 = Parameter(torch.zeros(2, 1, self.encoder_out_dim), requires_grad=True)

        self.gru2 = nn.GRU((self.encoder_out_dim*2+self.edge_hidden_size)*2, self.encoder_out_dim, 2)
        self.gru2_hidden0 = Parameter(torch.zeros(2, 1, self.encoder_out_dim), requires_grad=True)



    def forward(self, input_ids, segment_ids, attention_masks, candidate_offsets, is_his,
                graph_adj, topic_node_indicate, knowl_node_indicate, knowl_to_topic, topic_tar, knowl_tar):
        """
        inputs:
            input_ids:       [bs, topics_len, seq_len](torch.long)
            segment_ids:     [bs, topics_len, seq_len](torch.long)
            attention_masks: [bs, topics_len, seq_len](torch.bool)
            candidate_offsets: [bs, k_len, 2](torch.long)
            is_his:          [bs, topics_len, his_depth, seq_len](torch.bool), one hot
            graph_adj:            [bs, node_len, node_len](torch.long)
            topic_node_indicate:  [bs, node_len](torch.long) 1 for topic, 2 for pad, 0 for knowledge
            knowl_node_indicate:  [bs, node_len](torch.long) 1 for knowledge, 2 for pad, 0 for topic
            knowl_to_topic:   [bs, k_len, topics_len](torch.long)
            topic_tar:        [bs, his_depth+1], 因为0是current gt
            knowl_tar:        [bs, his_depth+1]
        return:
            topic_logits:     [bs, topics_len]
            knowl_logits:     [bs, k_len]
        """
        bsz, tl, l = input_ids.size()
        kl = knowl_to_topic.size()[1]
        his_depth = is_his.size()[2]

        # indicate processing
        is_topic_node = topic_node_indicate>0
        is_topic_node_padding = topic_node_indicate[is_topic_node].view(bsz, -1) # b*tl
        is_knowl_node = knowl_node_indicate>0
        is_knowl_node_padding = knowl_node_indicate[is_knowl_node].view(bsz, -1) # b*kl
        # print("bsz, t, k, his_depth", bsz, tl, kl, his_depth)

        # encoder
        input_ids = input_ids.view(-1, l)
        segment_ids = segment_ids.view(-1, l)
        attention_masks = attention_masks.view(-1, l)
        is_his = is_his.view(-1, his_depth, l)
        # last_hidden_state[b*tl, seq_len, d]
        passage_res = self.transformer(input_ids, attention_masks, segment_ids, return_dict=True)
        last_hiddens = passage_res.last_hidden_state # (b*tl, l, d)
        if last_hiddens.shape[-1] != self.encoder_out_dim:
            last_hiddens = self.encoder_trans(last_hiddens)
        topic_hiddens = last_hiddens[:,0,:].view(bsz, tl, -1) # b*tl*d
        his_topic_hiddens = [last_hiddens[is_his[:,hi,:]].view(bsz, tl, -1) for hi in range(his_depth)]  # b*tl*d

        # edge encoder
        edge_hiddens = self.edge_embedding(graph_adj) # b*nl*nl*d1

        # obtain knode hiddens b*kl*d
        last_hiddens = last_hiddens.view(bsz, tl, l, -1)
        knowl_hiddens = []
        for bi in range(bsz):
            knowl_hiddens_ = [last_hiddens[bi, candidate_offsets[bi, ki, 0], candidate_offsets[bi, ki, 1], :] for ki in range(kl)]
            knowl_hiddens_ = torch.stack(knowl_hiddens_, dim=0) # kl*d
            knowl_hiddens.append(knowl_hiddens_)
        knowl_hiddens = torch.stack(knowl_hiddens, dim=0) # b*kl*d

        # obtain node
        node_hiddens = torch.cat((topic_hiddens, knowl_hiddens), dim=1) # b*nl*d
        knowl_to_topic = knowl_to_topic.type_as(node_hiddens)
        # obtain know2topic edge hiddens
        knowl_topic_edge_hiddens = edge_hiddens[:, tl:tl+kl, :tl, :] # b*kl*tl*d1
        knowl_topic_graph_adj = knowl_to_topic.unsqueeze(-1).expand(-1, -1, -1, self.edge_hidden_size).contiguous() # b*kl*tl*d1
        knowl_topic_edge_hiddens = torch.mul(knowl_topic_edge_hiddens, knowl_topic_graph_adj) # b*kl*tl*d1
        knowl_topic_edge_hiddens = torch.sum(knowl_topic_edge_hiddens, dim=2) # b*kl*d1
        

        # res gnn for graph
        node_hiddens_trans0 = self.graph_encoder[0](node_hiddens, 
            edge_hiddens, graph_adj) # b*nl*d
        node_hiddens_trans1 = self.graph_encoder[1](torch.cat([node_hiddens, node_hiddens_trans0], dim=-1), 
            edge_hiddens, graph_adj)
        node_hiddens_gnn_trans = self.graph_node_trans(torch.cat([node_hiddens_trans0, node_hiddens_trans1], dim=-1))

        # temporal encode and concat
        topic_node_tem_input = node_hiddens_gnn_trans[:,:tl,:].contiguous()
        knowl_node_tem_input = node_hiddens_gnn_trans[:,tl:(tl+kl),:]
        knowl_node_tem_input = torch.cat([knowl_node_tem_input,
            torch.bmm(knowl_to_topic, topic_node_tem_input), 
            knowl_topic_edge_hiddens], dim=-1).contiguous()
        node_hiddens_tem_trans = torch.cat((
            self.temporal_cross_encoding(topic_node_tem_input, topic_tar[:, 1:], self.gru1, self.gru1_hidden0), 
            self.temporal_cross_encoding(knowl_node_tem_input, knowl_tar[:, 1:], self.gru2, self.gru2_hidden0),
        ), dim=1) # b*nl*d
        node_hiddens_gnn_trans1 = torch.cat((node_hiddens_gnn_trans, node_hiddens_tem_trans), dim=2)


        # get Node hiddens after trans
        # obtain topic hiddens
        topic_hiddens_trans = node_hiddens_gnn_trans1[:,:tl,:] # b*tl*d
        # obtain know hiddens
        knowl_hiddens_trans = node_hiddens_gnn_trans1[:,tl:tl+kl,:] # b*kl*d

        # obtain topic logit
        topic_logits = self.topic_cls_layer(topic_hiddens_trans).squeeze(-1) # b*tl

        # obtain k logit
        knowl_hiddens_trans = torch.cat([knowl_hiddens_trans, 
            torch.bmm(knowl_to_topic, topic_hiddens_trans), 
            knowl_topic_edge_hiddens], dim=-1)
        knowl_logits = self.knowl_cls_layer(knowl_hiddens_trans).squeeze(-1) # b*kl     
        

        # mask padding nodes
        zero_topic_vec = float("-inf")*torch.ones_like(topic_logits)
        topic_logits = torch.where(is_topic_node_padding > 1, zero_topic_vec, topic_logits) # b*tl
        zero_knowl_vec = float("-inf")*torch.ones_like(knowl_logits)
        knowl_logits = torch.where(is_knowl_node_padding > 1, zero_knowl_vec, knowl_logits) # b*kl


        his_topic_logits = []
        his_knowl_logits = []
        for i in range(len(his_topic_hiddens)):
            his_knowl_hidden = knowl_hiddens
            his_node_hiddens = torch.cat((his_topic_hiddens[i], his_knowl_hidden), dim=1) # b*nl*d

            # res gnn
            his_node_hiddens_trans0 = self.graph_encoder[0](his_node_hiddens, edge_hiddens, graph_adj) # b*nl*d
            his_node_hiddens_trans1 = self.graph_encoder[1](torch.cat([his_node_hiddens, his_node_hiddens_trans0], dim=-1), 
                edge_hiddens, graph_adj)
            his_node_hiddens_trans = self.graph_node_trans(torch.cat([his_node_hiddens_trans0, his_node_hiddens_trans1], dim=-1))


            his_topic_hiddens_trans = his_node_hiddens_trans[:,:tl,:]
            his_knowl_hiddens_trans = his_node_hiddens_trans[:,tl:tl+kl,:]

            his_topic_logits.append(self.his_topic_cls_layer(his_topic_hiddens_trans).squeeze(-1))
            his_knowl_hiddens_trans = torch.cat([his_knowl_hiddens_trans, 
                torch.bmm(knowl_to_topic, his_topic_hiddens_trans), 
                knowl_topic_edge_hiddens], dim=-1)
            his_knowl_logits.append(self.his_knowl_cls_layer(his_knowl_hiddens_trans).squeeze(-1))

            # mask padding nodes
            his_topic_logits[-1] = torch.where(is_topic_node_padding > 1, zero_topic_vec, his_topic_logits[-1])
            his_knowl_logits[-1] = torch.where(is_knowl_node_padding > 1, zero_knowl_vec, his_knowl_logits[-1])

        return topic_logits, knowl_logits, his_topic_logits, his_knowl_logits

    def temporal_cross_encoding(self, node_hiddens, his_ind, rnn, hidden0):
        bsz, n, d = node_hiddens.size()
        his_depth = his_ind.size()[1]
        if his_depth == 1:
            t = node_hiddens.view(bsz*n, d)
            t_in = self.cross_core1(t, t)
            seq_input = t_in.unsqueeze(0)

        elif his_depth <= 3:
            t = node_hiddens.view(bsz*n, d)
            t_1 = self.construct_last(node_hiddens, his_ind[:, 1]).view(bsz*n, d)
            t_1_in = self.cross_core1(t_1, t)
            seq_input = t_1_in.unsqueeze(0)


        elif his_depth == 4:
            t = node_hiddens.view(bsz*n, d)
            t_1 = self.construct_last(node_hiddens, his_ind[:, 1]).view(bsz*n, d)
            t_2 = self.construct_last(node_hiddens, his_ind[:, 3]).view(bsz*n, d)
            t_2_in = self.cross_core1(t_2, t)
            t_1_in = self.cross_core1(t_1, t)
            seq_input = torch.stack((t_2_in, t_1_in), dim=0)
        else:
            raise Exception("his depth overflow")
        hidden0 = hidden0.repeat(1, bsz*n, 1).contiguous()
        rnn.flatten_parameters()
        # seq_input(L, bsz*n, d), output(L, bsz*n, d)
        L = seq_input.size(0)
        output, _ = rnn(seq_input, hidden0)
        last_hidden = output[-1, :, :].view(bsz, n, self.encoder_out_dim)
        return last_hidden

    def cross_core1(self, tensor_a, tensor_b):
        """
        Return: [a-b;a*b]
        """
        return torch.cat((tensor_a-tensor_b, tensor_a*tensor_b), dim=-1)

    def construct_last(self, node_hiddens, last_ind):
        """
        node_hiddens: (bsz, n, d)
        last_ind: (bsz, )
        """
        bsz, n, d = node_hiddens.size()
        last_node_hiddens = []
        for bi in range(bsz):
            if last_ind[bi] == -1:
                last_node_hiddens.append(node_hiddens[bi, :, :])
            else:
                last_node_hiddens.append(node_hiddens[bi, last_ind[bi], :].unsqueeze(0).expand(n, -1))
        last_node_hiddens = torch.stack(last_node_hiddens, dim=0)
        return last_node_hiddens
        

