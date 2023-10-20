# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime

from .feed_forward import MLP

class ETFR(nn.Module):
    def __init__(self, name, detr,
                 num_edge_queries,
                 num_types,
                 edge_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 return_obj_class=None):
        super().__init__()

        self.detr = detr
        self.name=name
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_edge_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.Pointer1_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.Pointer2_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.type_embed = nn.Linear(hidden_dim, num_types + 1)  # return connection types
        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)    #what obj class is
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1] + 1]
        # ------------------------------------------------------------------

        # * Transformer Options ---------------------------------------------
        self.edge_transformer = edge_transformer

        if share_enc:
            self.edge_transformer.encoder = detr.transformer.encoder

        if pretrained_dec:  # free variables for interaction decoder   and the separate decoder
            self.edge_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.edge_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------

        self.tau = temperature
        # ----------------------------------

    def forward(self, samples):
        "output of DETR model"
        out, detr_src, detr_pos, detr_hs, means, logvars = self.detr(samples)
        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)

        edge_hs = self.edge_transformer(self.detr.input_proj(detr_src), None, self.query_embed.weight, detr_pos)[0]

        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in   #what tau
                        Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                        Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]
        return out, means, logvars

    def inference(self, condition_input, z):
        c_feature = self.detr.cnet(condition_input)['0']
        feature,pos = self.detr.backbone(torch.cat([c_feature,z],1).to(torch.float32))
        src_key_padding_mask = None

        detr_src = feature[-1]
        detr_pos = pos[-1]

        detr_hs = self.detr.transformer(self.detr.input_proj(detr_src), src_key_padding_mask, self.detr.query_embed.weight, detr_pos)[0]

        out = {'pred_logits': self.detr.class_embed(detr_hs)[-1], 'pred_points': self.detr.point_embed(detr_hs).sigmoid()[-1]}

        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)
        edge_hs = self.edge_transformer(self.detr.input_proj(detr_src), None, self.query_embed.weight, detr_pos)[0]
        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in  # what tau
                       Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                       Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]

        return out


    def load_model(self, path, from_multi_GPU=False):
        if not from_multi_GPU:
            self.load_state_dict(torch.load(path)['model'])
        else:
            state_dict=torch.load(path)
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]
                new_state_dict[namekey] = v
            self.load_state_dict(new_state_dict)



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action,
                               outputs_tgt):
        return [
            {'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
            for a, b, c, d, e, f in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1],
                outputs_tgt[:-1])]

class ETFR_bin(nn.Module):
    def __init__(self, name, detr,
                 num_edge_queries,
                 num_types,
                 edge_cnet,
                 edge_backbone,
                 edge_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 return_obj_class=None):
        super().__init__()

        self.detr = detr
        self.name=name


        hidden_dim = edge_transformer.d_model    #统一向量维度
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_edge_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.Pointer1_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.Pointer2_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.type_embed = nn.Linear(hidden_dim, num_types + 1)
        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1] + 1]
        # ------------------------------------------------------------------

        self.edge_cnet = edge_cnet
        self.edge_backbone = edge_backbone
        # * Transformer Options ---------------------------------------------
        self.edge_transformer = edge_transformer

        if share_enc:  # share encoder
            self.edge_transformer.encoder = detr.transformer.encoder

        if pretrained_dec:  # free variables for interaction decoder   and the separate decoder
            self.edge_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.edge_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------

        # * Loss Options -------------------
        self.tau = temperature

    def forward(self, samples):
        "output of DETR model"
        out, input, detr_hs, means, logvars = self.detr(samples)
        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)

        c_feature = self.edge_cnet(input)['0']
        src, pos = self.edge_backbone(c_feature)

        edge_hs = self.edge_transformer(src[-1], None, self.query_embed.weight, pos[-1])[0]

        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in
                        Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                        Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]

        return out, means, logvars

    def inference(self, condition_input, z):

        c_feature = self.detr.cnet(condition_input)['0']
        feature,pos = self.detr.backbone(torch.cat([c_feature,z],1).to(torch.float32))
        src_key_padding_mask = None
        detr_src = feature[-1]
        detr_pos = pos[-1]
        detr_hs = self.detr.transformer(self.detr.input_proj(detr_src), src_key_padding_mask, self.detr.query_embed.weight, detr_pos)[0]
        out = {'pred_logits': self.detr.class_embed(detr_hs)[-1], 'pred_points': self.detr.point_embed(detr_hs).sigmoid()[-1]}

        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)
        edge_feature = self.edge_cnet(condition_input)['0']
        src_edge, pos_edge = self.edge_backbone(edge_feature)
        edge_hs = self.edge_transformer(src_edge[-1], None, self.query_embed.weight, pos_edge[-1])[0]
        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in
                       Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                       Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]
        return out

    def load_model(self, path, from_multi_GPU=False):
        if not from_multi_GPU:
            self.load_state_dict(torch.load(path)['model'])
        else:
            state_dict=torch.load(path)
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]
                new_state_dict[namekey] = v
            self.load_state_dict(new_state_dict)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action,
                               outputs_tgt):
        return [
            {'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
            for a, b, c, d, e, f in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1],
                outputs_tgt[:-1])]

class Construct_BubbleFormer(nn.Module):
    def __init__(self, name, detr,
                 num_edge_queries,
                 num_types,
                 edge_cnet,
                 edge_backbone,
                 edge_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 return_obj_class=None):
        super().__init__()

        self.detr = detr
        self.name=name


        hidden_dim = edge_transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_edge_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.Pointer1_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.Pointer2_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.type_embed = nn.Linear(hidden_dim, num_types + 1)  # return connection types
        self.Nodes_embed = MLP(8, 160, 325, 5)   #in, hidden, out dimension, num_layers

        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)    #what obj class is
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1] + 1]
        # ------------------------------------------------------------------

        self.edge_cnet = edge_cnet
        self.edge_backbone = edge_backbone
        # * Transformer Options ---------------------------------------------
        self.edge_transformer = edge_transformer

        if share_enc:
            self.edge_transformer.encoder = detr.transformer.encoder

        if pretrained_dec:
            self.edge_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.edge_transformer.decoder.parameters():
                p.requires_grad_(True)

        self.tau = temperature
        # ----------------------------------

    def forward(self, samples):
        "output of DETR model"
        samples.cuda()


        out, input, detr_hs, means, logvars = self.detr(samples)

        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)

        c_feature = self.edge_cnet(input)['0']    #bs c h w


        src, pos = self.edge_backbone(c_feature)


        detr_hs = detr_hs[-1]   #(bs, q, c)
        detr_hs = detr_hs.permute(0, 2, 1)
        detr_hs = self.Nodes_embed(detr_hs)
        edge_trans_src = torch.cat([src[-1].flatten(2),detr_hs], -1)

        edge_trans_src = edge_trans_src.reshape(4, 192, 35, 35)

        edge_hs = self.edge_transformer(edge_trans_src, None, self.query_embed.weight, pos[-1])[0]

        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in
                        Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                        Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]

        return out, means, logvars

    def inference(self, condition_input, z):

        c_feature = self.detr.cnet(condition_input)['0']
        feature,pos = self.detr.backbone(torch.cat([c_feature,z],1).to(torch.float32))
        src_key_padding_mask = None
        detr_src = feature[-1]
        detr_pos = pos[-1]
        detr_hs = self.detr.transformer(self.detr.input_proj(detr_src), src_key_padding_mask, self.detr.query_embed.weight, detr_pos)[0]
        out = {'pred_logits': self.detr.class_embed(detr_hs)[-1], 'pred_points': self.detr.point_embed(detr_hs).sigmoid()[-1]}

        node_repr = F.normalize(detr_hs[-1], p=2, dim=2)
        edge_feature = self.edge_cnet(condition_input)['0']
        src_edge, pos_edge = self.edge_backbone(edge_feature)
        pos_edge = pos_edge[0][0].reshape(1, 192, 35, 35)

        detr_hs = detr_hs[-1].permute(0, 2, 1)
        detr_hs = self.Nodes_embed(detr_hs)

        edge_trans_src = torch.cat([src_edge[-1].flatten(2), detr_hs], -1)
        edge_trans_src = edge_trans_src.reshape(1, 192, 35, 35)

        edge_hs = self.edge_transformer(edge_trans_src, None, self.query_embed.weight, pos_edge)[0]
        Pointer1_reprs = F.normalize(self.Pointer1_embed(edge_hs), p=2, dim=-1)
        Pointer2_reprs = F.normalize(self.Pointer2_embed(edge_hs), p=2, dim=-1)

        output_idx1 = [(torch.bmm(Pointer1_repr, node_repr.transpose(1, 2))) / self.tau for Pointer1_repr in
                       Pointer1_reprs]
        output_idx2 = [(torch.bmm(Pointer2_repr, node_repr.transpose(1, 2))) / self.tau for Pointer2_repr in
                       Pointer2_reprs]
        output_type = self.type_embed(edge_hs)

        out['pred_idx1'] = output_idx1[-1]
        out['pred_idx2'] = output_idx2[-1]
        out['pred_type'] = output_type[-1]

        return out

    def load_model(self, path, from_multi_GPU=False):
        if not from_multi_GPU:
            self.load_state_dict(torch.load(path)['model'])
        else:
            state_dict=torch.load(path)
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]
                new_state_dict[namekey] = v
            self.load_state_dict(new_state_dict)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action,
                               outputs_tgt):
        return [
            {'pred_logits': a, 'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
            for a, b, c, d, e, f in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1],
                outputs_tgt[:-1])]

