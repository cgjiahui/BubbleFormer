# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    The matcher for the point
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: We remove it
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            "GT target label"
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:   #(predict_i, target_j)
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(
            -1)
        out_points = outputs["pred_points"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  # tgt_labels [bs, num_target_boxes]
        tgt_points = torch.cat([v["points"] for v in targets])  # [bs, num_target_boxes, 3]

        cost_class = -out_prob[:, tgt_ids.long()]

        flag = 0
        if len(tgt_points) != 0:
            cost_point = torch.cdist(out_points, tgt_points.float(),
                                     p=1)
            flag = 1
        C = self.cost_point * cost_point + self.cost_class * cost_class if flag else self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                indices]


class HungarianMatcher_triple_box(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    The matcher for the point
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: We remove it
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:   #(predict_i, target_j)
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_points = outputs["pred_points"].flatten(0, 1)

        "tgt label & points"
        tgt_ids = torch.cat([v["labels"] for v in targets])  # (bs*num_rooms)
        tgt_points = torch.cat([v["points"] for v in targets])  # (bs*num_rooms, 3)

        cost_class = -out_prob[:, tgt_ids]  # (bs*num_queries, bs*num_rooms)

        flag = 0
        if len(tgt_points) != 0:
            cost_point = torch.cdist(out_points, tgt_points.float(),
                                     p=1)
            flag = 1

        C = self.cost_point * cost_point + self.cost_class * cost_class if flag else self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                indices]


class Edge_matcher(nn.Module):
    ""

    def __init__(self, args):
        """Creates the matcher for the pair
        Params:
        """
        super().__init__()
        "Attention: This is the match weight"
        self.cost_type = args.set_cost_type
        self.cost_target_node = args.set_cost_target_node
        assert self.cost_type != 0 or self.cost_target_node != 0, "all costs cant be 0"

        self.num_edge_queries = args.num_edge_queries

    @torch.no_grad()
    def forward(self, outputs, targets, indices):

        bs, num_queries = outputs['pred_idx1'].shape[:2]
        return_list = []

        for batch_idx in range(bs):
            tgt_points = targets[batch_idx]['points']
            tgt_labels = targets[batch_idx]['labels']

            tgt_pnodes = targets[batch_idx]['pair_nodes']
            tgt_ptypes = torch.ones(len(tgt_pnodes))
            tgt_pnodes1 = tgt_pnodes[:, :3]
            tgt_pnodes2 = tgt_pnodes[:, 3:]

            k_idx, points_idx = indices[batch_idx]

            cost_pnodes1 = torch.cdist(tgt_pnodes1, tgt_points,
                                       p=1)
            cost_pnodes2 = torch.cdist(tgt_pnodes2, tgt_points, p=1)

            p1_match_indices = torch.nonzero(cost_pnodes1 == 0, as_tuple=False)
            p2_match_indices = torch.nonzero(cost_pnodes2 == 0, as_tuple=False)

            tgt_q4P1_idx, tgt_q4P2_idx = [], []

            for p1_match_idx, p2_match_idx in zip(p1_match_indices,
                                                  p2_match_indices):

                p1_nodes_idx, P1_points_idx = p1_match_idx
                p2_nodes_idx, P2_points_idx = p2_match_idx

                GT_idx_for_P1 = (points_idx == P1_points_idx.cpu()).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_P1 = k_idx[GT_idx_for_P1]
                tgt_q4P1_idx.append(query_idx_for_P1)

                GT_idx_for_P2 = (points_idx == P2_points_idx.cpu()).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_P2 = k_idx[GT_idx_for_P2]
                tgt_q4P2_idx.append(query_idx_for_P2)


            tgt_q4P1_idx = torch.cat(tgt_q4P1_idx)
            tgt_q4P2_idx = torch.cat(tgt_q4P2_idx)

            out_p1prob = outputs['pred_idx1'][batch_idx].softmax(-1)

            out_p2prob = outputs['pred_idx2'][batch_idx].softmax(-1)
            out_type = outputs['pred_type'][batch_idx].clone()

            cost_4_p1 = -out_p1prob[:, tgt_q4P1_idx]
            cost_4_p2 = -out_p2prob[:, tgt_q4P2_idx]


            cost_4_node = self.cost_target_node * cost_4_p1 + self.cost_target_node * cost_4_p2

            C = cost_4_node

            C = C.view(num_queries, -1).cpu()

            return_list.append(linear_sum_assignment(C))

            targets[batch_idx]['idx1_labels'] = tgt_q4P1_idx.to(tgt_pnodes1.device)
            targets[batch_idx]['idx2_labels'] = tgt_q4P2_idx.to(tgt_pnodes2.device)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                return_list], targets


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)


def build_edge_matcher(args):
    return Edge_matcher(args)
