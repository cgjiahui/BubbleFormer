import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
# from models import build_LayoutGraph_bin_mixed

class Set_TRF_Criterion(nn.Module):
    """
    losses: 'labels', 'points', 'cardinality', 'kld'
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, edge_losses, edge_matcher, args=None):

        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.edge_losses = edge_losses
        self.TRF_matcher = edge_matcher

        empty_weight = torch.ones(self.num_classes+1)
        empty_weight[-1] = 1     #
        self.empty_weight = empty_weight.cuda()



    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.long()


        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty points
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}
        return losses
    def loss_points(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding points, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "points" containing a tensor of dim [nb_target_boxes, 4]
           The target points are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        if len(target_points)==0:
            losses['loss_points']=0
        else:
            loss_bbox = F.l1_loss(src_points, target_points, reduction='none')
            losses['loss_points'] = loss_bbox.sum() / num_boxes
        return losses

    def loss_kld(self, outputs, targets, edge_indices, num_points, means, logvars, **kwargs):
        losses = {}
        kld_loss = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp())
        losses['loss_kld'] = kld_loss
        return losses

    def loss_pair_target(self, outputs, targets, edge_indices, num_boxes, **kwargs):
        """
        outputs:
        'pred_logits: the label of the nodes'  'pred_points: the coordinate of the nodes'    #boxes' label & pos
        ETFR(outputs:
        'pred_idx1: node1, 'pred_idx2': node2, 'pred_type': the edge label
        targets:
        'points': (num_nodes, 3)
        'labels': (num_nodes)  the room label array
        'pair_nodes'
        'idx1_labels'
        'idx2_labels'
        """
        src_idx1 = outputs['pred_idx1']
        src_idx2 = outputs['pred_idx2']
        idx = self._get_src_permutation_idx(edge_indices)

        target_idx1_classes = torch.full(src_idx1.shape[:2], -1, dtype=torch.int64, device=src_idx1.device)
        target_idx2_classes = torch.full(src_idx2.shape[:2], -1, dtype=torch.int64, device=src_idx2.device)

        target_classes_1 = torch.cat([t['idx1_labels'] for t,(_, J) in zip(targets, edge_indices)])


        target_idx1_classes[idx] = target_classes_1

        target_classes_2 = torch.cat([t['idx2_labels'] for t,(_,J) in zip(targets, edge_indices)])
        target_idx2_classes[idx] = target_classes_2

        loss_idx1 = F.cross_entropy(src_idx1.transpose(1,2), target_idx1_classes, ignore_index=-1)
        loss_idx2 = F.cross_entropy(src_idx2.transpose(1,2), target_idx2_classes, ignore_index=-1)

        losses = {'loss_idx1': loss_idx1, 'loss_idx2': loss_idx2}

        return losses



    def loss_pair_type(self, outputs, targets, edge_indices, num_boxes, **kwargs):

        src_type = outputs['pred_type']
        idx = self._get_src_permutation_idx(edge_indices)

        logits = src_type.sigmoid()

        target_type = torch.full(src_type.shape[:2], 0, dtype=torch.int64, device=src_type.device)

        target_type[idx] = 1


        loss_bce = F.cross_entropy(logits.transpose(1,2), target_type)
        losses = {'loss_type':loss_bce}

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'points': self.loss_points,
            'kld': self.loss_kld
        }
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)


    def get_edge_loss(self, loss, outputs, targets, edge_indices, num_points, **kwargs):
        loss_map = {
            'edge_target': self.loss_pair_target,
            'edge_type': self.loss_pair_type
        }

        return loss_map[loss](outputs, targets, edge_indices, num_points, **kwargs)

    "the indices shape : (bs,2,match_length)"
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])

        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx

    def forward(self, outputs, targets, means, logvars):
        """
        outputs:
        'pred_logits: the label of the nodes'  'pred_points: the coordinate of the nodes'    #boxes' label & pos
        ETFR(outputs:
        'pred_idx1: node1, 'pred_idx2': node2, 'pred_type': the edge label

        targets:
        'points': (num_nodes, 3)
        'labels': (num_nodes)  the room label array
        """

        indices = self.matcher(outputs, targets)
        edge_indices, trf_targets = self.TRF_matcher(outputs, targets, indices)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float,
                                     device=next(iter(outputs.values())).device)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, means=means, logvars=logvars))

        for loss in self.edge_losses:
            losses.update(self.get_edge_loss(loss, outputs, trf_targets, edge_indices, num_points))
        return losses

if __name__ == '__main__':
    pass







