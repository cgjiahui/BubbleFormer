from collections import OrderedDict
import torch
from .TRF_Criterion import Set_TRF_Criterion
from .transformer import build_transformer,build_edge_transformer
from .backbone import build_backbone, build_backbone_etfr, build_position_encoding_shape
from .vae import build_vae
import torch.nn.functional as F
from torch import nn
from .matcher import build_matcher, build_edge_matcher
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .edge_tfr import Construct_BubbleFormer
from .vae import Cnet
def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)

class TFR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)   #Batch,c,h,w
        src_key_padding_mask=None
        src = features[-1]

        hs = self.transformer(self.input_proj(src), src_key_padding_mask, self.query_embed.weight, pos[-1])[0]


        outputs_class = self.class_embed(hs)   #class embedding layer
        outputs_coord = self.point_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
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

class vae_TFR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,name, cnet, enet, backbone, transformer, num_classes, num_queries, aux_loss=False, GT_e_channels=[3,6,11]):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        """
        Updates:
        Increase the number channels of input noise
        """
        super().__init__()
        self.name = name
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        "vae"
        self.cnet=cnet
        self.enet=enet

        self.in_enet=nn.Sequential(
            nn.Conv2d(GT_e_channels[0],GT_e_channels[1],3,2,padding=1),
            nn.Conv2d(GT_e_channels[1],GT_e_channels[2],3,2,padding=1)
        )

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        samples_input=samples[:,:-3,:,:]
        samples_gt=samples[:,-3:,:,:]
        c_fearure=self.cnet(samples_input)['0']
        gt_feature=self.in_enet(samples_gt)

        "Enet: encode feature and gt"

        "Enet: encode gt"
        means, logvars = self.enet(gt_feature)

        z=reparameterize(means,logvars)

        "reconstruction"
        features, pos = self.backbone(torch.cat([c_fearure,z],1))
        src_key_padding_mask=None    #
        src = features[-1]


        hs = self.transformer(self.input_proj(src), src_key_padding_mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.point_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, src, pos[-1], hs, means, logvars

    def inference(self, condition_input, z):
        c_feature=self.cnet(condition_input)['0']
        features,pos = self.backbone(torch.cat([c_feature,z],1).to(torch.float32))
        src_key_padding_mask = None

        src=features[-1]
        hs = self.transformer(self.input_proj(src), src_key_padding_mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)  # class embedding layer
        outputs_coord = self.point_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
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

class NodeFormer(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,name, cnet, enet, backbone, transformer, num_classes, num_queries, aux_loss=False, GT_e_channels=[3,6,11]):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        """
        Updates:
        Increase the number channels of input noise
        """
        super().__init__()
        self.name = name
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)   #transfer the input dim 2 the in_transformer dim
        self.backbone = backbone
        self.aux_loss = aux_loss

        "vae"
        self.cnet=cnet
        self.enet=enet

        self.in_enet=nn.Sequential(
            nn.Conv2d(GT_e_channels[0],GT_e_channels[1],3,2,padding=1),
            nn.Conv2d(GT_e_channels[1],GT_e_channels[2],3,2,padding=1)
        )
    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        samples_input=samples[:,:-5,:,:]
        samples_gt=samples[:,-3:,:,:]
        c_fearure=self.cnet(samples_input)['0']
        gt_feature=self.in_enet(samples_gt)

        "Enet: encode gt"
        means, logvars = self.enet(gt_feature)  # 30*30 32

        z=reparameterize(means,logvars)

        features, pos = self.backbone(torch.cat([c_fearure,z],1))
        src_key_padding_mask=None
        src = features[-1]

        hs = self.transformer(self.input_proj(src), src_key_padding_mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.point_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)


        return out, samples_input, hs, means, logvars


    def inference(self, condition_input, z):
        c_feature=self.cnet(condition_input)['0']
        features,pos = self.backbone(torch.cat([c_feature,z],1).to(torch.float32))
        src_key_padding_mask = None

        src=features[-1]
        hs = self.transformer(self.input_proj(src), src_key_padding_mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)  # class embedding layer
        outputs_coord = self.point_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            weight_dict: {loss_ce  loss_points  loss_kld}
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty points
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
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

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            # "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            # "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_kld(self, outputs, targets, indices, num_points, means, logvars, **kwargs):
        losses={}
        kld_loss = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp())
        losses['loss_kld']=kld_loss
        return losses

    def _get_src_permutation_idx(self, indices):
        "indices"
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])

        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices，
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'points': self.loss_points,
            'masks': self.loss_masks,
            'kld': self.loss_kld
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, means, logvars):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        "indices"

        # Compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)


        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, means=means, logvars=logvars))


        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs):    #(batch_size,hw)
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        points=out_bbox*120
        results = [{'scores': s, 'labels': l, 'points': b} for s, l, b in zip(scores, labels, points)]
        return results


def build(args):
    num_classes = args.target_class
    device = torch.device(args.device)

    backbone = build_backbone(args)
    cnet,enet = build_vae(args)
    transformer = build_transformer(args)

    if args.vae_triger:
        model=vae_TFR(
            args.model_name,
            cnet,
            enet,
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            GT_e_channels=args.GT_e_channels
        )
    else:
        model = TFR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
        )


    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_points': args.bbox_loss_coef, 'loss_kld':args.kld_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'points', 'cardinality', 'kld']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion





def build_BubbleFormer_all(args):

    num_classes = args.room_class
    backbone = build_backbone(args)
    cnet, enet = build_vae(args)
    node_transformer = build_transformer(args)

    model = NodeFormer(
        args.model_name,
        cnet,
        enet,
        backbone,
        node_transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        GT_e_channels=args.GT_e_channels
    )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': 1, 'loss_points': args.bbox_loss_coef, 'loss_kld': args.kld_loss_coef}
    losses = ['labels', 'points', 'cardinality', 'kld']

    "Edges Part"
    edge_matcher = build_edge_matcher(args)
    edge_losses = ['edge_target', 'edge_type']

    "edge loss weight"
    weight_dict['loss_idx1'] = args.edge_target_loss_coef
    weight_dict['loss_idx2'] = args.edge_target_loss_coef
    weight_dict['loss_type'] = args.edge_type_loss_coef

    criterion = Set_TRF_Criterion(args.room_class, matcher=matcher, weight_dict=weight_dict,
                                  losses=losses,
                                  edge_losses=edge_losses, edge_matcher=edge_matcher, args=args)

    edge_transformer = build_edge_transformer(args)
    backbone_edge_transfomrmer = build_backbone_etfr(args)

    cnet_edge_transformer = Cnet(args.c_backbone, args.c_train_backbone, args.c_dilation, args.c_pretrained, args.c_channels,
                     args.c_input_channels)
    kwargs = {}


    BubbleFormer_all = Construct_BubbleFormer(
        name=args.model_name,
        detr=model,
        num_edge_queries=args.num_edge_queries,
        num_types=args.num_edge_types,
        edge_cnet=cnet_edge_transformer,
        edge_backbone=backbone_edge_transfomrmer,
        edge_transformer=edge_transformer,
        freeze_detr=(args.frozen_weights is not None),
        share_enc=args.share_enc,
        pretrained_dec=args.pretrained_dec,
        temperature=args.temperature,
        **kwargs
    )

    return BubbleFormer_all, criterion

