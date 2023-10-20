import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
# from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    weight_dict = criterion.weight_dict
    if epoch <= 10:
        weight_dict['loss_kld'] = epoch * (1e-10)
    elif epoch <= 20:
        weight_dict['loss_kld'] = 10*(1e-10) + (epoch-10) * (1e-8)
    elif epoch <= 30:
        weight_dict['loss_kld'] = 10*(1e-10) + 10 * (1e-8) +(epoch-20)*(1e-6)
    elif epoch <= 70:
        weight_dict['loss_kld'] = 10*(1e-10) + 10*(1e-8) + 10 * (1e-6) + (epoch-30)*(1e-5)
    elif epoch<= 90 :
        weight_dict['loss_kld'] = 10*(1e-10) + 10*(1e-8) + 10*(1e-6) + 40*(1e-5) +(epoch-70)*(4e-5)
    else:
        weight_dict['loss_kld'] = 10 * (1e-10) + 10 * (1e-8) + 10 * (1e-6) + (70 - 30) * (1e-5) + 20 * (4e-5) + (epoch - 90)*(8e-5)

    print(f"Weight_dict in epoch:{epoch} is {weight_dict}")

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples_composite=torch.zeros((4,8,120,120))
        samples_composite[0]=samples[0]
        samples_composite[1]=samples[1]
        samples_composite[2]=samples[2]
        samples_composite[3]=samples[3]
        samples=samples_composite
        samples = samples.to(device)


        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        outputs, means, logvars = model(samples)

        loss_dict = criterion(outputs, targets, means, logvars)    #Loss in the criterion

        weights_detr = ['loss_ce', 'loss_points', 'loss_kld']
        weights_edge = ['loss_idx1', 'loss_idx2', 'loss_type']

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_point = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weights_detr)
        losses_edge = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weights_edge)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_unscaled['detr_unscaled'] = loss_dict_reduced_unscaled['loss_ce_unscaled'] + \
                                                      loss_dict_reduced_unscaled['loss_points_unscaled']

        loss_dict_reduced_unscaled['edge_unscaled'] = loss_dict_reduced_unscaled['loss_idx1_unscaled'] + \
                                                      loss_dict_reduced_unscaled['loss_idx2_unscaled'] + \
                                                      loss_dict_reduced_unscaled['loss_type_unscaled']
        loss_dict_reduced_unscaled['all_unscaled'] = loss_dict_reduced_unscaled['detr_unscaled'] + \
                                                     loss_dict_reduced_unscaled['edge_unscaled']
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        if epoch>= 15:
            for name, param in model.named_parameters():
                if 'detr' in name:
                    param.requires_grad = False
            losses_edge.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if 'detr' in name:
                param.requires_grad = True
        losses_point.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    writer.add_scalar(tag='Total/all_reconstruction', scalar_value=metric_logger.__getattr__('all_unscaled').global_avg,
                      global_step=epoch)
    writer.add_scalar(tag='Total/detr_reconstruction', scalar_value=metric_logger.__getattr__('detr_unscaled').global_avg,
                      global_step=epoch)
    writer.add_scalar(tag='Total/edge_reconstruction', scalar_value=metric_logger.__getattr__('edge_unscaled').global_avg,
                      global_step=epoch)
    writer.add_scalar(tag='Total/kld', scalar_value=metric_logger.__getattr__('loss_kld_unscaled').global_avg,
                      global_step=epoch)

    writer.add_scalars('Detail/detr',
                       {'label': metric_logger.__getattr__('loss_ce_unscaled').global_avg,
                        'points': metric_logger.__getattr__('loss_points_unscaled').global_avg
                        }, epoch)
    writer.add_scalars('Detail/edge',
                       {'edge1': metric_logger.__getattr__('loss_idx1_unscaled').global_avg,
                        'edge2': metric_logger.__getattr__('loss_idx2_unscaled').global_avg,
                        'type': metric_logger.__getattr__('loss_type_unscaled').global_avg
                        }, epoch)
    writer.add_scalars('recon&kld',
                       {'reconstruction': metric_logger.__getattr__('all_unscaled').global_avg*100,
                        'kld': metric_logger.__getattr__('loss_kld_unscaled').global_avg
                        }, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# the model output of detr: out, src, pos[-1], hs, means, logvars
def train_one_epoch_detr(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    weight_dict = criterion.weight_dict
    if epoch <= 10:
        weight_dict['loss_kld'] = (epoch + 1) * 0.00000001
    elif epoch <= 20:
        weight_dict['loss_kld'] = (epoch + 1) * 0.0000001
    elif epoch <=40:
        weight_dict['loss_kld'] = (epoch + 1) * 0.00001
    else:
        weight_dict['loss_kld'] = 40 * 0.00001
    print(f"Weight_dict in epoch:{epoch} is {weight_dict}")

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples_composite = torch.zeros((4, 8, 120, 120))
        samples_composite[0] = samples[0]
        samples_composite[1] = samples[1]
        samples_composite[2] = samples[2]
        samples_composite[3] = samples[3]
        samples = samples_composite
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        "outputs & targets"

        outputs,_ ,_ ,_ , means, logvars = model(samples)
        """
        outputs:: pred_logits, pred_points, pred_idx1, pred_idx2, pred_type
        """

        loss_dict = criterion(outputs, targets, means, logvars)
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v    #the unscaled
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_unscaled['reconstruction_unscaled'] = loss_dict_reduced_unscaled['loss_ce_unscaled']+loss_dict_reduced_unscaled['loss_points_unscaled']
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples_composite = torch.zeros((4, 8, 120, 120))
        samples_composite[0] = samples[0]
        samples_composite[1] = samples[1]
        samples_composite[2] = samples[2]
        samples_composite[3] = samples[3]
        samples = samples_composite

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]   #to cuda

        outputs, means, logvars = model(samples)
        loss_dict = criterion(outputs, targets, means, logvars)    #loss_output
        weight_dict = criterion.weight_dict

        # we skip the reduce process
        loss_dict_reduced = loss_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_unscaled['reconstruction_unscaled'] = loss_dict_reduced_unscaled['loss_ce_unscaled'] + \
                                                                loss_dict_reduced_unscaled['loss_points_unscaled']
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats

def evaluate_detr(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples_composite = torch.zeros((4, 6, 120, 120))
        samples_composite[0] = samples[0]
        samples_composite[1] = samples[1]
        samples_composite[2] = samples[2]
        samples_composite[3] = samples[3]
        samples = samples_composite

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs,_ ,_ ,_ ,means, logvars = model(samples)
        loss_dict = criterion(outputs, targets, means, logvars)    #loss_output
        weight_dict = criterion.weight_dict

        # we skip the reduce process
        loss_dict_reduced = loss_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_unscaled['reconstruction_unscaled'] = loss_dict_reduced_unscaled['loss_ce_unscaled'] + \
                                                                loss_dict_reduced_unscaled['loss_points_unscaled']
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats