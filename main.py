import argparse
import datetime
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from models import build_BubbleFormer
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('Set Transfloormer', add_help=False)
    parser.add_argument('--model_name', default='bin_transf_rbf_5.55')
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--lr_backbone', default=1.5e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=2, type=int)
    parser.add_argument('--room_class',default=6,type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--cnet', default='resnet18', type=str,
                        help="Name of the cnet to use")
    parser.add_argument('--enet', default='resnet18', type=str,
                        help="Name of the enet to use")

    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pos_code_len_node', default=96, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pos_code_len_edge', default=96, type=int,
                        help="Number of decoding layers in the transformer")
    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=192, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--hidden_dim_edge', default=192, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=8, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--pretrained_backbone',default=False)

    # * Edge Transformer
    parser.add_argument('--edge_pos_shape', default=[4, 192, 35, 35])
    parser.add_argument('--num_edge_types', default=1, type=int)
    parser.add_argument('--num_edge_queries', default=16, type=int)
    parser.add_argument('--edge_nheads', default=4, type=int,)
    parser.add_argument('--edge_dim_feedforward', default=1024, type=int)
    parser.add_argument('--edge_enc_layers', default=1, type=int)
    parser.add_argument('--edge_dec_layers', default=6)



    parser.add_argument('--share_enc', default=0, type=int)
    parser.add_argument('--pretrained_dec', default=0, type=int)
    parser.add_argument('--temperature', default=0.2, type=float)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', default=False, action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher cost
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=10, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # edge Matcher
    parser.add_argument('--set_cost_type', default=1, type=float)
    parser.add_argument('--set_cost_target_node', default=1, type=float)

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--kld_loss_coef', default=0, type=float)

    # loss coef for edge
    parser.add_argument('--edge_target_loss_coef', default=1, type=float)
    parser.add_argument('--edge_type_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='./logs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True,action='store_true')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--data_pth',default="F:/sjh_study/gs_next_works/RPLAN_data_compact/")
    #datapath


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #process channels of the GT
    parser.add_argument('--GT_e_channels', default=[3,12,32])

    #Tbackbone
    parser.add_argument('--Tbackbone_strides', default=[1, 1])  # the stride of tbackbone
    parser.add_argument('--Tbackbone_input_channels', default=160)
    parser.add_argument('--Tbackbone_inplanes', default=160)
    parser.add_argument('--Tbackbone_channels', default=[160,160,192,192])

    parser.add_argument('--bin_backbone_strides', default=[1, 1])  # the stride of tbackbone
    parser.add_argument('--bin_backbone_input_channels', default=128)
    parser.add_argument('--bin_backbone_inplanes', default=128)
    parser.add_argument('--bin_backbone_channels', default=[128, 128, 192, 192])

    #Vae_control
    parser.add_argument('--vae_triger', default=1,type=int,
                        help='use vae mode or not')
    "cnet"
    parser.add_argument('--c_backbone', default='resnet18', type=str)
    parser.add_argument('--c_train_backbone', default=1, type=int)
    parser.add_argument('--c_dilation', default=1, type=int)
    parser.add_argument('--c_pretrained', default=0, type=int)
    parser.add_argument('--c_channels', default=[64,128,128,128], type=int)
    parser.add_argument('--c_input_channels', default=3, type=int)
    "enet"
    parser.add_argument('--e_backbone', default='resnet18', type=str)
    parser.add_argument('--e_train_backbone', default=1, type=int)
    parser.add_argument('--e_dilation', default=1, type=int)
    parser.add_argument('--e_pretrained', default=0, type=int)
    parser.add_argument('--e_inplanes', default=32, type=int)
    parser.add_argument('--e_channels', default=[32, 32, 32, 32], type=int)
    parser.add_argument('--e_input_channels', default=32, type=int)
    parser.add_argument('--e_strides', default=[1,1])
    parser.add_argument('--noise_channels', default=32)

    return parser
def main(args):

    device = torch.device(args.device)
    "Seed for reproduce "
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion = build_BubbleFormer(args)

    model.to(device)

    "model parameters"
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    "Optimizer"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop,0.8)

    "Dataset set up, packaging dataloader etc."
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_train=torch.utils.data.RandomSampler(dataset_train)
    sampler_val  =torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train=torch.utils.data.BatchSampler(
        sampler_train,args.batch_size,drop_last=True
    )

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_new,num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=True,collate_fn=utils.collate_fn_new, num_workers=args.num_workers)
    train_time = time.strftime('%m%d_%H:%M')

    # output_dir=Path(args.output_dir +f'{args.model_name}_{train_time}/')
    output_dir=Path(args.output_dir)



    if os.path.exists(output_dir):
        print("path already exist")
    else:
        os.mkdir(output_dir)
    writer = SummaryWriter(log_dir=output_dir, flush_secs=60)


    "Starting_Training"
    start_time=time.time()
    start_epoch = 0
    "Insert training"
    insert_training = 0
    if insert_training:
        model.load_model('./.pth')
        start_epoch = insert_training + 1
    for epoch in range(start_epoch, args.epoch):
        if epoch%30 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

        train_stats=train_one_epoch(model,criterion,data_loader_train,optimizer,device,epoch, writer, args.clip_max_norm)
        lr_scheduler.step()
        checkpoint_paths=[]
        checkpoint_paths.append(output_dir / f'{model.name}{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            torch.save(
                {
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict(),
                    'epoch':epoch,
                    'args':args
                },
                checkpoint_path
            )
        pth_list = [pth for pth in os.listdir(output_dir) if re.match(model.name, pth)]
        pth_list = sorted(pth_list, key=lambda x: os.path.getmtime(os.path.join(output_dir,x)))
        if len(pth_list)>=25 and pth_list is not None:
            to_delete = str(output_dir) + '/' + pth_list[0]
            if os.path.exists(to_delete):
                os.remove(to_delete)

        test_stats=evaluate(model, criterion, data_loader_val, device, output_dir)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            with (output_dir/'log.txt').open('a') as f:
                f.write(json.dumps(log_stats)+"\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transfloormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()


    "Training code"
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(args)





