'''
 * RAM++ & RAM & Tag2Text finetune
 * Written by Xinyu Huang
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from ram.models import ram_plus, ram, tag2text
import utils
from utils import cosine_lr_schedule
from ram.data import create_dataset, create_sampler, create_loader

import clip

def build_text_embed(model_clip, caption):
    run_on_gpu = torch.cuda.is_available()
    with torch.no_grad():

        texts = clip.tokenize(caption,truncate = True)  # tokenize
        if run_on_gpu:
            texts = texts.cuda()
            model_clip = model_clip.cuda()
        text_embeddings = model_clip.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings



def train_ram_plus(model, data_loader, optimizer, epoch, device, config, model_clip):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_tag', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_dis', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_alignment', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image, image_224, caption, image_tag, parse_tag) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        batch_text_embed = build_text_embed(model_clip,caption)
        
        image = image.to(device,non_blocking=True)
        image_224 = image_224.to(device,non_blocking=True)

        clip_image_feature = model_clip.encode_image(image_224)

        loss_tag, loss_dis, loss_alignment = model(image, caption, image_tag, clip_image_feature, batch_text_embed)  
        loss = loss_tag + loss_dis + loss_alignment

        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_tag=loss_tag.item())
        metric_logger.update(loss_dis=loss_dis.item())
        metric_logger.update(loss_alignment=loss_alignment.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def train_ram(model, data_loader, optimizer, epoch, device, config, model_clip):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_t2t', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_tag', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_dis', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image, image_224, caption, image_tag, parse_tag) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)
        image_224 = image_224.to(device,non_blocking=True)

        clip_image_feature = model_clip.encode_image(image_224)

        loss_t2t, loss_tag, loss_dis = model(image, caption, image_tag, parse_tag, clip_image_feature)  
        loss = loss_t2t + loss_tag/(loss_tag/loss_t2t).detach() + loss_dis  

        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_t2t=loss_t2t.item())
        metric_logger.update(loss_tag=loss_tag.item())
        metric_logger.update(loss_dis=loss_dis.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def train_tag2text(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_t2t', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_tag', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image, _, caption, _, parse_tag) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)

        loss_t2t, loss_tag = model(image, caption, parse_tag)  
        loss = loss_t2t + loss_tag/(loss_tag/loss_t2t).detach()   

        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_t2t=loss_t2t.item())
        metric_logger.update(loss_tag=loss_tag.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('finetune', config, min_scale=0.2)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)         

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]      
    
    print("Creating model")
    if args.checkpoint:  
        print("load from:", args.checkpoint)

    #### Model #### 
    if args.model_type == 'ram_plus':
        print("Creating pretrained CLIP model")
        model_clip, _ = clip.load("ViT-B/16", device=device)

        print("Creating RAM model")
        model = ram_plus(pretrained = args.checkpoint,image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                vit_ckpt_layer=config['vit_ckpt_layer'])

    elif args.model_type == 'ram':
        print("Creating pretrained CLIP model")
        model_clip, _ = clip.load("ViT-B/16", device=device)
        
        print("Creating RAM model")
        model = ram(pretrained = args.checkpoint,image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                vit_ckpt_layer=config['vit_ckpt_layer'])

    elif args.model_type == 'tag2text':
        print("Creating Tag2Text model")
        model = tag2text(pretrained = args.checkpoint,image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                vit_ckpt_layer=config['vit_ckpt_layer'], tag_list='ram/data/ram_tag_list.txt')
    model = model.to(device)   
    
    ### Frozen label embedding for open-set recogniztion ###
    model.label_embed.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        if args.model_type == 'ram_plus':
            train_stats = train_ram_plus(model, data_loader, optimizer, epoch, device, config, model_clip) 
        elif args.model_type == 'ram':
            train_stats = train_ram(model, data_loader, optimizer, epoch, device, config, model_clip) 
        elif args.model_type == 'tag2text':
            train_stats = train_tag2text(model, data_loader, optimizer, epoch, device, config) 

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument("--model-type",type=str,choices=("ram_plus", "ram", "tag2text"),required=True)
    parser.add_argument('--output-dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)