# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified By ShengYin For Adversarial Training

# CUDA_VISIBLE_DEVICES=2 python train_AT.py --hypes_yaml logs/dair_centerpoint_multiscale_att_4ATA_from_0-1018-2/config.yaml --model_dir logs/dair_centerpoint_multiscale_att_4ATA_from_0-1018-2 --fusion_method intermediate --attack config/attack/single_agent/erase_and_shift_and_pgd/AT_attack.yaml  --attack_type erase_and_shift_and_pgd

import argparse
import os
import statistics
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic
from omegaconf import OmegaConf

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')

    # attack
    parser.add_argument('--attack_type', type=str, default='None', help='Attack mode: [pgd, shift]')
    parser.add_argument('--attack', type=str, default=False, help="Attack config file, "
                            "if it is \"TRUE\", following attack hyperparameters will be used")
    parser.add_argument('--save_path', type=str, default=False, help="Saving the model")
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=True)
    validation_index = np.load('validation_index.npy')
    train_index = np.load('train_index.npy')

    train_loader = DataLoader(Subset(opencood_train_dataset, train_index),
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(Subset(opencood_validate_dataset, validation_index),
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    # import ipdb; ipdb.set_trace()
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    if opt.attack != False:
        attack = opt.attack
        attack_conf = OmegaConf.load(opt.attack)
        attack_target = attack_conf.attack.attack_target
        attack_type = opt.attack_type
    else:
        attack = False
        attack_target = 'pred'
        attack_type = 'pgd'

    for epoch in range(init_epoch, max(epoches, init_epoch)):

        # model = train_utils.load_saved_model_with_epoch(saved_path, model, epoch)

        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            
            # generate simple_idx list
            num_list = batch_data['ego']['sample_idx_list']

            ouput_dict = model(batch_data['ego'], dataset = opencood_train_dataset, num = num_list, attack=attack, attack_target=attack_target, attack_type = attack_type)
            
            # import ipdb; ipdb.set_trace()
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            # with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                if batch_data is None:
                    continue
                model.zero_grad()
                optimizer.zero_grad()
                model.eval()

                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch

                num_list = batch_data['ego']['sample_idx_list']
                # import ipdb; ipdb.set_trace()

                ouput_dict = model(batch_data['ego'], dataset = opencood_validate_dataset)

                final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            if opt.save_path != False:
                saved_path = opt.save_path
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step()

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
