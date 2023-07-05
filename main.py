# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The main code for training and evaluating HTSAT
import os

import numpy as np
import argparse

import logging
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import create_folder, dump_config

import config
from engine import SEDWrapper

from dataset.dataset import JY_Dataset
# from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset


from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings



def train():
    # set exp folder
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)
    
    device_num = torch.cuda.device_count()
    # print("each batch size:", config.batch_size // device_num)

    train_dataset = JY_Dataset(
        dataset = np.load(os.path.join(config.dataset_path, "train.npy"), allow_pickle = True),
        config = config
    )
    val_dataset = JY_Dataset(
        dataset = np.load(os.path.join(config.dataset_path, "val.npy"), allow_pickle = True),
        config = config
    )
    
    train_dataloader = DataLoader(
        dataset = train_dataset,
        num_workers = config.num_workers,
        batch_size = config.batch_size,
        shuffle = True
    )
    
    val_dataloader = DataLoader(
        dataset = val_dataset,
        num_workers = config.num_workers,
        batch_size = config.batch_size,
        shuffle = False
    )

    # audioset_data = data_prep(train_dataset, val_dataset, device_num)
    
    # checkpoint_callback = ModelCheckpoint(
    #     monitor = "acc",
    #     filename='l-{epoch:d}-{acc:.3f}',
    #     save_top_k = 20,
    #     mode = "max"
    # )
    checkpoint_callback = ModelCheckpoint(
    monitor = "val_loss",
    filename='l-{epoch:d}-{val_loss:.5f}-{val_acc:.3f}',
    save_top_k = 10,
    mode = "min"
    )
    
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir = checkpoint_dir,
        devices = config.device, 
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "gpu",
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        # finetune on the esc and spv2 dataset
        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(config.resume_checkpoint + " loaded!")
    # elif config.swin_pretrain_path is not None: # train with pretrained model
    #     ckpt = torch.load(config.swin_pretrain_path, map_location="cpu")
    #     # load pretrain model
    #     ckpt = ckpt["model"]
    #     found_parameters = []
    #     unfound_parameters = []
    #     model_params = dict(model.state_dict())

    #     for key in model_params:
    #         m_key = key.replace("sed_model.", "")
    #         if m_key in ckpt:
    #             if m_key == "patch_embed.proj.weight":
    #                 ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
    #             if m_key == "head.weight" or m_key == "head.bias":
    #                 ckpt.pop(m_key)
    #                 unfound_parameters.append(key)
    #                 continue
    #             assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
    #             found_parameters.append(key)
    #             ckpt[key] = ckpt.pop(m_key)
    #         else:
    #             unfound_parameters.append(key)
    #     print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
    #     print("unfound parameters: ", unfound_parameters)
    #     model.load_state_dict(ckpt, strict = False)
    #     model_params = dict(model.named_parameters())
    trainer.fit(model, train_dataloaders= train_dataloader, val_dataloaders = val_dataloader)


def test():
    device_num = torch.cuda.device_count()
    # print("each batch size:", config.batch_size // device_num)

    test_dataset = JY_Dataset(
        dataset = np.load(os.path.join(config.dataset_path, "test.npy"), allow_pickle = True),
        config = config
    )
    
    test_dataloader = DataLoader(
        dataset = test_dataset,
        num_workers=config.num_workers,
        batch_size = config.batch_size,
    )
        
    # audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    
    trainer = pl.Trainer(
        deterministic=False,
        devices = config.device, 
        # max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "gpu" ,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config
    )
    
    if config.test_checkpoint is not None:
        ckpt = torch.load(config.test_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(config.test_checkpoint + " loaded!")
    else :
        print("no test_checkpoint")
    trainer.test(model, test_dataloaders= test_dataloader)


def main():
    parser = argparse.ArgumentParser(description="HTS-AT")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        raise Exception("Error Mode!")
    

if __name__ == '__main__':
    main()

