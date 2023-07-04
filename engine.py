# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The Model Training Wrapper
import numpy as np
import os
import bisect
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from utils import get_loss_func, get_mix_lambda, d_prime
import torch
import torch.optim as optim
import torch.distributed as dist
import pytorch_lightning as pl


class SEDWrapper(pl.LightningModule):
    def __init__(self, sed_model, config):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        # self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)

    def val_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"val_acc": acc}  
    
    def test_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"test_acc": acc} 
    
    def forward(self, x, mix_lambda = None):
        output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        mix_lambda = None

        pred, _ = self(batch["waveform"], mix_lambda)
        loss = self.loss_func(pred, batch["target"])
        self.log("train_loss", loss, on_epoch= True, prog_bar=True)

        # Calculate and log training accuracy
        preds = torch.argmax(pred, dim=1)
        target = torch.argmax(batch["target"], dim=1)
        acc = (preds == target).float().mean()
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
        
    # def training_epoch_end(self, outputs):
    #     # Change: SWA, deprecated
    #     # for opt in self.trainer.optimizers:
    #     #     if not type(opt) is SWA:
    #     #         continue
    #     #     opt.swap_swa_sgd()
    #     self.dataset.generate_queue()


    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
    
        # Calculate and log validation loss
        loss = self.loss_func(pred, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return [pred.detach(), batch["target"].detach()]
    
    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)

        # # ? 
        # metric_dict = {
        #     "val_acc":0.
        # }
        
        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        gather_target = np.argmax(gather_target, 1)
        
        metric_dict = self.val_evaluate_metric(gather_pred, gather_target)
        print(self.device_type, metric_dict, flush = True)
    
        self.log("val_acc", metric_dict["val_acc"], on_epoch = True, prog_bar=True, sync_dist=False)
            
        
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        # time shifting optimization
        shift_num = 1 # framewise localization cannot allow the time shifting
        for i in range(shift_num):
            pred, pred_map = self(batch["waveform"])
            preds.append(pred.unsqueeze(0))
            batch["waveform"] = self.time_shifting(batch["waveform"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)

        return [pred.detach(), batch["target"].detach()]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device

        pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
        dist.barrier()

        # metric_dict = {
        #     "test_acc":0.
        # }
        
        dist.all_gather(gather_pred, pred)
        dist.all_gather(gather_target, target)
        if dist.get_rank() == 0:
            gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
            gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
            gather_target = np.argmax(gather_target, 1)
            metric_dict = self.test_evaluate_metric(gather_pred, gather_target)
            print(self.device_type, dist.get_world_size(), metric_dict, flush = True)

        self.log("test_acc", metric_dict["test_acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
        dist.barrier()
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )
        # Change: SWA, deprecated
        # optimizer = SWA(optimizer, swa_start=10, swa_freq=5)
        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        
        return [optimizer], [scheduler]
