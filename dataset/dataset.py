from torch.utils.data import Dataset
import logging
import numpy as np 
import random

class JY_Dataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        # self.eval_mode = eval_mode
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
    #     if not eval_mode:
    #         self.generate_queue()

    # def generate_queue(self):
    #     random.shuffle(self.queue)
    #     logging.info("queue regenerated:%s" %(self.queue[-5:]))

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]

        waveform  = self.dataset[p]["waveform"]
        while len(waveform) < self.config.clip_samples:
            waveform = np.concatenate((waveform, waveform))
        waveform = waveform[:self.config.clip_samples]

        target = np.zeros(self.config.classes_num).astype(np.float32)
        target[int(self.dataset[p]["target"])] = 1.
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": waveform,
            "real_len": len(waveform),
            "target": target
        }
        return data_dict

    def __len__(self):
        return self.total_size