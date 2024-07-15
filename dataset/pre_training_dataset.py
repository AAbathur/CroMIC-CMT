import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64
from io import BytesIO
import lmdb
import time
import PIL

class EncodedTextLmdbImage(Dataset):
    def __init__(self, text_path, image_path):
        super().__init__()
        text_dict = torch.load(text_path)
        
        self.all_did = text_dict["all_did"]
        self.id2post = text_dict["did2post"]
        self.id2cmt = text_dict["did2cmts"]
        self.image_path = image_path
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        self.img_env = lmdb.open(self.image_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    def __len__(self):
        return len(self.all_did)
    
    def get_raw_image(self, did):
      
        with self.img_env.begin(write=False) as txn:
            value = txn.get(did.encode())
        image = Image.open(BytesIO(value))
        image = image.convert('RGB')
        return self.transform(image)
    
    def __getitem__(self, index):
        did = self.all_did[index]
        post = self.id2post[did]
        cmts = self.id2cmt[did]
        img = self.get_raw_image(did)
        return did, post, img, cmts
    
    def collate_fn(self, batch):
        batch_did = []
        batch_post = []
        batch_img = []
        batch_cmts = []
        for i in range(len(batch)):
            data = batch[i]
            batch_did.append(data[0])
            batch_post.append(data[1])
            batch_img.append(data[2])
            batch_cmts.append(data[3])
        batch_post = torch.tensor(batch_post)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_did": batch_did, "batch_post":batch_post, "batch_image":batch_img, "batch_cmts": batch_cmts} 
        
def lazily_data_load_lmdb_image(fnames, text_folder, image_folder, batch_size, num_worker, shuffle, epoch, distributed=False, world_size=-1):
    for i, fname in enumerate(fnames):
        text_path = os.path.join(text_folder, fname+".pt")
        image_path = os.path.join(image_folder, fname+"_image_lmdb")
        ds = EncodedTextLmdbImage(text_path, image_path)
        
        if distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size)
            dist_sampler.set_epoch(epoch*len(fnames)+i)
            
            data_loader = DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=num_worker, pin_memory=True, sampler=dist_samper, collate_fn=ds.collate_fn)
        
        else:
            data_loader = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, pin_memory=True, num_workers=num_worker, collate_fn=ds.collate_fn)
        yield data_loader
            

