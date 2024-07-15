import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import base64
from io import BytesIO
from sklearn.utils import shuffle
from collections import OrderedDict
import time

class FMIQADataset(Dataset):
    ## FMIQA dataset
    """build a dataset based on encoded text in .pt file and raw image folder
    """
    def __init__(self, mode):
        super().__init__()
        self.base_img_folder = "data/downstream_data/VQA_images"
        if mode == "train":
            self.text_path = "data/downstream_data/FMIQA/FMIQA_example.pt"
        elif mode == "val":
            self.text_path = "data/downstream_data/FMIQA/FMIQA_example.pt"
        elif mode == "test":
            self.text_path = "data/downstream_data/FMIQA/FMIQA_example.pt"
            
        self.mode = mode
        text_dict = torch.load(self.text_path)
        self.all_qid = text_dict["all_qid"]
        self.qid2ques = text_dict["qid2ques"]
        self.qid2vid = text_dict["qid2vid"]
        self.qid2ans = text_dict["qid2ans"]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.all_qid)

    def get_raw_image(self, qid):
        vid = self.qid2vid[qid]
        img_path1 = "{}/COCO_{}2014_{:012}.jpg".format(self.base_img_folder, "train", int(vid))
        img_path2 = "{}/COCO_{}2014_{:012}.jpg".format(self.base_img_folder, "val", int(vid))
        if os.path.exists(img_path1):
            img_path = img_path1
        else:
            img_path = img_path2
        
        image = Image.open(img_path)
        image = image.convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        qid = self.all_qid[index]
        ques = self.qid2ques[qid]
        ans = self.qid2ans[qid]
        img = self.get_raw_image(qid)
        return qid, ques, ans, img
    
    def collate_fn(self, batch):
        batch_qid = []
        batch_ques = []
        batch_img = []
        batch_ans = []
        for i in range(len(batch)):
            data = batch[i]
            batch_qid.append(data[0])
            batch_ques.append(data[1])
            batch_ans.append(data[2])
            batch_img.append(data[3])
        batch_ques = torch.tensor(batch_ques)
        batch_ans = torch.tensor(batch_ans)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_did": batch_qid, "batch_post": batch_ques, "batch_image": batch_img, "batch_ans": batch_ans}

class VQAv2Dataset(Dataset):
    def __init__(self, mode, token_vocab="bert"):
        super().__init__()
        
        self.base_img_folder = "data/downstream_data/VQA_images"
        if mode == "train":
            self.text_path = "data/downstream_data/VQA2.0/VQA2.0_example.pt"
 
        elif mode == "val":
            self.text_path = "data/downstream_data/VQA2.0/VQA2.0_example.pt"
            
        elif mode == "test":
            raise KeyError("test data set has not been provided yet")
        self.mode = mode
        text_dict = torch.load(self.text_path)
        self.all_qid = text_dict["all_qid"]
        self.qid2ques = text_dict["qid2ques"]
        self.qid2vid = text_dict["qid2vid"]
        self.qid2ans = text_dict["qid2ans"]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.all_qid)

    def get_raw_image(self, qid):
        vid = self.qid2vid[qid]
        img_path = "{}/COCO_{}2014_{:012}.jpg".format(self.base_img_folder, "train", int(vid))
        if not os.path.exists(img_path):
            img_path = "{}/COCO_{}2014_{:012}.jpg".format(self.base_img_folder, "val", int(vid))
        
        image = Image.open(img_path)
        image = image.convert("RGB")
        return self.transform(image)
    
    def __getitem__(self, index):
        qid = self.all_qid[index]
        ques = self.qid2ques[qid]
        ans_idxs = self.qid2ans[qid]
        ans_label = [0] * 3129
        for idx in ans_idxs:
            ans_label[idx] += 1
        img = self.get_raw_image(qid)
        return qid, ques, ans_label, img
    
    def collate_fn(self, batch):
        batch_qid = []
        batch_ques = []
        batch_img = []
        batch_ans_label = []
        for i in range(len(batch)):
            data = batch[i]
            batch_qid.append(data[0])
            batch_ques.append(data[1])
            batch_ans_label.append(data[2])
            batch_img.append(data[3])
        batch_ques = torch.tensor(batch_ques)
        batch_ans_label = torch.tensor(batch_ans_label)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_did": batch_qid, "batch_post": batch_ques, "batch_image": batch_img, "batch_ans_label": batch_ans_label}
    

class EcomICDataset(Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        if mode == "train":  
            ic_path = "pre-processed/.pt/data/path"
            img_path = "tsv/file/released/by/the/official" ## tsv file released by the official
            multi_caption = True
        elif mode == "val":
            ic_path = "pre-processed/.pt/data/path"
            img_path = "tsv/file/released/by/the/official"
            multi_caption = True
        elif mode == "test":
            ic_path = "pre-processed/.pt/data/path"
            img_path = "tsv/file/released/by/the/official"
            multi_caption = False
        self.vid2captions = torch.load(ic_path)
        self.df = pd.read_csv(img_path, header=None)
        self.multi_caption = multi_caption
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        if self.multi_caption:
            return len(self.df) * 10 ## ten captions - one image
        return len(self.df)
    
    def get_raw_image(self, index):
        line = self.df.loc[index].values
        vid, image_base64 = line[0].split('\t')
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
        image = image.convert("RGB")
        return vid, self.transform(image)

    def __getitem__(self, index):
        
        if self.multi_caption:
            real_index = index // 10
            intra_index = index % 10
        else:
            real_index = index
            intra_index = None
        vid, img = self.get_raw_image(real_index)
    
        all_caption = self.vid2captions[vid]
        
        if intra_index == None:
            return vid, all_caption, img, all_caption
        else:
            caption = all_caption[intra_index]
            return vid, caption, img, all_caption

    def collate_fn(self, batch):
        batch_vid = []
        batch_img = []
        batch_cap = []
        batch_reference = []
        for i in range(len(batch)):
            data = batch[i]
            batch_vid.append(data[0])
            batch_cap.append(data[1])
            batch_img.append(data[2])
            batch_reference.append(data[3])
        
        batch_cap = torch.tensor(batch_cap)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_qid": batch_vid, "batch_post": batch_cap, "batch_image": batch_img, "batch_reference": batch_reference}

class Flickr30kCN_IC_Dataset(Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        self.image_folder = ""
        if mode == "train":  
            ic_path = "pre-processed/.pt/data/path"
        elif mode == "val":
            ic_path = "pre-processed/.pt/data/path"
        elif mode == "test":
            ic_path = "pre-processed/.pt/data/path"
        ret = torch.load(ic_path)
        self.all_did = ret["all_did"]
        self.id2cap = ret["id2cap"]
        del ret
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.all_did)
    
    def get_raw_image(self, vid):
        img_path = os.path.join(self.image_folder, vid+".jpg")
        image = Image.open(img_path)
        image = image.convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        did = self.all_did[index]
        cap = self.id2cap[did]
        if self.mode == "test":
            vid, _ = did.split("#zhm#")
            refs = [self.id2cap[vid+"#zhm#"+str(i)] for i in range(5)]
        else:
            vid, _ = did.split("#zhb#")
            refs = [self.id2cap[vid+"#zhb#"+str(i)] for i in range(5)]
        image = self.get_raw_image(vid)
        return did, cap, image, refs

    def collate_fn(self, batch):
        batch_qid = []
        batch_cap = []
        batch_img = []
        batch_reference = []
        for i in range(len(batch)):
            data = batch[i]
            batch_qid.append(data[0])
            batch_cap.append(data[1])
            batch_img.append(data[2])
            batch_reference.append(data[3])
        
        batch_cap = torch.tensor(batch_cap)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_qid": batch_qid, "batch_post": batch_cap, "batch_image": batch_img, "batch_reference": batch_reference}
    
class AIC_ICC_Dataset(Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        
        if mode == "train":
            self.image_folder = "data/downstream_data/AIC_ICC/images"
            ic_path = "data/downstream_data/AIC_ICC/AIC_ICC_example.pt"
        elif mode == "val":
            self.image_folder = "data/downstream_data/AIC_ICC/images"
            ic_path = "data/downstream_data/AIC_ICC/AIC_ICC_example.pt"
        elif mode == "test_a":
            self.image_folder = "data/downstream_data/AIC_ICC/images"
            ic_path = "data/downstream_data/AIC_ICC/AIC_ICC_example.pt"
        elif mode == "test_b":
            self.image_folder = "data/downstream_data/AIC_ICC/images"
            ic_path = "data/downstream_data/AIC_ICC/AIC_ICC_example.pt"
            
        ret = torch.load(ic_path)
        self.all_did = ret["all_did"]
        self.id2cap = ret["did2cap"]
        del ret
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return 5 * len(self.all_did)
    
    def get_raw_image(self, vid):
        img_path = os.path.join(self.image_folder, vid)
        image = Image.open(img_path)
        image = image.convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        real_index = index // 5
        intra_index = index % 5
        
        did = self.all_did[real_index]
        refs = self.id2cap[did]
        cap = refs[intra_index]
        image = self.get_raw_image(did)
        
        return did, cap, image, refs

    def collate_fn(self, batch):
        batch_qid = []
        batch_cap = []
        batch_img = []
        batch_reference = []
        for i in range(len(batch)):
            data = batch[i]
            batch_qid.append(data[0])
            batch_cap.append(data[1])
            batch_img.append(data[2])
            batch_reference.append(data[3])
        
        batch_cap = torch.tensor(batch_cap)
        batch_img = torch.stack(batch_img, dim=0)
        return {"batch_qid": batch_qid, "batch_post": batch_cap, "batch_image": batch_img, "batch_reference": batch_reference}



        
              
    
          
