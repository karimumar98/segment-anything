import sys
import torch
import clip
import util

sys.path.append("..")
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch import nn
from torchvision.ops import roi_pool, roi_align
import numpy as np
#import pytorch_lightning as pl
#from lightning.pytorch import Trainer

from typing import Dict, List, Optional, Tuple

# from detectron2.data.dataset_mapper import DatasetMapper
# from detectron2.data.build import build_detection_train_loader
import random

from open_clip.loss import ClipLoss

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

class DAAM_BASELINE (torch.nn.Module):
    
    ## Very crude abseline implementation, crops the image and extracts features using openAI's CLIP
    def __init__(self, 
            device = "cuda", 
            model_type = "vit_b", 
            sam_checkpoint = "sam_vit_b_01ec64.pth",
            temperature = 0.9):
        super().__init__()

        self.device = device

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)        
        ## Load the CLIP model
        self.clip, self.preprocess_clip = clip.load("ViT-B/32", device=device)
        
        self.coco_classes = util.get_coco_classes()
        self.coco_emb = np.load("coco_emb.npy")
        self.coco_emb = self.coco_emb / np.linalg.norm(self.coco_emb, axis=0)
        
    def preprocess_images_for_sam(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        This does all of the operations required to transform an image from detectron2 dataloader format to that, that is expected by SAM
        1: permute and resize longest edge to 1024
        2: convert to tensor and push to device
        3: permute and convert to batched input format
        4: Normalize pixel values and pad to a square input
        
        ** BBoxes will be resized seperately
        """
        
        numpy_images = [batch['image'].permute(1,2,0).cpu().numpy() for batch in batched_inputs]
        resized_longest_edge = [self.transform.apply_image(image) for image in numpy_images]
        torch_images = [torch.as_tensor(image, device=self.device) for image in resized_longest_edge]
        torch_images = [image.permute(2, 0, 1).contiguous() for image in torch_images]
        normalized = [self.sam.preprocess(image) for image in torch_images]
        
        ## Now stack together to form a batch
        return torch.stack(normalized, dim = 0)       
   

    def xyhw_toxyxy(self, xyxy):
        ## SAM outputs boxes in xyhw format, wheras roi_pool expects boxes to be in x1y1x2y2 format
        return [xyxy[0], xyxy[1], xyxy[2]+xyxy[0], xyxy[3]+xyxy[1]]  

    def forward (self, batched_inputs):     
        
        #images = [self.preprocess_clip(x["images"]) for x in batched_inputs]
        mask_generator = SamAutomaticMaskGenerator(self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        boxes = []
        masks = []
        num_boxes = []
        transform = T.ToPILImage()

        for item in batched_inputs:
            ## TODO: forward the whole batch in a single go
            pix = item['image'].permute(1,2,0).numpy()
            masks_ = mask_generator.generate(pix)

            for mask in masks_:
            #for i in range(0,5):
                bbox = mask["bbox"]
                crop_coords = util.xyhw_to_xyxy(bbox)
                
                ## Crop image to the bbox of the mask
                cropped_image = item['image'][:,crop_coords[1]:crop_coords[3],crop_coords[0]:crop_coords[2]]
                cropped_image.shape
                '''
                print(i)
                plt.figure(figsize=(8,8))
                plt.imshow(cropped_image.permute(1,2,0))
                plt.show()
                '''

                pil_img = transform(cropped_image)
                clip_input = self.preprocess_clip(pil_img).unsqueeze(0).to("cuda")
                clip_features = self.clip.encode_image(clip_input).squeeze().cpu().detach()
                clip_features = clip_features / np.linalg.norm(clip_features, axis=0)
                mask["mask_features"] = clip_features

                scores = np.dot(mask["mask_features"], self.coco_emb.T)
                s = [x for x in zip(scores, self.coco_classes)]
                s.sort(reverse = True)
                mask["top_5"] = s[:5]
                mask["top_1"] = s[0]
                
            masks += [masks_]
            boxes += [torch.tensor([self.xyhw_toxyxy(box["bbox"]) for box in masks_]).to('cuda', dtype=torch.float)]
        return masks
#        return boxes, predicted, masks
        


## Detect (almost) Anything Model
class DAAM (torch.nn.Module):

    
    def __init__(self, device = "cuda", model_type = "vit_b", sam_checkpoint = "sam_vit_b_01ec64.pth", input_resolution = (7,7)):
        super().__init__()

        self.device = device

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 91),
        ).to(self.device)
        
        self.train = False


    def preprocess_images(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        This does all of the operations required to transform an image from detectron2 dataloader format to that, that is expected by SAM
        1: permute and resize longest edge to 1024
        2: convert to tensor and push to device
        3: permute and convert to batched input format
        4: Normalize pixel values and pad to a square input
        
        ** BBoxes will be resized seperately
        """
        
        numpy_images = [batch['image'].permute(1,2,0).cpu().numpy() for batch in batched_inputs]
        resized_longest_edge = [self.transform.apply_image(image) for image in numpy_images]
        torch_images = [torch.as_tensor(image, device=self.device) for image in resized_longest_edge]
        torch_images = [image.permute(2, 0, 1).contiguous() for image in torch_images]
        normalized = [self.sam.preprocess(image) for image in torch_images]
        
        ## Now stack together to form a batch
        return torch.stack(normalized, dim = 0)        

    def xyhw_toxyxy(self, xyxy):
        ## SAM outputs boxes in xyhw format, wheras roi_pool expects boxes to be in x1y1x2y2 format
        return [xyxy[0], xyxy[1], xyxy[2]+xyxy[0], xyxy[3]+xyxy[1]]  
        
    def forward (self, batched_inputs):
        
        
        ## Resize the detectron2 imagesizes and normalize
        input_images = self.preprocess_images(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.sam.image_encoder(input_images)
        scale = 64.0/1024.0

        if self.train:
            
            bboxes = [x.gt_boxes for x in gt_instances]
            b = [[x.to('cpu').numpy() for x in bbox.__iter__()] for bbox in bboxes]
            classes = [x.gt_classes for x in gt_instances]

            ## Resize the detectron2 bounding boxes to the input size of SAM
            original_sizes = [(x['image'].shape[1], x['image'].shape[2]) for x in batched_inputs]
            resized_bboxes = [self.transform.apply_boxes(np.array(boxes), hw) for boxes, hw in zip(b, original_sizes)]


            boxes = [torch.tensor(x).to('cuda', dtype=torch.float) for x in resized_bboxes]



            target = torch.cat(classes)
            x = roi_align (features, boxes, output_size=(7,7), spatial_scale = scale)        
            probs = self.classifier(x.view(x.shape[0], -1))
            return probs
        else:
            
            mask_generator = SamAutomaticMaskGenerator(self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
                                                      
            boxes = []
            masks = []
            num_boxes = []
            for item in batched_inputs:
                ## TODO: forward the whole batch in a single go
                pix = item['image'].permute(1,2,0).numpy()
                mask = mask_generator.generate(pix)
                masks += [mask]
                boxes += [torch.tensor([self.xyhw_toxyxy(box["bbox"]) for box in mask]).to('cuda', dtype=torch.float)]
            x = roi_pool (features, boxes, output_size=(7,7), spatial_scale = scale)  
            predicted = self.classifier(x.view(x.shape[0], -1))
            
            return boxes, predicted, masks
            

            
            
        return probs

class DAAM_CLIP_ALIGNED (torch.nn.Module):
    def __init__(self, 
            device = "cuda", 
            model_type = "vit_b", 
            sam_checkpoint = "sam_vit_b_01ec64.pth",
            temperature = 0.9,
            input_channels = 256,
            input_resolution = (7,7),
            embeding_dim = 512):
        super().__init__()

        self.device = device

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)


        self.sam.to(device=device)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        self.temperature = nn.Parameter(torch.tensor(1.))

        self.mlp = nn.Sequential(
            nn.Linear(input_channels * input_resolution[0] * input_resolution[1], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, embeding_dim),
        ).to(self.device)

        
        self.train = False

        # self.class_embeddings = np.load("coco_emb.npy")
        # self.class_embeddings = torch.Tensor(self.class_embeddings)#.to("cuda", dtype=torch.float)
        # self.class_embeddings = self.class_embeddings.to(self.device, dtype=torch.float)

        ## Load the CLIP model to encode refs
        self.clip, preprocess = clip.load("ViT-B/32", device=device)
        # Get rid of visual part ## TODO: Create a custom model that avoids loading visual part in the first place
        self.clip.visual.transformer = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print("logit_scale: ", self.logit_scale)
        self.clip_loss = ClipLoss() ## Use the clip loss implementation from OpenCLIP TODO: Maybe not the correct approach
        self.normalize = True


        ## Hack to get around training the SAM and CLIP models
        for param in self.sam.parameters():
            param.requires_grad = False

        for param in self.clip.parameters():
            param.requires_grad = False





    def preprocess_images(self, batched_inputs):
        """
        This does all of the operations required to transform an image from detectron2 dataloader format to that, that is expected by SAM
        1: permute and resize longest edge to 1024
        2: convert to tensor and push to device
        3: permute and convert to batched input format
        4: Normalize pixel values and pad to a square input
        
        ** BBoxes will be resized seperately
        """
        
        #numpy_images = [batch['image'].permute(1,2,0).cpu().numpy() for batch in batched_inputs]

        numpy_images = [x["image"] for x in batched_inputs]
        resized_longest_edge = [self.transform.apply_image(image) for image in numpy_images]
        torch_images = [torch.as_tensor(image, device=self.device) for image in resized_longest_edge]
        torch_images = [image.permute(2, 0, 1).contiguous() for image in torch_images]
        normalized = [self.sam.preprocess(image) for image in torch_images]
        
        ## Now stack together to form a batch
        return torch.stack(normalized, dim = 0)        

    def xyhw_toxyxy(self, xyxy):
        ## SAM outputs boxes in xyhw format, wheras roi_pool expects boxes to be in x1y1x2y2 format
        return [xyxy[0], xyxy[1], xyxy[2]+xyxy[0], xyxy[3]+xyxy[1]]  

    def encode_text (self, refs: List[List]):

        tokenized = torch.cat([clip.tokenize(x).to(self.device) for x in refs])
        with torch.no_grad():
            text_features = self.clip.encode_text(tokenized)
        return text_features


    def forward (self, batched_inputs):        
        # Visual
        ## First preprocess images for SAM
        input_images = self.preprocess_images(batched_inputs).to(self.device)

        ## Now resize the boxes to the 1024x1024 shape of SAM using SAM's tools
        bboxes = [x["bboxes"] for x in batched_inputs]
        original_sizes = [(x['image'].shape[1], x['image'].shape[2]) for x in batched_inputs]
        resized_bboxes = [torch.Tensor(self.transform.apply_boxes(np.array(boxes), hw)).to(self.device) for boxes, hw in zip(bboxes, original_sizes)]      

        # By how much we must scale the bboxes to be correct on the ROI patch
        scale = 64.0/1024.0

        ## Extract features using SAM's Image encoder
        with torch.no_grad():
            features = self.sam.image_encoder(input_images)       


        x = roi_align (features, resized_bboxes, output_size=(7,7), spatial_scale = scale)  

        box_features = self.mlp(x.view(x.shape[0], -1))


        ## Refs
        ## Extract ref from batch
        refs = [x["refexp"] for x in batched_inputs]

        ## Sample a random ref to be used in training
        refs = [[random.choice(k)for k in x["refexp"]] for x in batched_inputs]

        ## Compute Clip embeddings of the refs
        ref_features = self.encode_text(refs).to(dtype=torch.float)

        if self.normalize:
            ref_features = torch.nn.functional.normalize (ref_features, dim = 1)
            box_features = torch.nn.functional.normalize (box_features, dim = 1)


        loss = self.clip_loss (box_features, ref_features, self.logit_scale.exp())

        return loss


    # def clip_like_loss (self, box_features, ref_features):
    #     ## open_clip loss implementation
    #     normalized_ref_features = torch.nn.functional.normalize (ref_features, dim = 1).to(dtype=torch.float)
    #     normalized_box_features = torch.nn.functional.normalize (box_features, dim = 1)

    #     logit_scale = self.logit_scale.exp()

    #     logit_scale = logit_scale.mean()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     batch_size = images.shape[0]
    #     labels = torch.arange(batch_size, device=device).long()
    #     total_loss = (
    #         F.cross_entropy(logits_per_image, labels) +
    #         F.cross_entropy(logits_per_text, labels)
    #     ) / 2

    #     ## Create similarity matrix
    #     temp = self.temperature.exp()
    #     logits = (normalized_ref_features @ normalized_box_features.T) * temp

    #     ## Unlike the OpenAI 
    #     labels = torch.arange(logits.shape[0], device = self.device)

    #     images_similarity = normalized_box_features @ normalized_box_features.T
    #     texts_similarity = normalized_ref_features @ normalized_ref_features.T
    #     targets = torch.nn.functional.softmax(
    #         (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
    #     )

    #     image_loss = torch.nn.functional.cross_entropy (logits, targets, reduction='none')
    #     text_loss = torch.nn.functional.cross_entropy (logits.T, targets.T, reduction='none')

    #     loss = (image_loss + text_loss) / 2

    #     return loss

# class DAAM_Lighning (pl.LightningModule):
    
#     ## ETH has very few > 24Gb cards, so yeah....

    
#     def __init__(self, device = "cpu", model_type = "vit_b", sam_checkpoint = "sam_vit_b_01ec64.pth"):
#         super().__init__()

#         #self.device = device

#         self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#         self.sam.to(device=device)
        
#         self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        
#         self.classifier = nn.Sequential(
#             nn.Linear(256 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 91),
#         ).to(self.device)
        
#         self.train = False
#         self.loss_fn = nn.CrossEntropyLoss()


#     def preprocess_images(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         This does all of the operations required to transform an image from detectron2 dataloader format to that, that is expected by SAM
#         1: permute and resize longest edge to 1024
#         2: convert to tensor and push to device
#         3: permute and convert to batched input format
#         4: Normalize pixel values and pad to a square input
#         """
        
#         numpy_images = [batch['image'].permute(1,2,0).numpy() for batch in batched_inputs]
#         resized_longest_edge = [self.transform.apply_image(image) for image in numpy_images]
#         torch_images = [torch.as_tensor(image, device=self.device) for image in resized_longest_edge]
#         torch_images = [image.permute(2, 0, 1).contiguous() for image in torch_images]
#         normalized = [self.sam.preprocess(image) for image in torch_images]
        
#         ## Now stack together to form a batch
#         return torch.stack(normalized, dim = 0)        

#     def xyxy_toxywh(self, xyxy):
#                     return [xyxy[0], xyxy[1], xyxy[2]+xyxy[0], xyxy[3]+xyxy[1]]  
#     def forward (self, batched_inputs):
        
        
#         ## Resize the detectron2 imagesizes and normalize
#         input_images = self.preprocess_images(batched_inputs)
#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         features = self.sam.image_encoder(input_images)
#         scale = 64.0/1024.0

#         if self.train:
            
#             bboxes = [x.gt_boxes for x in gt_instances]
#             b = [[x.to('cpu').numpy() for x in bbox.__iter__()] for bbox in bboxes]
#             classes = [x.gt_classes for x in gt_instances]


#             ## Resize the detectron2 bounding boxes
#             original_sizes = [(x['image'].shape[1], x['image'].shape[2]) for x in batched_inputs]
#             resized_bboxes = [self.transform.apply_boxes(np.array(boxes), hw) for boxes, hw in zip(b, original_sizes)]


#             boxes = [torch.tensor(x).to('cuda', dtype=torch.float) for x in resized_bboxes]



#             target = torch.cat(classes)
#             x = roi_pool (features, boxes, output_size=(7,7), spatial_scale = 0.1)        
#             probs = self.classifier(x.view(x.shape[0], -1))
#             return probs
#         else:
            
#             mask_generator = SamAutomaticMaskGenerator(self.sam)
#             boxes = []
#             masks = []
#             num_boxes = []
#             for item in batched_inputs:
#                 ## TODO: forward the whole batch in a single go
#                 pix = item['image'].permute(1,2,0).numpy()
#                 mask = mask_generator.generate(pix)
#                 masks += [mask]
#                 boxes += [torch.tensor([self.xyxy_toxywh(box["bbox"]) for box in mask]).to('cuda')]
#             x = roi_pool (features, boxes, output_size=(7,7), spatial_scale = 0.1)  
#             predicted = self.classifier(x.view(x.shape[0], -1))
            
#             return boxes, predicted, masks
            
#     def training_step (self, batch, batch_idx):

#         probs = self(batch)
#         loss = self.loss_fn(probs, probs)

#         return {"loss", loss}

def get_coco_detection_cfg ():
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True,
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000,
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000,
    cfg.DATASETS.PROPOSAL_FILES_TEST = [],
    cfg.DATASETS.PROPOSAL_FILES_TRAIN = [],
    cfg.DATASETS.TEST = ["coco_2017_val"]
    cfg.DATASETS.TRAIN = ["coco_2017_train"]

    return cfg

if __name__ == "__main__":
    daam = DAAM_L()


    ## Create Dataloader
    cfg = get_coco_detection_cfg()
    mapper = DatasetMapper(cfg, True)
    train_data_loader = build_detection_train_loader(cfg, mapper=mapper)

    trainer = Trainer(fast_dev_run = True)
    trainer.fit(daam, train_dataloader = train_data_loader)



     
            
    
