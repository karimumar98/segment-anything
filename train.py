from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
import pickle
from dam import DAAM
import torch
from torch import nn
import wandb

cfg = pickle.load(open('coco_cfg.pkl', 'rb'))
cfg.defrost()
cfg.SOLVER.IMS_PER_BATCH = 2

use_custom_mapper = False
MapperClass = DatasetMapper

mapper = MapperClass(cfg, True)
data_loader = build_detection_train_loader(cfg, mapper=mapper)

#daam = DAAM(model_type="vit_l", sam_checkpoint = "../sam_vit_l_0b3195.pth", )
daam = DAAM()
daam.train = True

start_iter = 0
max_iter = 100000
optimizer = torch.optim.SGD(daam.classifier.parameters(), lr=0.001, momentum=0.9)
checkpoint_name = "roi_align"

if start_iter != 0:
    checkpoint = torch.load(f"checkpoints/_with_optim{start_iter-1}.pth")
    daam.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"loaded from checkpoint {start_iter}")    

loss_fn = nn.CrossEntropyLoss()

running_loss = 0.0

run = wandb.init(
    # Set the project where this run will be logged
    project="DAM_b_roi_align",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
    })


for batch, iteration in zip(data_loader, range(start_iter, max_iter)):    
    iteration = iteration + 1
    optimizer.zero_grad()

    prediction = daam(batch)
    
    gt_instances = [x["instances"].to(daam.device) for x in batch]
    classes = [x.gt_classes for x in gt_instances]
    target = torch.cat(classes)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()

    if iteration % 100 == 99:    # print every 2000 mini-batches
        print(f'[{iteration + 1}] loss: {running_loss / 100:.3f}')
        wandb.log({"loss": running_loss / 200})
        running_loss = 0.0


    if iteration % 4000 == 3999:    # checkpoint every 4000 mini-batches
        torch.save({
            'iteration': iteration,
            'model_state_dict': daam.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },  f"checkpoints/{checkpoint_name}_{iteration}.pth")
