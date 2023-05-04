import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import numpy as np
import random
from dam import DAAM, DAAM_CLIP_ALIGNED
import wandb

## Hack to make TF work together with PyTorch, otherwise TF hogs all memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def main():

    run = wandb.init(
      # Set the project where this run will be logged
      project="DAM_clip_align",
      # Track hyperparameters and run metadata
      config={
          "learning_rate": 0.01,
      })

    checkpoint_name = "clip_align"

    daam = DAAM_CLIP_ALIGNED()
    daam.train = True

    ds = tfds.load("ref_coco", split ="train", data_dir = "/cluster/project/zhang/umarka/clip_detector/datasets/coco")

#    parameters = []
#    parameters.extend(daam.instance_embedder.parameters())
#    parameters.extend(daam.logit_scale)

#    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)
    #optimizer = torch.optim.SGD(daam.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(daam.parameters(), lr=0.01)

    running_loss = 0.0

    ## Get data:
    start_iter = 0
    max_iter = 10000


    def resize_boxes (boxes, scale):
        ## refcoco delivers boxes with the wrong scale, resize them according to the image size
        ## TODO: Make sure this is correct
        for box in boxes:
            box[0], box[2] = scale[0]*box[0], scale[0]*box[2]
            box[1], box[3] = scale[1]*box[1], scale[1]*box[3]
        
        return boxes
    iteration = 0
    batched_inputs = []
    epochs = 100
    for e in range(epochs):
        for example in ds:
            ## Build input dict for DAAM

            ## TODO: Handle more than a single batch
            item = {}
            item["image"] = example["image"].numpy()
            item["bboxes"] = resize_boxes(example["objects"]["bbox"].numpy(), item["image"].shape)
            item["refexp"] = [[sub_ref.numpy().decode("utf-8") for sub_ref in x] for x in example['objects']["refexp"]["raw"]]
        
            batched_inputs += [item]

            if len(batched_inputs) < 4:
              continue


            # for i in range(1000):
            #     optimizer.zero_grad()

            #     loss = daam(batched_inputs)

            #     #print("loss", loss)
            #     loss.backward()
            #     optimizer.step()

            #     print(f'[{iteration + 1}] loss: {loss / 100:.3f}')

        #print(batched_inputs)

            optimizer.zero_grad()

            loss = daam(batched_inputs)

        #print("loss", loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if iteration % 50 == 49:    # print every 2000 mini-batches
                print(f'[{iteration + 1}] loss: {running_loss / 100:.3f}')
                wandb.log({"loss": running_loss / 100})
                running_loss = 0.0
            batched_inputs = []
            iteration += 1

    
        torch.save({
            'epoch': e,
            'model_state_dict': daam.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },  f"checkpoints/{checkpoint_name}_{e}.pth")





if __name__ == "__main__":
    main()
