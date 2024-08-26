import torch
from condition_dataset import ConditionalVideoDataset
from train import LitAutoEncoder
import random
import torchvision.transforms as transforms

# # load checkpoint
checkpoint = "checkpoints/checkpoint.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, init_dim=16, num_blocks=1)

# # choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

decoder = autoencoder.decoder
decoder.eval()

dataset = ConditionalVideoDataset('frames_dataset')

idx = random.randint(0, len(dataset))
batch = dataset[idx].unsqueeze(0).to('cuda') # batch size 1

with torch.no_grad():
    pred = decoder(encoder(batch))

invTrans = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        transforms.ToPILImage(),
    ]
)

images = map(invTrans, pred[0].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/pred.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)

images = map(invTrans, batch[0].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/gt.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)