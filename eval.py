import torch
from condition_dataset import ConditionalVideoDataset
from train import LitAutoEncoder
import random
import torchvision.transforms as transforms

# # load checkpoint
checkpoint = "checkpoints/5m.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, init_dim=18, num_blocks=2)

# # choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

decoder = autoencoder.decoder
decoder.eval()

dataset = ConditionalVideoDataset('frames_dataset')
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

batch = next(iter(loader)).to('cuda') # batch size 1

with torch.no_grad():
    pred = decoder(encoder(batch))

scores = torch.nn.functional.mse_loss(pred, batch, reduction="none").mean(dim=(1, 2, 3, 4))
idx = scores.argmin()
print(scores.min())

invTrans = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        transforms.ToPILImage(),
    ]
)

images = map(invTrans, pred[idx].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/pred.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)

images = map(invTrans, batch[idx].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/gt.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)