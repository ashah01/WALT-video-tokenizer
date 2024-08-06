import torch
import torchvision.transforms as transforms
from magvit2_pytorch import VideoTokenizer, VideoTokenizerTrainer
from condition_dataset import ConditionalVideoDataset

dataset = ConditionalVideoDataset("frames_dataset")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
x = next(iter(dataloader)).to('cuda')

# torch.cuda.get_mem_info() is your friend in combating nasty OOMs


tokenizer = VideoTokenizer(
    image_size=128,
    init_dim=22,
    num_res_blocks=1,
    ch_mult=(1, 2),
    z_channels=256,
    perceptual_loss_weight=0,
    use_gan=False,
    adversarial_loss_weight=0,
).to('cuda')

tokenizer.load("checkpoints/checkpoint.pt")
tokenizer.eval()

with torch.no_grad():
    loss, video = tokenizer(x, return_recon_loss_only=True)
    print(loss)


invTrans = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        transforms.ToPILImage(),
    ]
)

images = map(invTrans, video[0].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/pred.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)

images = map(invTrans, x[0].unbind(dim = 1))
first_img, *rest_imgs = images
first_img.save("output/gt.gif", save_all = True, append_images = rest_imgs, duration = 120, loop = 0, optimize = True)