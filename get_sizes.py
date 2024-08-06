import torch
import torchvision.transforms as transforms
from magvit2_pytorch import VideoTokenizer, VideoTokenizerTrainer
from condition_dataset import ConditionalVideoDataset

# dataset = ConditionalVideoDataset("frames_dataset")

# torch.cuda.get_mem_info() is your friend in combating nasty OOMs
def get_params(in_dim, blocks, ch_mult):

    tokenizer = VideoTokenizer(
        image_size=128,
        init_dim=in_dim,
        num_res_blocks=blocks,
        ch_mult=ch_mult,
        z_channels=256,
        perceptual_loss_weight=0,
        use_gan=False,
        adversarial_loss_weight=0,
    )

    return sum([p.numel() for p in tokenizer.parameters()])

print(get_params(22, 1, (1, 2)))
