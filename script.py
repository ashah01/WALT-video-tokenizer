import torch
import torchvision.transforms as transforms
from magvit2_pytorch import VideoTokenizer, VideoTokenizerTrainer
from condition_dataset import ConditionalVideoDataset

dataset = ConditionalVideoDataset("frames_dataset")

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
)


trainer = VideoTokenizerTrainer(
    tokenizer,
    use_wandb_tracking=True,  # use True for metric logging MODIFIED
    dataset=dataset,
    dataset_type="videos",  # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size=16,
    grad_accum_every=1,
    learning_rate=3e-4,
    num_train_steps=5000,
    # validate_every_step = 1000,
    num_frames=16,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={"T_max": 5000, "eta_min": 3e-5},
    warmup_steps=500
)

# trainer.load("checkpoints/checkpoint.pt")

with trainer.trackers(project_name="magvit2-video", run_name="wider (size=1M, dim=22)"):
    trainer.train()
