import math
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from magvit2_pytorch.arnav_modules import Encoder, Decoder
from condition_dataset import ConditionalVideoDataset


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, init_dim, num_blocks, ch_mult):
        super().__init__()
        self.encoder = Encoder(
            ch=init_dim,
            in_channels=3,
            num_res_blocks=num_blocks,
            z_channels=256,
            resolution=128,
            ch_mult=ch_mult
        )

        self.decoder = Decoder(
            out_ch=3,
            z_channels=256,
            num_res_blocks=num_blocks,
            ch=init_dim,
            resolution=128,
            ch_mult=ch_mult
    )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch # currently all actions data is omitted, just receive input signal
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("recon_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("valid_recon_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        def cosine_annealing_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_max, eta_min):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return eta_min/eta_max + (1 - eta_min/eta_max) * 0.5 * (1.0 + math.cos(math.pi * progress))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


        return {"optimizer": optimizer, "lr_scheduler": cosine_annealing_with_warmup(optimizer, 500, 5000, 3e-4, 3e-5)}

def main():
    L.seed_everything(42)

    # Data
    dataset = ConditionalVideoDataset("frames_dataset")

    # Split data in to train, val, test
    train_set_size = int(len(dataset) * 0.9)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=False)

    # Model
    model = LitAutoEncoder(28, 1, (2,))

    # Trainer
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", filename="checkpoint", every_n_train_steps=100)
    logger = WandbLogger(project="mavit2-video", name="super_wide (size=1M, dim=28 blocks=1, mult=(2,))")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision="16-mixed",
        max_steps=5000,
        check_val_every_n_epoch=None,
        val_check_interval=100,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()

# # load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# # embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)