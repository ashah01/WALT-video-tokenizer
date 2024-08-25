import torch
from model import Encoder, Decoder

def get_params(in_dim, blocks, ch_mult):

    encoder = Encoder(
        ch=in_dim,
        in_channels=3,
        num_res_blocks=blocks,
        z_channels=256,
        resolution=128,
        ch_mult=ch_mult
    )

    decoder = Decoder(
        out_ch=3,
        z_channels=256,
        num_res_blocks=blocks,
        ch=in_dim,
        resolution=128,
        ch_mult=ch_mult
    )

    tokenizer = torch.nn.Sequential(encoder, decoder)
    return sum([p.numel() for p in tokenizer.parameters()])

# print(get_params(12, 3, (1, 2, 2, 4)))
print(get_params(16, 1, (1, 2, 2, 4)))
