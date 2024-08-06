import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets.video_utils import VideoClips
import pickle
import torch.nn.functional as F


"""

Input -> prior frame + N previous actions
Output -> current frame

Pad the first frames with no action, and give the first frame itself as the prior frame

"""

class ConditionalVideoDataset(Dataset):
    def __init__(self, directory, num_actions=16+13, clip_length=16, skip_length=16):
        self.num_actions = num_actions
        self.dir = directory
        self.clip_length = clip_length
        self.skip_length = skip_length


        # Video clip handling
        eps = []
        for x in next(os.walk(self.dir))[1]:
            if x.startswith("episode"):
                eps.append(os.path.join("frames_dataset", x, "output.mp4"))

        eps = sorted(eps, key=lambda x: int(x.split("_")[2].split("/")[0]))

        self.video_clips = VideoClips(eps, clip_length_in_frames=self.clip_length, frames_between_clips=self.skip_length, output_format="TCHW")

        # Transformation handling
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Lambda(lambda x: x.float().div(255)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Action handling
        self.actions = []
        for ep in range(1, len(eps) + 1):
            with open(f"{self.dir}/episode_{ep}/actions.pkl", "rb") as f:
                a_ep = pickle.load(f)
                self.actions.append(a_ep)
                f.close()


    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        # TCHW -> C,D,H,W
        video = video.permute(1, 0, 2, 3)
        video = torch.stack([*map(self.transform, video.unbind(dim=1))], dim=1)
        #
        # if video_idx > 0:
        #     idx -= self.video_clips.cumulative_sizes[video_idx - 1]
        #
        # # clip_length - 1 because actions occur after frames are saved
        # start_clip = max(0, idx * self.clip_length - (self.num_actions - self.clip_length + 1))
        # act_window = self.actions[video_idx][start_clip:idx * self.clip_length + (self.clip_length - 1)]
        # padding = [3] * (self.num_actions - len(act_window))
        # act_window = padding + act_window

        # return video, F.one_hot(torch.tensor(act_window), 4)
        return video
class ConditionalFramesDataset(Dataset):
    def __init__(self, directory, num_actions):
        self.num_actions = num_actions
        self.dir = directory

        num_ep = len(next(os.walk(self.dir))[1])

        self.ep_lengths = [
            len(
                [
                    name
                    for name in os.listdir(f"{self.dir}/episode_{ep}")
                    if os.path.isfile(f"{self.dir}/episode_{ep}/" + name)
                    and name.endswith(".png")
                ]
            )
            for ep in range(1, num_ep + 1)
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((120, 160)),
                transforms.Lambda(lambda x: x.float().div(255)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.actions = []

        for ep in range(1, num_ep + 1):
            with open(f"{self.dir}/episode_{ep}/actions.pkl", "rb") as f:
                a_ep = pickle.load(f)
                self.actions.append(a_ep)
                f.close()

    def __len__(self):
        return sum(self.ep_lengths)

    def __getitem__(self, idx):
        ptr = idx
        i = 0
        while ptr >= self.ep_lengths[i]:
            ptr -= self.ep_lengths[i]
            i += 1

        target = read_image(f"{self.dir}/episode_{i+1}/{ptr + 1}.png")
        target = self.transform(target)

        if ptr == 0:
            previous = target.clone()
        else:
            previous = read_image(f"{self.dir}/episode_{i+1}/{ptr}.png")
            previous = self.transform(previous)

        if ptr >= self.num_actions + 1:
            act_window = self.actions[i][ptr - self.num_actions : ptr]
        else:
            padding = [3] * (self.num_actions - ptr)
            act_window = padding + self.actions[i][:ptr]

        act_window = F.one_hot(torch.tensor(act_window), 4)

        return previous, act_window, target


if __name__ == "__main__":
    dataset = ConditionalFramesDataset(5)
    dataloader = DataLoader(dataset, batch_size=16)
    prev, actwin, target = next(iter(dataloader))
    import IPython

    IPython.embed()
