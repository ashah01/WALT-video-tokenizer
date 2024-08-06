## WALT-video-tokenizer - Pytorch

Implementation of the MagViT2 video tokenizer from <a href="https://walt-video-diffusion.github.io/assets/W.A.L.T.pdf">Photorealistic Video Generation with Diffusion Models</a> in Pytorch. This technique currently holds SOTA for video generation / understanding. Additionally, this repository is [at the time of writing] the only faithful implementation of the architecture laid out in the paper for video modeling.

## Usage


A sample script for running a forward pass is included in `test.py`. The `VideoTokenizerTrainer` object can be employed to train the model on any custom video dataset.


To track your experiments on <a href="https://wandb.ai">Weights & Biases</a> set `use_wandb_tracking = True` on `VideoTokenizerTrainer`, and then use the `.trackers` context manager

```python

trainer = VideoTokenizerTrainer(
    use_wandb_tracking = True,
    ...
)

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()

```

## Citations

```bibtex
@misc{yu2023language,
    title   = {Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation}, 
    author  = {Lijun Yu and José Lezama and Nitesh B. Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G. Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A. Ross and Lu Jiang},
    year    = {2023},
    eprint  = {2310.05737},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Zhang2021TokenST,
    title   = {Token Shift Transformer for Video Classification},
    author  = {Hao Zhang and Y. Hao and Chong-Wah Ngo},
    journal = {Proceedings of the 29th ACM International Conference on Multimedia},
    year    = {2021}
}
```

```bibtex
@inproceedings{Arora2023ZoologyMA,
    title   = {Zoology: Measuring and Improving Recall in Efficient Language Models},
    author  = {Simran Arora and Sabri Eyuboglu and Aman Timalsina and Isys Johnson and Michael Poli and James Zou and Atri Rudra and Christopher R'e},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:266149332}
}
```
