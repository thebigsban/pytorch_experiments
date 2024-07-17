# PyTorch Experiments

Conda environment because setting up Docker with Jupyter notebook and remote file access takes too long. 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge jupyter matplotlib 
pip3 install einops
pip3 install tqdm
```

```bash
conda activate torchnb
```


## Rotary Positional Encoding
Not sure about terminology, but the `rotate_half` implementation gives the correct results. Tried to implement one that was more in-line with the original Roformer paper, but that actually didn't end up working (the dot product after rotation depended on the index itself as well as the difference in indices). 


## Embedding
Basically just trying to figure out how `nn.Embedding` works. All it does is translate `int` to a `torch.tensor`. Also seems like the resulting `torch.tensor` can be updated via backprop. 

## HuggingFace Tokenizer
[Github](https://github.com/huggingface/tokenizers) and [Documentation](https://huggingface.co/docs/tokenizers/index)
```
pip install tokenizers
```

Did a test training on UniprotKB dataset, took a few minutes for 1k merges on 500k ish sequences. Also code for saving and loading models. 
