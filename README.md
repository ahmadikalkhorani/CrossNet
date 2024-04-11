This is the official pytorch implementation of 
# [CrossNet: Leveraging Global, Cross-Band, Narrow-Band, and Positional Encoding for Single- and Multi-Channel Speaker Separation](https://arxiv.org/abs/2403.03411).

#### Vahid Ahmadi Kalkhorani, DeLiang Wang | Perception and Neurodynamics Laboratory (PNL), The Ohio State University


## abstract 
We introduce CrossNet, a complex spectral mapping approach to speaker separation and enhancement in reverberant and noisy conditions. The proposed architecture comprises an encoder layer, a global multi-head self-attention module, a cross-band module, a narrow-band module, and an output layer. CrossNet captures global, cross-band, and narrow-band correlations in the time-frequency domain. To address performance degradation in long utterances, we introduce a random chunk positional encoding. Experimental results on multiple datasets demonstrate the effectiveness and robustness of CrossNet, achieving state-of-the-art performance in tasks including reverberant and noisy-reverberant speaker separation. Furthermore, CrossNet exhibits faster and more stable training in comparison to recent baselines. Additionally, CrossNet's high performance extends to multi-microphone conditions, demonstrating its versatility in various acoustic scenarios.



[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.03411) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ahmadikalkhorani.CrossNet)





# Pre-requisites
## creating a conda environment
```bash create_env.sh```

# Training 

```bash
conda activate crossnet

B = # batch size

python -u Trainer.py \
    --config=/path/to/model/config/file.yaml \
    --config=/path/to/data/config/file.yaml \
    --data.batch_size=[B,B] \
    --data.audio_time_len=[4.0,4.0,null] \
    --model.exp_name="experiment_name" \
    --model.arch.dim_input=2 \ # 2 x number of channels
    --model.arch.dim_output=4 \ # 2 x number of speakers
    --trainer.precision="16-mixed" \
    --trainer.accelerator="gpu" \
    --trainer.devices=-1 \
    --trainer.max_epochs=200 \
    --ckpt_path=path/to/last/ckpt.ckpt \ # path to last checkpoint to resume training
```

# Testing 

```bash
conda activate crossnet

best_ckpt=$(python scripts/best_ckpt.py --path path/to/current/exp/log/folder) 

cfg_path="$(dirname $(dirname "$best_ckpt"))/config.yaml"


python -u Trainer.py test \
    --config=$cfg_path \
    --model.metrics=[SDR,SI_SDR,NB_PESQ,eSTOI] \
    --model.write_examples=20 \
    --ckpt_path=$best_ckpt \

```

# Acknowledgements

This is repository is mainly adopted from [NBSS](https://github.com/Audio-WestlakeU/NBSS). We thank the authors for their great code. 

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{kalkhorani2024crossnet,
    title={{CrossNet}: Leveraging Global, Cross-Band, Narrow-Band, and Positional Encoding for Single- and Multi-Channel Speaker Separation},
  author={Ahmadi Kalkhorani, Vahid and Wang, De Liang},
  journal={arXiv: 2403.03411},
  year={2024}
}
```