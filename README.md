This is the official implementation for 
# [CrossNet: Leveraging Global, Cross-Band, Narrow-Band, and Positional Encoding for Single- and Multi-Channel Speaker Separation](https://arxiv.org/abs/2403.03411). arXiv:2403.03411.


## abstract 
We introduce CrossNet, a complex spectral mapping approach to speaker separation and enhancement in reverberant and noisy conditions. The proposed architecture comprises an encoder layer, a global multi-head self-attention module, a cross-band module, a narrow-band module, and an output layer. CrossNet captures global, cross-band, and narrow-band correlations in the time-frequency domain. To address performance degradation in long utterances, we introduce a random chunk positional encoding. Experimental results on multiple datasets demonstrate the effectiveness and robustness of CrossNet, achieving state-of-the-art performance in tasks including reverberant and noisy-reverberant speaker separation. Furthermore, CrossNet exhibits faster and more stable training in comparison to recent baselines. Additionally, CrossNet's high performance extends to multi-microphone conditions, demonstrating its versatility in various acoustic scenarios.
