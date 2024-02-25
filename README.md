# (ICLR'24) LIFT: Rethinking Channel Dependence for Multivariate Time Series Forecasting with Leading Indicators

This repo is the official Pytorch implementation of [LIFT: Rethinking Channel Dependence for Multivariate Time Series Forecasting with Leading Indicators](https://arxiv.org/pdf/2401.17548.pdf). 

## Takeaways
- **Rethinking channel dependence in MTS from a perspective of lead-lag relationships.**
![Illustration of locally stationary lead-lag relationships](lead-lag.png)
- **Reasoning why CD models show inferior performance.**
  - Many variates are unaligned with each other, while traditional models (*e.g.*, Informer) simply mix multivariate information at the same time step. Thus they introduce outdated information from lagged variates which are noise and disturb predicting leaders.
  - Though other models (*e.g.*, Vector Auto-Regression) memorize CD from different time steps by static weights, they can suffer from overfiting since the leading indicators and leading steps vary over time. 
- **Alleviating distribution shifts by dynamically selecting and shifting indicators.** 
  - Recent works (*e.g.*, instance normalization methods) focus on distribution shifts in statistical properties (*e.g.*, mean and variance). We take a novel investigation into a different kind of *distribution shifts in channel dependence*.
  - As the leading indicators vary over time, we evaluate all pairs between variates *as fast as possible* and dynamically select a subset from all variates. After the coarse selection, we design a neural network to finely model the channel dependence within the small subset.
  - As the leading steps vary over time, we dynamically shift the selected indicators to get aligned with the target variate, mitigating the varying misalignment.
- **A lightweight yet strong baseline for MTS forecasting.**
  - We propose a parameter-efficient baseline named LightMTS. With the parameter efficiency close to DLinear, LightMTS outperforms DLinear by a large margin and achieves SOTA performance on several benchmarks.

## Scripts
An example:
```bash
python -u run_longExp.py --dataset Weather --model DLinear --lift --seq_len 336 --pred_len 96 --leader_num 8 --state_num 16 --learning_rate 0.001
```

We are still preparing the scripts of experiments, which will come soon!

## Datasets
All benchmarks can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1mJzKrdq-M8C0DrjHeXofcRm-3T3dJ-Gj?usp=sharing).

## Requirements
```bash
pip3 install -r requirements.txt
```

## Citation
If you find this useful for your work, please consider citing it as follows:
```
@inproceedings{
LIFT,
title={LIFT: Rethinking Channel Dependence for Multivariate Time Series Forecasting with Leading Indicators},
author={Lifan Zhao and Yanyan Shen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=JiTVtCUOpS}
}
```