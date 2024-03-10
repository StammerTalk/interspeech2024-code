# Fairseq hubert finetuning

This code is based on [fairseq](https://github.com/facebookresearch/fairseq).

We finetuned the [Hubert Wenetspeech pretrained model](https://huggingface.co/TencentGameMate/chinese-hubert-large) with the train and dev splits, and evaluate on the test split.

Please refer to [hubert](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) for data preprocessing, finetuning and evaluation. The finetuning configs for AISHELL-1 and AS-70 can be found at `config/finetune/large_200h.yaml` and `config/finetune/large_100h.yaml`, respectively.

## Results

AISHELL-1 model evaluation results
```
Level     |Category
----------|------------
mild      |all         :        WER=16.28% N= 99270 C= 86336 D=2948 S=9986 I=3223
moderate  |all         :        WER=22.86% N= 36711 C= 30097 D=1214 S=5400 I=1777
severe    |all         :        WER=26.76% N= 17802 C= 15042 D= 318 S=2442 I=2003
all       |conversation:        WER=16.80% N=104103 C= 91951 D=3900 S=8252 I=5338
all       |command     :        WER=23.79% N= 49680 C= 39524 D= 580 S=9576 I=1665
all       |all         :        WER=19.06% N=153783 C=131475 D=4480 S=17828 I=7003
```

AS-70 model evaluation results
```
Level     |Category
----------|------------
mild      |all         :        WER= 6.25% N= 99270 C= 94669 D=1981 S=2620 I=1607
moderate  |all         :        WER= 9.67% N= 36711 C= 33962 D=1152 S=1597 I= 801
severe    |all         :        WER= 7.85% N= 17802 C= 16720 D= 603 S= 479 I= 316
all       |conversation:        WER= 9.75% N=104103 C= 96548 D=3209 S=4346 I=2594
all       |command     :        WER= 2.03% N= 49680 C= 48803 D= 527 S= 350 I= 130
all       |all         :        WER= 7.25% N=153783 C=145351 D=3736 S=4696 I=2724
```

On all levels and categories, the finetuning achieved 61.96% WERR.