# Wenet Wenetspeech finetuning

This code is based on [Wenet](https://github.com/wenet-e2e/wenet) (git commit: 188e5e9c2c3250f4ed44af4a09d2a8866e4a0ab6). Please check `run.sh`.

We finetuned the Conformer Wenetspeech pretrained model with the train and dev splits, and evaluate on the test split.

Please refer to `run.sh` for data preprocessing, finetuning and evaluation. The finetuning config is located in `models/train.yaml`.

## Results

Pretrained model evaluation results: `stats_pretrained.txt`
```
Level     |Category
----------|------------
mild      |all         :	WER=11.41% N= 64095 C= 58821 D= 982 S=4292 I=2042
moderate  |all         :	WER=17.81% N= 22143 C= 19251 D= 509 S=2383 I=1051
severe    |all         :	WER=33.21% N=  9204 C=  7988 D= 274 S= 942 I=1841
all       |conversation:	WER=11.82% N= 60099 C= 56098 D=1540 S=2461 I=3105
all       |command     :	WER=20.40% N= 35343 C= 29962 D= 225 S=5156 I=1829
all       |all         :	WER=15.00% N= 95442 C= 86060 D=1765 S=7617 I=4934
```

Finetuned model evaluation results: `stats_finetuned.txt`
```
Level     |Category
----------|------------
mild      |all         :    WER= 5.20% N= 64095 C= 61875 D= 876 S=1344 I=1115
moderate  |all         :    WER= 8.00% N= 22143 C= 20903 D= 459 S= 781 I= 532
severe    |all         :    WER= 9.25% N=  9204 C=  8623 D= 302 S= 279 I= 270
all       |conversation:    WER= 7.96% N= 60099 C= 57051 D=1331 S=1717 I=1736
all       |command     :    WER= 3.32% N= 35343 C= 34350 D= 306 S= 687 I= 181
all       |all         :    WER= 6.24% N= 95442 C= 91401 D=1637 S=2404 I=1917
```

On all levels and categories, the finetuning achieved 35.73% WERR.