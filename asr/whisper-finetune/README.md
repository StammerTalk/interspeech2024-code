# Whisper finetuning

This code is based on [Whisper-Finetune](https://github.com/yeyupiaoling/Whisper-Finetune).

We finetuned the [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) with the train and dev splits, and evaluate on the test split.

Please refer to [Whisper-Finetune](https://github.com/yeyupiaoling/Whisper-Finetune) for data preprocessing, finetuning and evaluation. 

## Results

Whisper-large-v2 model evaluation results
```
Level     |Category
----------|------------
mild      |all         :        WER=14.50% N= 99270 C= 88271 D=4430 S=6569 I=3397
moderate  |all         :        WER=28.50% N= 36711 C= 30109 D=2685 S=3917 I=3860
severe    |all         :        WER=95.33% N= 17806 C= 14606 D=1091 S=2109 I=13775
all       |conversation:        WER=17.83% N=104103 C= 92948 D=7073 S=4082 I=7403
all       |command     :        WER=46.85% N= 49684 C= 40038 D=1133 S=8513 I=13629
all       |all         :        WER=27.20% N=153787 C=132986 D=8206 S=12595 I=21032
```

Finetuned model evaluation results
```
Level     |Category
----------|------------
mild      |all         :        WER= 5.18% N= 99270 C= 95905 D=1192 S=2173 I=1782
moderate  |all         :        WER=13.46% N= 36711 C= 34477 D= 633 S=1601 I=2709
severe    |all         :        WER=18.92% N= 17806 C= 16720 D= 499 S= 587 I=2283
all       |conversation:        WER=10.19% N=104103 C= 98642 D=2136 S=3325 I=5146
all       |command     :        WER= 5.74% N= 49684 C= 48460 D= 188 S=1036 I=1628
all       |all         :        WER= 8.75% N=153787 C=147102 D=2324 S=4361 I=6774
```

On all levels and categories, the finetuning achieved 67.83% WERR.