#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0"

stage=0 # start from 0 if you need to start from data preparation
stop_stage=6

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

num_utts_per_shard=500

# data folder should contain ./ref/{0001..0070}.csv and ./wav/{0001..0070}/meeting_01.wav
data_source=

nj=16
dict=data/dict/lang_char.txt

train_set=train
dev_set=dev
test_sets="test/mild/A test/mild/P test/moderate/A test/moderate/P test/severe/A test/severe/P"

train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=false
decode_checkpoint=$dir/42.pt
average_num=30

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/prepare_data.sh ${data_source} || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  tools/compute_cmvn_stats.py --num_workers 32 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/$train_set/global_cmvn
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in $train_set $dev_set; do
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads 32 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text $(realpath data/$x/shards) data/$x/data.list
  done

  for level in mild moderate severe; do
    # A: conversation, P: commands
    for category in A P; do
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 32 --segments data/test/$level/${category}/segments \
        data/test/$level/wav.scp data/test/$level/${category}/text \
        $(realpath data/test/$level/${category}/shards) data/test/$level/${category}/data.list
    done
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  # You had better rm it manually before you start run.sh on first node.
  # rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  # The number of gpus runing on each node/machine
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_sed.py \
      --config $train_config \
      --data_type shard \
      --train_data data/$train_set/data.list \
      --cv_data data/${dev_set}/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num}
  fi
  for test_set in ${test_sets}; do
    test_dir=$dir/${test_set}
    mkdir -p $test_dir
    python wenet/bin/infer_sed.py --gpu 0 \
      --config $dir/train.yaml \
      --data_type shard \
      --test_data data/${test_set}/data.list \
      --checkpoint $decode_checkpoint \
      --result_dir $test_dir \
      --batch_size 128
  done
  python local/mk_stat.py $dir "${test_sets}"
fi
