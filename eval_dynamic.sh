#!/bin/sh

# dataset from 'nlvr2', 'adobe', 'spotdiff'
dataset=fake_media

# Main metric to use
metric=BLEU

# model from 'init', 'newheads', 'newcross', 'dynamic', which are the four model in paper 
# related to the four subsections in the paper
model=dynamic

# Name of the model, used in snapshot
name=${model}_2pixel

task=speaker
if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

log_dir=$dataset/$task/$name
mkdir -p snap/$log_dir
mkdir -p log/$dataset/$task
cp $0 snap/$log_dir/run.bash
cp -r src snap/$log_dir/src

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src/main.py --output snap/$log_dir \
    --maxInput 40 --metric $metric --model $model --imgType pixel --worker 4 --train testspeaker --dataset $dataset \
    --batchSize 95 --hidDim 512 --dropout 0.5 \
    --seed 9595 \
    --optim adam --lr 1e-4 --epochs 500 \
    --load /home/jzda/VisualRelationships/snap/fake_media/speaker/dynamic_2pixel/best_eval
