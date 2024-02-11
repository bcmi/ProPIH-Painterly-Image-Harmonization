#!/usr/bin/env bash
DISPLAY_PORT=8097

G='vgg19hrnet'
D='conv'
model_name=vgg19hrnet
loadSize=256

#hyper-parameters
gpu_id=0
batchs=1
load_iter=0

# network design
patch_num=4

test_epoch=latest
datasetmode=cocoart
content_dir="../examples/"
style_dir="../examples/"

NAME="pretrained"

checkpoint="../checkpoints/"


CMD="python ../test.py \
--name $NAME \
--checkpoints_dir $checkpoint \
--model $model_name \
--netG $G \
--netD $D \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--is_train 0 \
--display_id 0 \
--normD batch \
--normG batch \
--preprocess none \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size $batchs \
--num_threads 6 \
--print_freq 1000 \
--display_freq 1 \
--save_latest_freq 1000 \
--gpu_ids $gpu_id \
--epoch $test_epoch \
"
echo $CMD
eval $CMD
