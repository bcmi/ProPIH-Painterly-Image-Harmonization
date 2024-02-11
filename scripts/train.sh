#!/usr/bin/env bash
DISPLAY_PORT=8097

G='vgg19hrnet'
D='conv'
model_name=vgg19hrnet
loadSize=256

#hyper-parameters
gpu_id=0
L_S=1
L_C=1
L_GAN=1
L_tv=1e-5
lr=2e-4
batchs=4
load_iter=0


datasetmode=cocoart
content_dir="../datasets/painterly/MS-COCO/"
style_dir="../datasets/painterly/wikiart/"

NAME="${G}_Content${L_C}_Style${L_S}_Ggan${L_GAN}_lr${lr}_batch${batchs}"

checkpoint="../checkpoints/"


CMD="python ../train.py \
--name $NAME \
--checkpoints_dir $checkpoint \
--model $model_name \
--netG $G \
--netD $D \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--is_train 1 \
--display_id 0 \
--normD batch \
--normG batch \
--preprocess none \
--niter 20 \
--niter_decay 10 \
--input_nc 3 \
--batch_size $batchs \
--num_threads 6 \
--print_freq 500 \
--display_freq 500 \
--save_latest_freq 1000 \
--patch_number 4 \
--gpu_ids $gpu_id \
--lambda_g $L_GAN \
--lambda_style $L_S \
--lambda_content $L_C \
--lambda_tv $L_tv \
--lr $lr \
--load_iter $load_iter  \
--continue_train \
--epoch latest \


"
echo $CMD
eval $CMD
