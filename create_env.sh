#!/bin/bash


mkdir tmp

export TMPDIR=./tmp/

conda create -n crossnet python==3.10

conda activate crossnet

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

pip install --pre torcheval-nightly

pip install -r requirements.txt

pip install -U tensorboard

pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics
