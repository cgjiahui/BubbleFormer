#!/bin/bash
module load anaconda/2021.05
source activate torch
export PYTHONUNBUFFERED=1
python ./BubbleFormer/main.py