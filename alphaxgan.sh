#!/usr/bin/env bash
# sh alphaxgan.sh
# 1.4.1確認済
#--run_mode normal \

COMMAND="python3 search.py \
--use_skip \
--use_controllerG \
--use_controllerD \
--reset_controller \
--grow_step1 1 \
--grow_step2 2 \
--loss BCE \
--use_dynamic_reset"

#echo command is
#echo $COMMAND
#$COMMAND

#<<EXPERIMENT
mkdir ../ALL_output

# 基本
python3 search.py \
--run_mode normal \
--use_skip \
--use_controllerG \
--use_controllerD \
--reset_controller \
--loss BCE \
--use_dynamic_reset \
--use_controller_rollout

mv ../output ../ALL_output/output_AlphaXGAN

#EXPERIMENT