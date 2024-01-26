# accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/sidd_nafnet_wf32.yml --eval_ddp True --verbose True --train 

# accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/psct_wf_32.yml --eval_ddp True --verbose True --train 

# accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/vem.yml --eval_ddp False --verbose True --train 

# accelerate launch --config_file=tools/single_acc.yml --num_processes=1 main.py --config config/pretrain_vdn.yml --eval_ddp False --verbose True --train 

accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/pretrain_vem.yml --eval_ddp False --verbose True --resume True --resume_state results/VEModelPretrain-SSCT/save_state/save_state_225000 --train 
