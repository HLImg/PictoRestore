# accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/sidd_nafnet_wf32.yml --eval_ddp True --verbose True --train 

accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/psct_wf_32.yml --eval_ddp True --verbose True --train 