# git push to github
git push --force origin master
# run training script
accelerate launch --config_file=resource/acc.yaml --machine_rank=0 --num_machines=1 --main_process_port=29506 main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml