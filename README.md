# SMR

python -m torch.distributed.run --nproc_per_node=8 main.py
nohup python main_1gpu.py > out-i2s.log 2>&1 &