# Uni-Retrieval

python -m torch.distributed.run --nproc_per_node=8 main.py > out_i2s_ddp.log 2>&1 &
nohup python main_1gpu.py > out-i2t-after-i2s.log 2>&1 &