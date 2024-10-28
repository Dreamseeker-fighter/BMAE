# automatic_battery_mask_coding_detector
利用MAE方式实现电池数据大模型预训练  

## 运行方式

1. 首先检查确认main_pretrain.py 中的数据路径和log路径
2. 单机多卡运行命令，nproc_per_node表示调用卡的数量
   ```python
    python -m torch.distributed.launch --nproc_per_node=10 main_pretrain.py
   
3.finetune
python -m torch.distributed.launch --nproc_per_node=10 main_finetune.py \
     --anomaly_days=30 --desc fengchao_new_30_d_8_small \
     --finetune=/data/nfsdata005/small_small_570.pth \
     --model=battery_mae_vit_small \
     --min_lr 0.00001 \
     --blr 0.001 \
     --data_path /data/nfsdata/database/FENGCHAO/batch1/volt88_temp32/1ep_s10_d50_t60/all_data/svolt1_pack_label_final \
     --train_label /code/zhangyang/test/fengchao_label/train_label.csv \
     --valid_label /code/zhangyang/test/fengchao_label/val_label.csv \
     --negative_sample_expanded 10


3.inference
python -m torch.distributed.launch --nproc_per_node=1 main_inference.py \
  --anomaly_days=30 --desc jeve_90_d_8_large_inference \
  --finetune /data/log/zengjunjie/mae/2023-01-30-12-57-47/model/checkpoint-23.pth \
  --model=battery_mae_vit_tiny \
  --eval \
  --data_path /data/nfsdata/database/JEVE/batch3/volt96_temp48/4ep_s10_d60_t30/all_data/feather \
  --test_label /code/zhangyang/test/jeve_b3_label/${data}_label.csv
