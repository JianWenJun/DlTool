


### 停止命令
ps -ef|grep python|grep -v grep|awk '{print $2}'|xargs sudo -E kill -9


### wandb使用
https://zhuanlan.zhihu.com/p/493093033

### 多卡训练
    """
    # nproc_per_node 每个节点多少张卡
    # nnodes 表示节点数
    # node_rank 节点编号
    # master_addr
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12355 main.py
    """