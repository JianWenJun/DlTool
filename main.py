# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2024/5/12 14:50
   @Author: ComeOnJian
   @Software: PyCharm
   @File Name:  main.py
   @Description:
            main运行不同的计算任务
-------------------------------------------------
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Automatic Weights & Biases logging enabled, to disable set 如下为true, 否则注销
# os.environ["WANDB_DISABLED"] = "true"

def do_mae_pretraining():
    """
        mae 预训练任务
    :return:
    """
    # step1 导入包
    from task.image_pt.run_mae_handler import run_mae_pretraining, get_run_args, run_pipline
    # step1.1 参数类别用于查询进行相应配置
    from task.image_pt.run_mae_handler import CustomTrainingArguments, ModelArguments, DataTrainingArguments, \
        TrainingArguments
    # step2 参数设置
    args = {
        # 训练-基础参数
        "seed": 1337,
        "output_dir": './result/cifar10_mae',
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "num_train_epochs": 800,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        # "load_best_model_at_end": True,
        # 训练-日志
        "logging_strategy": "steps",
        "logging_steps": 10,
        "log_on_each_node": False,
        "disable_tqdm": False,
        "report_to": "wandb",
        # 训练- 评测
        "evaluation_strategy": "steps",  # steps, epoch
        "eval_steps": 100,
        "eval_delay": 1,
        # 训练-保存
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "save_on_each_node": False,
        # 训练-优化学习 - 默认adamW
        "lr_scheduler_type": "cosine",
        "base_learning_rate": 1.5e-4,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        # 训练-分布式
        "local_rank": -1,
        "ddp_find_unused_parameters": None,  # ddp设置True
        # 模型 -
        "model_name_or_path": None,  # 随机初始化
        "config_name": "/root/models/vit-mae-large",  # 不带模型文件的地址
        "cache_dir": "./data/process/cifar10",
        "mask_ratio": 0.75,
        "norm_pix_loss": True,
        # 数据 -
        "label_names": ["pixel_values"],
        "train_dir": "./data/process/cifar10",
        "remove_unused_columns": False,  # 自定义DataCollate为False
        "dataset_name": 'cifar10',
        "dataloader_num_workers": 4
    }
    # step2.1 解析参数
    model_args, data_args, trainings_args = get_run_args(args=args)
    # step3 设置训练过程中的回调
    callbacks = []
    # step4 运行trainer
    run_mae_pretraining(model_args=model_args, data_args=data_args, training_args=trainings_args, callbacks=callbacks)


if __name__ == '__main__':
    # 运行mae预训练任务
    do_mae_pretraining()
