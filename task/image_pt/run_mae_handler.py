# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2024/5/12 14:29
   @Author: ComeOnJian
   @Software: PyCharm
   @File Name:  run_mae_handler.py
   @Description:
            运行mae进行模型预训练, https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mae.py
            TODO:
                1) 参数后期需要抽象出,放在core/hparams下, 分组放好
                2) 数据加载需要抽象出,放在core/data下, 分组放好
                [Optional]3) pipline需要抽象出
-------------------------------------------------
"""
import os
import sys
import logging
from typing import Optional, List
from dataclasses import dataclass, field

import torch
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, Trainer, TrainerCallback
from transformers import ViTMAEConfig, ViTImageProcessor, ViTMAEForPreTraining
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class ModelArguments:
    """
        MAE继续预训练模型参数
    """
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    mask_ratio: float = field(
        default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )


@dataclass
class DataTrainingArguments:
    """
        MAE继续预训练的数据参数
    """
    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_run_args(args):
    """
        获取配置信息
    :param args:
    :return:
    """
    # 定义整个流程中涉及到的参数组合，返回对应的参数对象
    parser = HfArgumentParser([ModelArguments, DataTrainingArguments, CustomTrainingArguments])
    if args is not None:
        args_object_tuple = parser.parse_dict(args=args)
        return args_object_tuple

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args_object_tuple = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        return args_object_tuple

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def data_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


def run_mae_pretraining(model_args: ModelArguments, data_args: DataTrainingArguments,
                        training_args: CustomTrainingArguments, callbacks: Optional[List[TrainerCallback]] = None, ):
    """
        pipline运行
    :return:
    """
    # step1 设置日志信息，并打印基础信息
    if training_args.should_log:
        _set_transformers_logging(log_level=logging.INFO)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # step2 加载数据集, 先按照这个链接下载：https://zhuanlan.zhihu.com/p/685765714
    ds = load_dataset(path = data_args.train_dir)
    # step2.1 对训练数据进行切分
    # data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    # if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
    #     split = ds["train"].train_test_split(data_args.train_val_split)
    #     ds["train"] = split["train"]
    #     ds["validation"] = split["test"]
    ds["validation"] = ds['test']

    # step3 模型配置加载
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
        }
    )
    # step4 加载image process和 model
    # #step 4.1加载image process
    if model_args.image_processor_name:
        image_processor = ViTImageProcessor.from_pretrained(model_args.image_processor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        image_processor = ViTImageProcessor()

    # #step 4.2加载Model
    if model_args.model_name_or_path:
        model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
        )
    else:
        logger.info("Training new model from scratch")
        model = ViTMAEForPreTraining(config)

    # #step 4.2加载image process处理
    if training_args.do_train:
        column_names = ds["train"].column_names
    else:
        column_names = ds["validation"].column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        return examples

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # 计算学习率
    total_train_batch_size = (
            training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # step5 设置训练Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=data_collate_fn,
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def run_pipline():
    pass
