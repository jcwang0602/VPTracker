# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for VPTracker: retain the SFT and inference source subset.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from swift.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'swift.trainers.TrainingArguments',
        'embedding': 'swift.trainers.TrainingArguments',
        'reranker': 'swift.trainers.TrainingArguments',
        'generative_reranker': 'swift.trainers.TrainingArguments',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if getattr(args, 'rlhf_type', None) is not None:
            raise ValueError('This VPTracker ms-swift source subset supports SFT and inference only.')
        train_method = args.task_type
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args._prepare_training_args(args_dict)
        training_args = training_args_cls(**args_dict)
        return training_args
