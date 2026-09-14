# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for VPTracker: retain the SFT and inference source subset.
# General inference arguments extracted from upstream rlhf_trainers/args_mixin.py.
from dataclasses import dataclass
from typing import Optional, Union

from swift.utils import json_parse_to_dict


@dataclass
class VllmArguments:
    """VllmArguments is a dataclass that holds the configuration for vllm.

    Args:
        vllm_gpu_memory_utilization (float): GPU memory utilization. Default is 0.9.
        vllm_tensor_parallel_size (int): Tensor parallelism size. Default is 1.
        vllm_pipeline_parallel_size (int): Pipeline parallelism size. Default is 1.
        vllm_enable_expert_parallel (bool): Flag to enable expert parallelism for MoE models. Default is False.
        vllm_max_num_seqs (int): Maximum number of sequences. Default is 256.
        vllm_max_model_len (Optional[int]): Maximum model length. Default is None.
        vllm_disable_custom_all_reduce (bool): Flag to disable custom all-reduce. Default is True.
        vllm_enforce_eager (bool): Flag to enforce eager execution. Default is False.
        vllm_limit_mm_per_prompt (Optional[str]): Limit multimedia per prompt. Default is None.
        vllm_max_lora_rank (int): Maximum LoRA rank. Default is 16.
        vllm_enable_prefix_caching (Optional[bool]): Flag to enable automatic prefix caching. Default is None.
        vllm_use_async_engine (Optional[bool]): Whether to use async engine for vLLM. Default is None,
            which will be set to True for encode tasks (embedding, seq_cls, reranker, generative_reranker),
            deployment scenarios (swift deploy) and False otherwise.
        vllm_quantization (Optional[str]): The quantization method for vLLM. Default is None.
        vllm_reasoning_parser (Optional[str]): The reasoning parser for vLLM. Default is None.
        vllm_disable_cascade_attn (bool): Flag to disable cascade attention. Default is False.
        vllm_mm_processor_cache_gb (Optional[float]): MM processor cache size in GB. Default is None.
        vllm_speculative_config (Optional[Union[dict, str]]): Speculative decoding configuration, passed in as a JSON
            string. Defaults to None.
        vllm_engine_kwargs (Optional[Union[dict, str]]): Additional parameters for vllm, formatted as a JSON string.
            Defaults to None.
        vllm_data_parallel_size (int): Data parallelism size for vLLM rollout. Default is 1.
    """
    # vllm
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_enable_expert_parallel: bool = False
    vllm_max_num_seqs: int = 256
    vllm_max_model_len: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_max_lora_rank: int = 16
    vllm_enable_prefix_caching: Optional[bool] = None
    vllm_use_async_engine: Optional[bool] = None
    vllm_quantization: Optional[str] = None
    vllm_reasoning_parser: Optional[str] = None
    vllm_disable_cascade_attn: bool = False
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_speculative_config: Optional[Union[dict, str]] = None
    vllm_engine_kwargs: Optional[Union[dict, str]] = None
    # rollout
    vllm_data_parallel_size: int = 1

    def __post_init__(self):
        if self.vllm_limit_mm_per_prompt is not None:
            self.vllm_limit_mm_per_prompt = json_parse_to_dict(self.vllm_limit_mm_per_prompt)
        if self.vllm_speculative_config is not None:
            self.vllm_speculative_config = json_parse_to_dict(self.vllm_speculative_config)
        self.vllm_engine_kwargs = json_parse_to_dict(self.vllm_engine_kwargs)

    def get_vllm_engine_kwargs(self):
        # Some parameters are provided by BaseArguments
        adapters = self.adapters
        if hasattr(self, 'adapter_mapping'):
            adapters = adapters + list(self.adapter_mapping.values())
        kwargs = {
            'gpu_memory_utilization': self.vllm_gpu_memory_utilization,
            'tensor_parallel_size': self.vllm_tensor_parallel_size,
            'pipeline_parallel_size': self.vllm_pipeline_parallel_size,
            'enable_expert_parallel': self.vllm_enable_expert_parallel,
            'max_num_seqs': self.vllm_max_num_seqs,
            'max_model_len': self.vllm_max_model_len,
            'disable_custom_all_reduce': self.vllm_disable_custom_all_reduce,
            'enforce_eager': self.vllm_enforce_eager,
            'limit_mm_per_prompt': self.vllm_limit_mm_per_prompt,
            'max_lora_rank': self.vllm_max_lora_rank,
            'enable_lora': len(adapters) > 0,
            'max_loras': max(len(adapters), 1),
            'enable_prefix_caching': self.vllm_enable_prefix_caching,
            'use_async_engine': self.vllm_use_async_engine,
            'quantization': self.vllm_quantization,
            'reasoning_parser': self.vllm_reasoning_parser,
            'disable_cascade_attn': self.vllm_disable_cascade_attn,
            'mm_processor_cache_gb': self.vllm_mm_processor_cache_gb,
            'speculative_config': self.vllm_speculative_config,
            'num_labels': self.num_labels,
            'engine_kwargs': self.vllm_engine_kwargs,
        }
        if self.task_type in ('embedding', 'seq_cls') or 'reranker' in self.task_type:
            kwargs['task_type'] = self.task_type

        return kwargs
