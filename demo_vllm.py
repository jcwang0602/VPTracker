# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List
import os


def infer_batch(engine: "InferEngine", infer_requests: List["InferRequest"]):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    query0 = infer_requests[0].messages[0]["content"]
    print(f"query0: {query0}")
    print(f"response0: {resp_list[0].choices[0].message.content}")
    print(f"metric: {metric.compute()}")
    # metric.reset()  # reuse


def infer_stream(engine: "InferEngine", infer_request: "InferRequest"):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    gen_list = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]["content"]
    print(f"query: {query}\nresponse: ", end="")
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end="", flush=True)
    print()
    print(f"metric: {metric.compute()}")


if __name__ == "__main__":
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats

    model = os.environ.get("VPTRACK_MODEL", "jcwang0602/VPTracker")
    infer_backend = "vllm"

    if infer_backend == "pt":
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == "vllm":
        from swift.llm import VllmEngine

        engine = VllmEngine(model, max_model_len=8192, seed=42)
    elif infer_backend == "sglang":
        from swift.llm import SglangEngine

        engine = SglangEngine(model)
    elif infer_backend == "lmdeploy":
        from swift.llm import LmdeployEngine

        engine = LmdeployEngine(model)

    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset(["AI-ModelScope/alpaca-gpt4-data-zh#1000"], seed=42)[0]
    print(f"dataset: {dataset}")
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    messages = [{"role": "user", "content": "who are you?"}]
    infer_stream(engine, InferRequest(messages=messages))
