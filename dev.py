import os
from vllm import LLM, SamplingParams


# environments
# os.environ[""] = 
# export TP_SOCKET_IFNAME=bond0
# export GLOO_SOCKET_IFNAME=bond0
# export disable_custom_all_reduce=True
# export NCCL_SOCKET_IFNAME=bond0
# export HF_ENDPOINT=https://hf-mirror.com
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
# os.environ["NCCL_SOCKET_IFNAME"] = "bond0"
# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["disable_custom_all_reduce"] = "True"
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
model_path = "/root/workspace/models/DeepSeek-R1-Distill-Qwen-7B"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# enforce_eager 用于是否开启 CUDA GRAPH 优化，详细见
# https://github.com/vllm-# project/vllm/pull/1926
# 这里设置 enforce_eager=True 的原因是可以调试生成阶段的模型推理过程，否则
# 会使用 CUDA GRAPH 优化模型的推理，就不能一步一步调试了。
# 如果暂时不理解 enforce_eager=True 的作用，可以先不设置这个参数

llm = LLM(model=model_path, enforce_eager=True, dtype='half', tensor_parallel_size=2)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")