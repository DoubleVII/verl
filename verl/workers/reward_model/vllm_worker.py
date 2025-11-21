# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
vLLM Reward Model Worker
"""
import os
import torch
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.import_utils import load_extern_type


class VLLMRewardModelWorker(Worker):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize vLLM engine
        model_path = config.model.path
        tensor_parallel_size = config.get("tensor_parallel_size", 1)
        # Default to 1 GPU if not specified, but typically this should match resource pool
        if not torch.cuda.is_available():
             raise RuntimeError("vLLM requires CUDA to run.")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=config.model.get("trust_remote_code", False),
            dtype=config.model.get("dtype", "auto"),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            enforce_eager=config.get("enforce_eager", False),
        )

        # Load custom processor if specified
        self.processor = None
        if config.get("custom_processor", None):
            processor_cls = load_extern_type(config.custom_processor.path, config.custom_processor.name)
            self.processor = processor_cls(config=config)
        
        # Default sampling params for reward generation (usually greedy or specific to the model)
        self.sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
            # Add other params as needed
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # vLLM model is initialized in __init__
        pass

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # 1. Construct prompts
        if self.processor and hasattr(self.processor, "process_input"):
            prompts = self.processor.process_input(data)
        else:
            # Default prompt construction: concatenate prompt and response
            # This assumes data.batch contains 'input_ids' and 'responses' or similar
            # For simplicity, let's assume we decode input_ids to text if no processor
            # But typically vLLM expects text prompts or token ids.
            # Let's try to use the tokenizer from vLLM to decode if needed, or expect text in data
            # For now, let's assume data has 'prompts' and 'responses' in text form in non_tensor_batch
            # or we decode from tensor batch.
            # A safer default might be to expect 'full_text' or similar.
            # Given the user requirement "User will handle prompt construction", 
            # we should probably rely on the processor or a simple default.
            
            # Let's assume data.batch['input_ids'] contains the full sequence (prompt + response)
            # We might need to decode it to text for vLLM if we want to re-process, 
            # but vLLM can take token_ids.
            # However, for reward models, we usually input the full text.
            
            # Placeholder for default logic:
            raise NotImplementedError("Default prompt construction not implemented. Please provide a custom_processor.")

        # 2. Generate
        outputs = self.llm.generate(prompts, self.sampling_params)

        # 3. Extract rewards
        if self.processor and hasattr(self.processor, "process_output"):
            reward_tensor = self.processor.process_output(outputs, data)
        else:
             raise NotImplementedError("Default reward extraction not implemented. Please provide a custom_processor.")
        
        return reward_tensor
