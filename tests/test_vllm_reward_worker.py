import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import torch

# Mock ray and vllm before importing verl
sys.modules["ray"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["tensordict"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()
sys.modules["codetiming"] = MagicMock()

from verl import DataProto
from verl.workers.reward_model.vllm_worker import VLLMRewardModelWorker

class MockConfig:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, MockConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

class TestVLLMRewardModelWorker(unittest.TestCase):
    @patch('verl.workers.reward_model.vllm_worker.LLM')
    @patch("verl.workers.reward_model.vllm_worker.load_extern_type")
    @patch("torch.cuda.is_available", return_value=True)
    def test_compute_rm_score(self, mock_cuda_available, mock_load_extern_type, mock_llm_cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        
        # Mock config
        config_data = {
            "model": {"path": "dummy_path"},
            "custom_processor": {"path": "dummy_processor_path", "name": "DummyProcessor"},
            "tensor_parallel_size": 1
        }
        config = MockConfig(config_data)

        # Mock Processor
        mock_processor_cls = MagicMock()
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        mock_load_extern_type.return_value = mock_processor_cls

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Initialize Worker
        worker = VLLMRewardModelWorker(config)

        # Mock Data
        data = DataProto(batch=None) # Dummy data

        # Mock Processor behavior
        mock_processor.process_input.return_value = ["prompt1", "prompt2"]
        mock_processor.process_output.return_value = torch.tensor([1.0, 0.5])

        # Mock LLM behavior
        mock_outputs = [MagicMock(), MagicMock()]
        mock_llm.generate.return_value = mock_outputs

        # Run compute_rm_score
        reward = worker.compute_rm_score(data)

        # Verify
        mock_processor.process_input.assert_called_once_with(data)
        mock_llm.generate.assert_called_once()
        mock_processor.process_output.assert_called_once_with(mock_outputs, data)
        self.assertTrue(torch.equal(reward, torch.tensor([1.0, 0.5])))

if __name__ == '__main__':
    unittest.main()
