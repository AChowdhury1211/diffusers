# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import unittest
import pytest

import torch

import torch.nn.functional as F

from diffusers import LTXVideoTransformer3DModel
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin

from unittest.mock import patch, MagicMock

enable_full_determinism()

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_ltx import LTXVideoAttentionProcessor2_0
try:
    from torch_xla.experimental.custom_kernel import flash_attention
except ImportError:
    print("flash_attention not available.")
    pass


class LTXTransformerTests(ModelTesterMixin, TorchCompileTesterMixin, unittest.TestCase):
    model_class = LTXVideoTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        embedding_dim = 16
        sequence_length = 16

        hidden_states = torch.randn((batch_size, num_frames * height * width, num_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        encoder_attention_mask = torch.ones((batch_size, sequence_length)).bool().to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }

    @property
    def input_shape(self):
        return (512, 4)

    @property
    def output_shape(self):
        return (512, 4)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "num_layers": 1,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"LTXVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class AttnAddedLTXVideoAttentionProcessor2_0Tests(unittest.TestCase):
    def get_constructor_arguments(self, cross_attention: bool = False):
        if cross_attention:
            cross_attention_dim = 8
        else:
            cross_attention_dim = None
            
        return {
            "query_dim": 8, # query_dim = num_attention_heads * attention_head_dim
            "heads": 2, 
            "kv_heads": 2, 
            "dim_head": 4, #  
            "bias" : True,
            "cross_attention_dim": cross_attention_dim,
            "out_bias": True,
            "qk_norm":"rms_norm_across_heads",
            "processor": LTXVideoAttentionProcessor2_0(),
            "use_tpu_flash_attention" : True,
        }
        
    def get_forward_arguments(self, query_dim):
        batch_size = 2
        sequence_length = 4096

        hidden_states = torch.rand(batch_size,sequence_length, query_dim)
        encoder_hidden_states = torch.rand(batch_size, 4, query_dim)
        attention_mask = None

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
        }
    
    def test_use_tpu_flash_attention_flag_when_true(self):
        torch.manual_seed(0)
        
        constructor_args = self.get_constructor_arguments()
        attn = Attention(**constructor_args)

        processor = attn.get_processor()
        assert isinstance(processor, LTXVideoAttentionProcessor2_0), "Processor not LTXVideoAttentionProcessor2_0"
        
        with patch.object(processor, '__call__') as mock_processor_call:
            with patch('torch_xla.experimental.custom_kernel.flash_attention') as mock_flash_attention:
                
                forward_args = self.get_forward_arguments(
                    query_dim=constructor_args["query_dim"]
                )
                attn_hidden_states = attn(**forward_args)

                mock_processor_call.assert_called_once()

                processor_args = mock_processor_call.call_args[0]
                processor_kwargs = mock_processor_call.call_args[1]
                
                assert processor_args[0] == attn
                assert torch.equal(processor_args[1], forward_args['hidden_states'])
                assert torch.equal(processor_kwargs['encoder_hidden_states'], forward_args['encoder_hidden_states'])
                assert processor_kwargs['attention_mask'] == forward_args['attention_mask']