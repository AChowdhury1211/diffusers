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

from diffusers.utils.import_utils import is_torch_xla_available

enable_full_determinism()

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


class TestLTXVideoAttentionProcessor2_0(unittest.TestCase):
    def setUp(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            F.scaled_dot_product_attention = MagicMock()
        
        self.processor = LTXVideoAttentionProcessor2_0()
        
        self.batch_size = 2
        self.sequence_length = 256
        self.hidden_dim = 64
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.hidden_states = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attention_mask = torch.ones(self.batch_size, 1, self.sequence_length, self.sequence_length)
        self.image_rotary_emb = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        
        self.attn = MagicMock()
        self.attn.use_tpu_flash_attention = True
        self.attn.heads = self.num_heads
        self.attn.scale = 0.125
        
        self.attn.prepare_attention_mask.return_value = self.attention_mask
        self.attn.to_q.return_value = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attn.to_k.return_value = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attn.to_v.return_value = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attn.norm_q.return_value = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attn.norm_k.return_value = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        
        mock_output_tensor = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.attn.to_out = [
            MagicMock(return_value=mock_output_tensor),
            MagicMock(return_value=mock_output_tensor)
        ]
    
    @patch('diffusers.models.transformers.transformer_ltx.apply_rotary_emb')
    @patch('torch_xla.experimental.custom_kernel.flash_attention')
    def test_tpu_flash_attention_path(self,mock_flash_attention, mock_apply_rotary_emb):
        mock_apply_rotary_emb.side_effect = lambda x, _: x        
        expected_output = torch.randn(
            self.batch_size, self.num_heads, self.sequence_length, self.head_dim
        )
        mock_flash_attention.return_value = expected_output
        
        reshaped_mask = self.attention_mask.expand(-1, self.num_heads, -1, -1)
        
        result = self.processor(
            self.attn,
            self.hidden_states,
            attention_mask=self.attention_mask,
            image_rotary_emb=self.image_rotary_emb
        )
        
        mock_flash_attention.assert_called_once()
        
        call_args = mock_flash_attention.call_args[1]
        
        self.assertEqual(call_args['q'].shape, (self.batch_size, self.num_heads, self.sequence_length, self.head_dim))
        self.assertEqual(call_args['k'].shape, (self.batch_size, self.num_heads, self.sequence_length, self.head_dim))
        self.assertEqual(call_args['v'].shape, (self.batch_size, self.num_heads, self.sequence_length, self.head_dim))
        
        self.assertEqual(call_args['q_segment_ids'].shape, (self.batch_size, self.sequence_length))
        self.assertTrue(torch.all(call_args['q_segment_ids'] == 1.0))
        
        self.assertEqual(call_args['kv_segment_ids'].dtype, torch.float32)
        
        self.assertEqual(call_args['sm_scale'], self.attn.scale)
        
        self.assertEqual(result.shape, (self.batch_size, self.sequence_length, self.hidden_dim))
        
        self.attn.to_out[0].assert_called_once()
        self.attn.to_out[1].assert_called_once()

    @patch('ltx_processor.apply_rotary_emb')
    @patch('ltx_processor.flash_attention')
    def test_attention_mask_handling_tpu(self, mock_flash_attention, mock_apply_rotary_emb):
        mock_apply_rotary_emb.side_effect = lambda x, _: x
        mock_flash_attention.return_value = torch.randn(
            self.batch_size, self.num_heads, self.sequence_length, self.head_dim
        )
        
        custom_mask = torch.zeros(self.batch_size, 1, self.sequence_length, self.sequence_length)
        custom_mask[:, :, :, :self.sequence_length//2] = 1.0
        
        self.attn.prepare_attention_mask.return_value = custom_mask
        
        self.processor(
            self.attn,
            self.hidden_states,
            attention_mask=custom_mask,
            image_rotary_emb=self.image_rotary_emb
        )
        
        call_args = mock_flash_attention.call_args[1]
        self.assertEqual(call_args['kv_segment_ids'].dtype, torch.float32)

    @patch('ltx_processor.apply_rotary_emb')
    @patch('ltx_processor.flash_attention')
    def test_sequence_length_requirement(self, mock_flash_attention, mock_apply_rotary_emb):
        mock_apply_rotary_emb.side_effect = lambda x, _: x
        
        invalid_seq_len = 100
        hidden_states = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        
        self.attn.to_q.return_value = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        self.attn.to_k.return_value = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        self.attn.to_v.return_value = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        self.attn.norm_q.return_value = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        self.attn.norm_k.return_value = torch.randn(self.batch_size, invalid_seq_len, self.hidden_dim)
        
        with self.assertRaises(AssertionError):
            self.processor(
                self.attn,
                hidden_states,
                attention_mask=None,
                image_rotary_emb=None
            )
