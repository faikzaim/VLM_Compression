from typing import Any

import torch
from transformers import LlamaConfig

from videollava.model.compressor.compression_module import CompressorModule
from videollava.model.compressor.keyframe_selector import KeyframeSelectorBase
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.model.multimodal_encoder.languagebind import LanguageBind, to_device
from videollava.model.multimodal_encoder.languagebind.image.tokenization_image import LanguageBindImageTokenizer


class KeyframeSelectorLanguageBind(KeyframeSelectorBase):
    def __init__(self, model_config: dict | None = None):
        super().__init__(model_config)

        # Initialised in load_model()
        self.model = None
        self.device = None
        self.tokenizer = None

    def load_model(self, device) -> None:
        tokenizer = LanguageBindImageTokenizer.from_pretrained(
            'LanguageBind/LanguageBind_Image',
            cache_dir='./cache_dir/tokenizer_cache_dir'
        )

        lb_model = LanguageBind(clip_type={'image': 'LanguageBind_Image'}, cache_dir='./cache_dir')
        lb_model = lb_model.to(device=device,  dtype=torch.float16)
        lb_model.eval()

        self.model = lb_model
        self.tokenizer = tokenizer
        self.device = device

    def process_video(self, videos: Any):
        video_tower = self.model.modality_encoder['image']
        video_tower.select_feature = 'cls_patch'
        pooled_out = []
        with torch.no_grad():
            for i in range(0, videos.shape[0], 8):
                chunk = videos[i:i + 8]
                video_forward_outs = video_tower(chunk)
                pooled_out.append(video_forward_outs[1])

        return torch.cat(pooled_out).unsqueeze(-2)

    def process_text(self, prompt: str):
        inputs = {
            'language': to_device(
                self.tokenizer(prompt, max_length=77, padding='max_length', truncation=True, return_tensors='pt'),
                self.device),
        }

        with torch.no_grad():
            text_embed = self.model(inputs)['language']

        return text_embed


    def project_features(self, features):
        projector = self.model.modality_proj['image']
        projected = projector(features)
        return projected / projected.norm(dim=-1, keepdim=True)


class LlavaLlamaForCausalLMWithCompression(LlavaLlamaForCausalLM):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config)

        self.compressor = None
        if compressor_config := kwargs.pop("compressor_config", None):
            self.compressor = CompressorModule(compressor_config)

    def encode_videos(self, videos):
        b, _, t, _, _ = videos.shape
        video_features, video_attentions = self.get_model().get_video_tower()(videos, output_attentions=True)
        video_attentions = video_attentions.view(b, t, *video_attentions.shape[1:])
        video_features = self.compressor(video_features, video_attentions)
        video_features = self.get_model().mm_projector(video_features)
        return video_features
