"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import os
import re
from typing import Any, List, Optional, Union

import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers import MBartConfig, MBartForCausalLM, AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel


class SwinEncoder(nn.Module):
    r"""
    Donut encoder based on SwinTransformer
    """
    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None
        if not name_or_path:
            swin_state_dict = timm.create_model("swin_base_patch4_window12_384", pretrained=True).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                    pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                    new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return self.to_tensor(ImageOps.expand(img, padding))


class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART, modified to use vinai/bartpho-large.
    """
    def __init__(
        self, decoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        # ================== THAY ĐỔI QUAN TRỌNG NHẤT ==================
        # Sử dụng `vinai/bartpho-large` (1024-dim) thay vì `vinai/bartpho-word` (768-dim)
        # để tương thích với `donut-base` (1024-dim)
        if not name_or_path or name_or_path == "":
             self.name_or_path = "vinai/bartpho-large"
        else:
             self.name_or_path = name_or_path
        # =============================================================

        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        # Tải config từ `vinai/bartpho-large` để lấy đúng d_model=1024
        bart_config = MBartConfig.from_pretrained(self.name_or_path) 

        bart_config.is_decoder = True
        bart_config.is_encoder_decoder = False
        bart_config.add_cross_attention = True
        
        # Ghi đè các tham số từ config của Donut (nếu cần, nhưng giữ nguyên d_model=1024)
        bart_config.decoder_layers = self.decoder_layer
        bart_config.max_position_embeddings = self.max_position_embeddings
        
        self.add_special_tokens(["<sep/>"])
        bart_config.vocab_size = len(self.tokenizer)

        # Khởi tạo mô hình Decoder với config đã được tùy chỉnh
        self.model = MBartForCausalLM(config=bart_config)
        self.model.forward = self.forward
        self.model.config.is_encoder_decoder = True
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        
        # Tải trọng số từ `vinai/bartpho-large`
        print(f"Initializing Donut decoder from pretrained BART model: {self.name_or_path}")
        bart_state_dict = MBartForCausalLM.from_pretrained(self.name_or_path).state_dict()
        
        # Resize position embedding nếu cần
        if bart_state_dict["model.decoder.embed_positions.weight"].shape[0] != self.max_position_embeddings + 2:
            bart_state_dict["model.decoder.embed_positions.weight"] = self.resize_bart_abs_pos_emb(
                bart_state_dict["model.decoder.embed_positions.weight"],
                self.max_position_embeddings + 2,
            )
        
        # Nạp trọng số, `strict=False` sẽ bỏ qua các layer không liên quan
        self.model.load_state_dict(bart_state_dict, strict=False)

    def add_special_tokens(self, list_of_tokens: List[str]):
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, past_key_values=None, past=None, use_cache: bool = None, attention_mask: torch.Tensor = None):
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.model.lm_head(outputs[0])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight


class DonutConfig(PretrainedConfig):
    model_type = "donut"
    def __init__(
        self,
        input_size: List[int] = [2560, 1920],
        align_long_axis: bool = False,
        window_size: int = 10,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 1536,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.name_or_path = name_or_path


class DonutModel(PreTrainedModel):
    config_class = DonutConfig
    base_model_prefix = "donut"

    def __init__(self, config: DonutConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
        )
        self.decoder = BARTDecoder(
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            name_or_path=self.config.name_or_path,
        )

    def forward(self, image_tensors: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor):
        encoder_outputs = self.encoder(image_tensors)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
        )
        return decoder_outputs

    def inference(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
    ):
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"
            return obj

    def token2json(self, tokens, is_inner_value=False):
        output = dict()
        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in self.decoder.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]
                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)
        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        # ================== SỬA LỖI TẠI ĐÂY ==================
        # 1. Đặt `ignore_mismatched_sizes=True` vào kwargs một cách an toàn.
        kwargs['ignore_mismatched_sizes'] = True
        
        # 2. Gọi hàm from_pretrained của class cha.
        #    Lệnh này sẽ tải trọng số của Encoder (donut-base), 
        #    và bỏ qua trọng số của Decoder cũ do không khớp size.
        model = super(DonutModel, cls).from_pretrained(
            pretrained_model_name_or_path, 
            *model_args,
            **kwargs
        )
        # =======================================================

        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,
                )
            )
            model.config.max_position_embeddings = max_length
        return model