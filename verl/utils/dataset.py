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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from .image_text_overlay import add_text_to_image


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        enable_dual_branch: bool = False,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.text_overlay = enable_dual_branch
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def _extract_question_text(self, example: dict[str, Any], fallback_prompt: Optional[str] = None) -> str:
        """
        Extract clean question text for overlay
        
        Priority:
        1. Use question_key field if available
        2. Extract from fallback_prompt by removing markers
        3. Return empty string
        """
        if self.prompt_key in example and example[self.prompt_key]:
            return str(example[self.prompt_key])
        elif fallback_prompt is not None:
            return fallback_prompt.replace("<image>", "").replace("<video>", "").strip()
        else:
            return ""
    
    def _add_text_overlay(self, example: dict[str, Any], text: str) -> dict[str, Any]:
        images = example.get(self.image_key, [])
        if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
            images = [os.path.join(self.image_dir, image) for image in images]

        processed_images = [] if len(images) != 0 else None  # text-only data
        for image in images:
            # load image first
            if isinstance(image, str):
                pil_img = Image.open(image)
            elif isinstance(image, dict):
                pil_img = Image.open(BytesIO(image["bytes"]))
            elif isinstance(image, bytes):
                pil_img = Image.open(BytesIO(image))
            else:
                pil_img = image
            pil_img.load()
            # add text overlay
            overlay_text_image = add_text_to_image(pil_img, f"Question: {text}")
            processed_images.append(process_image(overlay_text_image, self.min_pixels, self.max_pixels))

        return processed_images

    def _apply_chat_template(self, messages):
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _load_and_process_media(self, example, text_overlay, prompt_text):
        if self.image_key in example:
            images = self._resolve_paths(example[self.image_key])
            if text_overlay:
                images = self._add_text_overlay(example, prompt_text)
            processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]
            return {"images": processed_images}, "image"

        elif self.video_key in example:
            videos = self._resolve_paths(example[self.video_key])
            processed_videos = [process_video(v, self.min_pixels, self.max_pixels, self.video_fps) for v in videos]
            return {"videos": processed_videos}, "video"

        else:
            return None, "text"

    def _apply_processor(self, multimodal_inputs, prompt):
        if self.processor is not None:
            if multimodal_inputs and "videos" in multimodal_inputs:
                return self.processor(videos=multimodal_inputs["videos"], text=[prompt], return_tensors="pt")
            elif multimodal_inputs and "images" in multimodal_inputs:
                return self.processor(multimodal_inputs["images"], [prompt], return_tensors="pt")
        return self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")

    def _build_position_ids(self, model_inputs, input_ids, attention_mask):
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_pos = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            text_pos = torch.arange(len(input_ids)).unsqueeze(0)
            return torch.cat((text_pos, vision_pos), dim=0)
        return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

    def _truncate_prompt(self, prompt):
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                return raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                return raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt too long: {len(raw_prompt_ids)} > {self.max_prompt_length}")
        return raw_prompt_ids

    def _resolve_paths(self, items):
        if self.image_dir and isinstance(items, list) and len(items) > 0 and isinstance(items[0], str):
            return [os.path.join(self.image_dir, i) for i in items]
        return items

    def __len__(self):
        return len(self.dataset)

    def _make_branch(self, example, messages, original_prompt_text, text_overlay=False):
        if text_overlay:
            prompt_text = "<image>\nPlease answer the question in the image."
            messages = self._build_messages({self.prompt_key: prompt_text})
        else:
            prompt_text = original_prompt_text

        prompt = self._apply_chat_template(messages)

        multimodal_inputs, multimodal_type = self._load_and_process_media(example, text_overlay, prompt_text)
        example["multi_modal_data"] = multimodal_inputs

        model_inputs = self._apply_processor(multimodal_inputs, prompt)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        position_ids = self._build_position_ids(model_inputs, input_ids, attention_mask)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self._truncate_prompt(prompt)

        example_out = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            raw_prompt_ids=raw_prompt_ids,
            ground_truth=example.pop(self.answer_key, None),
            multi_modal_data=multimodal_inputs,
        )

        return example_out

    def __getitem__(self, index):
        example = self.dataset[index]
        prompt_text = example.get(self.prompt_key, None)
        messages = self._build_messages(example)

        # branch A: original prompt without overlay
        branch_A = self._make_branch(example, messages, prompt_text, text_overlay=False)

        # branch B: new prompt with overlay
        if self.text_overlay and self.image_key in example:
            branch_B = self._make_branch(example, messages, prompt_text, text_overlay=True)
            return branch_A, branch_B

        return branch_A
