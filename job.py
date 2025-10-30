from transformers import AutoProcessor
import torch
from PIL import Image
from io import BytesIO
import math
from typing import Union, Any, Optional
from PIL.Image import Image as ImageObject
import numpy as np

# 加载输入
model_inputs = torch.load("/home/yibop/ocr-unc/debug_data_compute_log_probs.pt", map_location="cpu", weights_only=False)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels=262144, max_pixels=4194304
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


# target = "EFGH$."
# matched_indices = []

# # 注意这里要取 model_inputs.batch
# input_ids = model_inputs.batch['input_ids']
# import pdb; pdb.set_trace()
# for i, ids in enumerate(input_ids):
#     text = processor.decode(ids, skip_special_tokens=False)
#     # import pdb; pdb.set_trace()
#     if target in text:
#         matched_indices.append(i)
#         print(f"Matched sample index: {i}")
#         print("Decoded text snippet:")
#         print(text)
#         print("=" * 60)

# print("All matched indices:", matched_indices)

# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
_cache = {}
def _process_multi_modal_inputs(data):
    if "multi_modal_data" not in data.non_tensor_batch:
        return

    if "uid" in _cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
        _cache.clear()

    if "multi_modal_inputs" not in _cache:
        min_pixels = data.meta_info["min_pixels"]
        max_pixels = data.meta_info["max_pixels"]
        video_fps = data.meta_info["video_fps"]
        batch_multi_modal_inputs = []
        multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples
        for index, multi_modal_data in zip(
            data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
        ):  # process multi modal data per sample
            if index not in multi_modal_inputs_cache:
                images, videos = [], []
                if "images" in multi_modal_data:
                    for image in multi_modal_data["images"]:
                        images.append(image)

                if len(images) != 0:
                    # it's necessary to add `dict` to properly convert batch features to dict
                    # otherwise the batch features will be converted to dict keys
                    # see https://github.com/hiyouga/EasyR1/pull/339
                    multi_modal_inputs = dict(processor.image_processor(images=images, return_tensors="pt"))
                    multi_modal_inputs["image"] = images
                elif len(videos) != 0:
                    multi_modal_inputs = dict(
                        processor.image_processor(images=None, videos=videos, return_tensors="pt")
                    )
                else:
                    multi_modal_inputs = {}
                multi_modal_inputs_cache[index] = multi_modal_inputs

            batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

        _cache["uid"] = data.non_tensor_batch["uid"]
        _cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

    data.non_tensor_batch["multi_modal_inputs"] = _cache["multi_modal_inputs"]
    import pdb; pdb.set_trace()
_process_multi_modal_inputs(model_inputs)
