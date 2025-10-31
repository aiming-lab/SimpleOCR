from transformers import AutoProcessor
import torch
from PIL import Image
from io import BytesIO
import math
from typing import Union, Any, Optional
from PIL.Image import Image as ImageObject
import numpy as np


model_inputs = torch.load("/home/yibop/ocr-unc/debug_adv.pt", map_location="cpu", weights_only=False)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# _switch_dual_branch_response(model_inputs)
# model_inputs = swap_cross_log_probs(model_inputs)
import pdb; pdb.set_trace()