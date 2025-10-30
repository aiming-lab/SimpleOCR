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

import os
import random
import logging
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTextOverlay:
    """Utility class for adding text overlays to images"""
    
    def __init__(self, seed: Optional[int] = None, enable_random: bool = True):
        """
        Args:
            seed: Random seed for reproducibility (optional)
            enable_random: Whether to use random fonts/colors (if False, use fixed settings)
        """
        self.enable_random = enable_random
        self.rng = random.Random(seed) if seed is not None else random
        
        # Font configuration
        self.font_options = {
            'sizes': list(range(18, 42, 3)),
            'colors': [
                (0, 0, 0),        # 黑色
                (25, 25, 112),    # 深蓝
                (139, 0, 0),      # 深红
                (0, 100, 0),      # 深绿
                (75, 0, 130),     # 靛蓝
                (128, 0, 128),    # 紫色
                (0, 0, 139),      # 深蓝
                (178, 34, 34),    # 火砖红
            ]
        }
        
        # 固定设置（当enable_random=False时使用）
        self.fixed_font_size = 24
        self.fixed_color = (0, 0, 0)  # 黑色
        
        self.fonts = self._load_fonts()
    
    def _load_fonts(self):
        """Load available fonts, separated by language support"""
        fonts = {
            'chinese': [],
            'english': []
        }
        
        # Chinese font paths
        chinese_font_paths = [
            # Linux Chinese fonts
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            # macOS Chinese fonts
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            # Windows Chinese fonts (WSL)
            "/mnt/c/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
            "/mnt/c/Windows/Fonts/simhei.ttf",  # SimHei
            "/mnt/c/Windows/Fonts/simsun.ttc",  # SimSun
        ]
        
        # English font paths
        english_font_paths = [
            # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Times.ttc",
            "/Library/Fonts/Arial.ttf"
        ]
        
        # Check local fonts folder
        local_fonts_dir = Path("fonts")
        if local_fonts_dir.exists():
            logger.info("Checking local fonts folder...")
            font_patterns = ["*.ttf", "*.otf", "*.TTF", "*.OTF", "*.ttc", "*.TTC"]
            for pattern in font_patterns:
                for font_file in local_fonts_dir.glob(pattern):
                    font_path = str(font_file)
                    font_name = font_file.name.lower()
                    # Determine if it's a Chinese font based on filename
                    if any(cn in font_name for cn in ['cjk', 'chinese', 'noto', 'wqy', 'zh', 'cn', 
                                                       'simhei', 'simsun', 'yahei', 'heiti', 'pingfang']):
                        fonts['chinese'].append(font_path)
                        logger.info(f"Found local Chinese font: {font_file.name}")
                    else:
                        fonts['english'].append(font_path)
                        logger.info(f"Found local English font: {font_file.name}")
        
        # Check system Chinese fonts
        for font_path in chinese_font_paths:
            if os.path.exists(font_path):
                fonts['chinese'].append(font_path)
                logger.debug(f"Found system Chinese font: {os.path.basename(font_path)}")
        
        # Check system English fonts
        for font_path in english_font_paths:
            if os.path.exists(font_path):
                fonts['english'].append(font_path)
                logger.debug(f"Found system English font: {os.path.basename(font_path)}")
        
        # Ensure at least default font
        if not fonts['chinese']:
            fonts['chinese'] = [None]
            logger.warning("No Chinese fonts found! Chinese text may not display correctly")
        
        if not fonts['english']:
            fonts['english'] = [None]
        
        # logger.info(f"Loaded: {len([f for f in fonts['chinese'] if f])} Chinese fonts, "
        #            f"{len([f for f in fonts['english'] if f])} English fonts")
        
        return fonts
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        for char in text:
            # Unicode range for CJK Unified Ideographs
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    def _get_random_font(self, font_size: int, text: str = "") -> ImageFont.FreeTypeFont:
        """Get appropriate font based on text content"""
        # Detect if text contains Chinese characters
        has_chinese = self._contains_chinese(text)
        
        # Select font list based on text content
        font_list = self.fonts['chinese'] if has_chinese else self.fonts['english']
        font_path = self.rng.choice(font_list) if self.enable_random else font_list[0]
        
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
                logger.debug(f"Using {'Chinese' if has_chinese else 'English'} font: "
                           f"{os.path.basename(font_path)} at size {font_size}")
                return font
            else:
                logger.debug(f"Using default font at size {font_size}")
                return ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}, using default")
            return ImageFont.load_default()
    
    def _get_random_color(self) -> tuple:
        """Get color (random or fixed based on settings)"""
        if self.enable_random:
            return self.rng.choice(self.font_options['colors'])
        else:
            return self.fixed_color
    
    def _get_font_size(self) -> int:
        """Get font size (random or fixed based on settings)"""
        if self.enable_random:
            return self.rng.choice(self.font_options['sizes'])
        else:
            return self.fixed_font_size
    
    def add_text_to_image(self, image: ImageObject, text: str) -> ImageObject:
        """
        Add text overlay to image
        
        Args:
            image: PIL Image object
            text: Text to add (e.g., "Question: ...")
        
        Returns:
            New PIL Image object with text overlay at the bottom
        """
        try:
            img_width, img_height = image.size
            padding = 20
            max_text_width = img_width - 2 * padding
            
            initial_font_size = self._get_font_size()
            color = self._get_random_color()
            
            font_size = initial_font_size
            wrapped_lines = []
            line_height = 0
            
            # Try to fit text with decreasing font sizes
            while font_size >= 12:
                font = self._get_random_font(font_size, text)
                wrapped_lines = []
                
                # Estimate character width
                test_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                if hasattr(font, 'getbbox'):
                    bbox = font.getbbox(test_text)
                    avg_char_width = (bbox[2] - bbox[0]) / len(test_text)
                    line_height = int((bbox[3] - bbox[1]) * 1.3)
                else:
                    try:
                        avg_char_width = font.getsize(test_text)[0] / len(test_text)
                        line_height = int(font.getsize('Ay')[1] * 1.3)
                    except:
                        avg_char_width = 10
                        line_height = int(font_size * 1.3)
                
                chars_per_line = int(max_text_width / avg_char_width * 0.85)
                chars_per_line = max(10, chars_per_line)
                
                # Wrap text by paragraphs and words
                for paragraph in text.split('\n'):
                    if not paragraph.strip():
                        wrapped_lines.append('')
                        continue
                    
                    words = paragraph.split()
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        word_length = len(word)
                        estimated_length = current_length + word_length + len(current_line)
                        
                        if estimated_length > chars_per_line and current_line:
                            test_line = ' '.join(current_line)
                            if hasattr(font, 'getbbox'):
                                actual_width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
                            else:
                                try:
                                    actual_width = font.getsize(test_line)[0]
                                except:
                                    actual_width = len(test_line) * avg_char_width
                            
                            if actual_width <= max_text_width:
                                wrapped_lines.append(test_line)
                            else:
                                wrapped_lines.append(' '.join(current_line[:-1]) if len(current_line) > 1 else current_line[0])
                                current_line = current_line[-1:] if len(current_line) > 1 else []
                            
                            current_line = [word]
                            current_length = word_length
                        else:
                            current_line.append(word)
                            current_length += word_length
                    
                    if current_line:
                        test_line = ' '.join(current_line)
                        if hasattr(font, 'getbbox'):
                            actual_width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
                        else:
                            try:
                                actual_width = font.getsize(test_line)[0]
                            except:
                                actual_width = len(test_line) * avg_char_width
                        
                        if actual_width <= max_text_width:
                            wrapped_lines.append(test_line)
                        else:
                            # Handle very long words
                            while current_line:
                                temp_line = []
                                temp_length = 0
                                while current_line and temp_length < chars_per_line:
                                    word = current_line.pop(0)
                                    if len(word) > chars_per_line:
                                        wrapped_lines.append(word[:chars_per_line-1] + '-')
                                        current_line.insert(0, word[chars_per_line-1:])
                                        break
                                    else:
                                        temp_line.append(word)
                                        temp_length += len(word) + 1
                                
                                if temp_line:
                                    wrapped_lines.append(' '.join(temp_line))
                
                total_text_height = len(wrapped_lines) * line_height + 2 * padding
                
                # Check if all lines fit
                all_lines_fit = True
                for line in wrapped_lines:
                    if line.strip():
                        if hasattr(font, 'getbbox'):
                            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
                        else:
                            try:
                                line_width = font.getsize(line)[0]
                            except:
                                line_width = len(line) * avg_char_width
                        
                        if line_width > max_text_width:
                            all_lines_fit = False
                            break
                
                if all_lines_fit or font_size <= 12:
                    break
                
                font_size -= 2
            
            # Create new image with space for text at bottom
            new_height = img_height + total_text_height + padding
            new_image = Image.new('RGB', (img_width, new_height), 'white')
            
            # Paste original image at top
            new_image.paste(image, (0, 0))
            
            # Draw text at bottom
            draw = ImageDraw.Draw(new_image)
            y_position = img_height + padding
            
            for line in wrapped_lines:
                if line.strip():
                    if hasattr(font, 'getbbox'):
                        bbox = font.getbbox(line)
                        text_width = bbox[2] - bbox[0]
                    else:
                        try:
                            text_width = font.getsize(line)[0]
                        except:
                            text_width = len(line) * avg_char_width
                    
                    # Center text horizontally
                    if text_width <= max_text_width:
                        x_position = (img_width - text_width) // 2
                        x_position = max(padding, x_position)
                    else:
                        x_position = padding
                    
                    draw.text((x_position, y_position), line, font=font, fill=color)
                
                y_position += line_height
            
            return new_image
            
        except Exception as e:
            logger.error(f"Failed to add text to image: {e}")
            # Return original image if processing fails
            return image


# Global instance for easy access (non-seeded version)
_overlay_instance = None


def add_text_to_image(image: ImageObject, text: str, seed: Optional[int] = None) -> ImageObject:
    """
    Convenience function to add text overlay to image
    
    Args:
        image: PIL Image object
        text: Text to add to image
        seed: Random seed for reproducibility (optional).
              If provided, creates a new instance for deterministic output.
              If None, uses global instance with random behavior.
    
    Returns:
        PIL Image object with text overlay
    """
    global _overlay_instance
    
    # If seed is provided, create a new instance for deterministic behavior
    if seed is not None:
        instance = ImageTextOverlay(seed=seed, enable_random=True)
        return instance.add_text_to_image(image, text)
    
    # Otherwise, use global instance for efficiency (but non-deterministic)
    if _overlay_instance is None:
        _overlay_instance = ImageTextOverlay(seed=None, enable_random=True)
    return _overlay_instance.add_text_to_image(image, text)