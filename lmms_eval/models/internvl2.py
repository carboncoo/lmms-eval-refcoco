from typing import List, Tuple
from lmms_eval.api.instance import Instance
from decord import VideoReader, cpu
import random
import subprocess
from collections import defaultdict
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from transformers import AutoModel, AutoTokenizer
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from lmms_eval.api.model import lmms
from tqdm import tqdm
import logging

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def add_noise_to_image(image, noise_std=25):
    """为图像添加高斯噪声"""
    # 确保图像是 numpy 数组，并且是正确的数据类型
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 确保图像是 3D 数组 (高度, 宽度, 通道)
    if image.ndim == 4:
        image = np.squeeze(image, axis=0)  # 移除第一个维度
    
    # 转换为浮点数以进行计算
    image = image.astype(np.float32)
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)

def detect_scenes(video_path):
    # 使用FFmpeg的scene detection功能
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-filter:v', 'select=\'gt(scene,0.3)\',showinfo',  # 0.3是场景变化的阈值,可以根据需要调整
        '-f', 'null',
        '-'
    ]
    
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    scene_changes = []
    
    for line in process.stderr:
        if "pts_time" in line:
            time = float(line.split("pts_time:")[1].split()[0])
            scene_changes.append(time)
    
    return scene_changes

def optimize_noise_sampling(video_path, frame_indices, num_frames_to_noise, fps):
    scene_changes = detect_scenes(video_path)
    
    # 将场景变化转换为帧索引
    scene_change_frames = [int(time * fps) for time in scene_changes]
    
    # 找出frame_indices中每个帧所属的场景
    scene_durations = defaultdict(int)
    frame_to_scene = {}
    current_scene = 0
    
    for frame in frame_indices:
        while current_scene < len(scene_change_frames) - 1 and frame >= scene_change_frames[current_scene + 1]:
            current_scene += 1
        
        frame_to_scene[frame] = current_scene
        scene_durations[current_scene] += 1
    
    # 按场景持续时间排序
    sorted_scenes = sorted(scene_durations.items(), key=lambda x: x[1], reverse=True)
    
    # 分配噪声帧到各个场景
    frames_to_noise = []
    remaining_frames = num_frames_to_noise
    for scene_index, duration in sorted_scenes:
        if remaining_frames <= 0:
            break
        
        # 为每个场景分配噪声帧，数量与场景中的采样帧数成正比
        frames_for_scene = min(int(duration * num_frames_to_noise / len(frame_indices)), remaining_frames)
        scene_frames = [frame for frame in frame_indices if frame_to_scene[frame] == scene_index]
        
        # 在场景内均匀分布噪声帧
        if frames_for_scene > 0 and scene_frames:
            noise_frames = np.linspace(0, len(scene_frames) - 1, frames_for_scene, dtype=int)
            frames_to_noise.extend([scene_frames[i] for i in noise_frames])
        
        remaining_frames -= frames_for_scene
    
    # 如果还有剩余帧，进行第二轮分配
    if remaining_frames > 0:
        available_frames = set(frame_indices) - set(frames_to_noise)
        additional_frames = np.random.choice(list(available_frames), min(remaining_frames, len(available_frames)), replace=False)
        frames_to_noise.extend(additional_frames)
    return sorted(frames_to_noise)

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, video_noise_strategy="random", video_noise_ratio=0.5, video_noise_std=1000):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    num_frames_to_noise = int(len(frame_indices) * video_noise_ratio)
    if video_noise_strategy == "random":
        frames_to_noise = random.sample(list(frame_indices), k=num_frames_to_noise)
    elif video_noise_strategy == "uniform":
        frames_to_noise = np.linspace(0, len(frame_indices)-1, num_frames_to_noise, dtype=int)
        frames_to_noise = frame_indices[frames_to_noise]
    elif video_noise_strategy == "scene_track":
        frames_to_noise = optimize_noise_sampling(video_path, frame_indices, num_frames_to_noise, fps)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        
        # 如果当前帧被选中添加噪声
        if frame_index in frames_to_noise:
            img = add_noise_to_image(np.array(img), video_noise_std)
        
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


from datetime import timedelta
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs


@register_model("internvl2")
class InternVL2(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2-2B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_segments: int = 8,
        video_noise_strategy: str = "random",
        video_noise_ratio: float = 0.0,
        video_noise_mean: float = 0.0,
        video_noise_std: float = 100.0,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self._model = AutoModel.from_pretrained(self.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality
        self.num_segments = num_segments
        
        self.video_noise_strategy = video_noise_strategy
        self.video_noise_ratio = video_noise_ratio
        self.video_noise_mean = video_noise_mean
        self.video_noise_std = video_noise_std

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                if visuals:
                    visuals = [load_image(visual).to(torch.bfloat16).cuda() for visual in visuals]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals]
                    image_tokens = ["<image>"] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patch_list = None
                response, history = self.model.chat(self.tokenizer, pixel_values, contexts, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                pixel_values, num_patches_list = load_video(video_path, num_segments=self.num_segments, max_num=1, video_noise_strategy=self.video_noise_strategy, video_noise_ratio=self.video_noise_ratio, video_noise_std=self.video_noise_std)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + contexts
                response, history = self.model.chat(self.tokenizer, pixel_values, question, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."
