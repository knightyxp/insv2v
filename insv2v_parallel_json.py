import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import imageio
import numpy as np
import torch
import torch.distributed as dist

from misc_utils.train_utils import unit_test_create_model
from pl_trainer.inference.inference import (
    InferenceIP2PVideo,
    InferenceIP2PVideoOpticalFlow,
)


def split_batch(cond: torch.Tensor, frames_in_batch: int = 16, num_ref_frames: int = 4):
    frames_in_following_batch = frames_in_batch - num_ref_frames
    conds = [cond[:, :frames_in_batch]]
    frame_ptr = frames_in_batch
    num_ref_frames_each_batch: List[int] = []

    while frame_ptr < cond.shape[1]:
        remaining_frames = cond.shape[1] - frame_ptr
        if remaining_frames < frames_in_batch:
            frames_in_following_batch = remaining_frames
        else:
            frames_in_following_batch = frames_in_batch - num_ref_frames
        this_ref_frames = frames_in_batch - frames_in_following_batch
        conds.append(cond[:, frame_ptr:frame_ptr + frames_in_following_batch])
        frame_ptr += frames_in_following_batch
        num_ref_frames_each_batch.append(this_ref_frames)

    return conds, num_ref_frames_each_batch


def derive_ground_instruction(edit_instruction_text: str) -> str:
    s = (edit_instruction_text or "").strip()
    if s.endswith("."):
        s = s[:-1]
    lower = s.lower()
    prefixes = [
        "remove ",
        "delete ",
        "erase ",
        "eliminate ",
        "add ",
        "make ",
        "ground ",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def build_prompt(item: Dict[str, Any], reasoning: bool) -> Tuple[str, str]:
    edit_text = item.get(
        "qwen_vl_72b_refined_instruction",
        item.get("edit_instruction", item.get("text", "")),
    )
    if reasoning:
        ground_instr = derive_ground_instruction(edit_text)
        prompt = (
            "A video sequence showing three parts: first the original scene, "
            f"then grounded {ground_instr}, and finally the same scene but {edit_text}"
        )
    else:
        prompt = (
            "A video sequence showing two parts: the first half shows the original scene, "
            f"and the second half shows the same scene but {edit_text}"
        )
    # allow explicit prompt override
    prompt = item.get("prompt", prompt)
    return prompt, edit_text


def tensor_to_video_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().contiguous()
    tensor = tensor.permute(0, 2, 3, 1)  # [T, H, W, C]
    tensor = ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def write_video(frames: np.ndarray, file_path: str, fps: int):
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    writer = imageio.get_writer(
        file_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_video_tensor(tensor: torch.Tensor, file_path: str, fps: int):
    frames = tensor_to_video_numpy(tensor)
    write_video(frames, file_path, fps)


def save_side_by_side(
    input_tensor: torch.Tensor,
    sample_tensor: torch.Tensor,
    file_path: str,
    fps: int,
):
    input_frames = tensor_to_video_numpy(input_tensor)
    sample_frames = tensor_to_video_numpy(sample_tensor)
    T = min(len(input_frames), len(sample_frames))
    if T == 0:
        raise ValueError("Empty video tensors provided.")
    H = min(input_frames.shape[1], sample_frames.shape[1])
    W = min(input_frames.shape[2], sample_frames.shape[2])
    combined_frames = []
    for i in range(T):
        left = input_frames[i][:H, :W]
        right = sample_frames[i][:H, :W]
        combined = np.concatenate([left, right], axis=1)
        combined_frames.append(combined)
    write_video(np.stack(combined_frames), file_path, fps)


def load_video_frames(
    video_path: str,
    num_frames: int,
) -> Tuple[torch.Tensor, int, int, int]:
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = int(meta.get("fps", 8))
    except Exception:
        fps = 8
    try:
        total_frames = reader.count_frames()
        if total_frames <= 0:
            total_frames = None
    except Exception:
        total_frames = None

    frames: List[np.ndarray] = []
    if total_frames is not None:
        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        for idx in indices:
            frame = reader.get_data(int(idx))
            frames.append(frame)
    else:
        for frame in reader:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
    reader.close()

    if not frames:
        raise ValueError(f"Unable to load frames from video: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    frames = frames[:num_frames]
    original_height, original_width = frames[0].shape[0], frames[0].shape[1]

    video = torch.from_numpy(np.stack(frames)).float()  # [T, H, W, C]
    video = video.permute(0, 3, 1, 2) / 255.0
    video = video * 2.0 - 1.0
    return video, original_height, original_width, fps


def resolve_item_basename(item: Dict[str, Any]) -> str:
    if "task_type" in item and "sample_id" in item:
        return f"{item['task_type']}_{item['sample_id']}"
    video_path = item.get("source_video_path", "sample")
    return os.path.splitext(os.path.basename(video_path))[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed instruct-video-to-video inference from JSON list."
    )
    parser.add_argument("--test_json", type=str, required=True, help="Path to JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/instruct_v2v.yaml",
        help="Model config path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--text_cfg", type=float, default=7.5, help="Text guidance.")
    parser.add_argument("--video_cfg", type=float, default=1.8, help="Image guidance.")
    parser.add_argument("--num_frames", type=int, default=33, help="Frames per video.")
    parser.add_argument(
        "--frames_in_batch",
        type=int,
        default=16,
        help="Frames processed per clip (including ref frames).",
    )
    parser.add_argument(
        "--num_ref_frames",
        type=int,
        default=4,
        help="Number of reference frames carried between clips.",
    )
    parser.add_argument(
        "--with_optical_flow",
        action="store_true",
        help="Enable motion compensation with optical flow.",
    )
    parser.add_argument(
        "--videoedit_reasoning",
        action="store_true",
        help="Use reasoning-style prompt template.",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=8,
        help="FPS for saved videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate outputs even if they already exist.",
    )
    return parser.parse_args()


def distribute_items(items: List[Dict[str, Any]], overwrite: bool, output_dir: str):
    filtered: List[Dict[str, Any]] = []
    for item in items:
        base = resolve_item_basename(item)
        output_video_path = os.path.join(output_dir, f"{base}_gen.mp4")
        if not overwrite and os.path.exists(output_video_path):
            continue
        filtered.append(item)
    return filtered


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Running inference with {world_size} GPUs")
        print(f"Loading tasks from {args.test_json}")

    with open(args.test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    items = distribute_items(items, args.overwrite, args.output_dir)
    if rank == 0:
        print(f"Total tasks after filtering: {len(items)}")

    subset = items[rank::world_size]
    print(f"[GPU {rank}] Processing {len(subset)} items")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    diffusion_model = unit_test_create_model(args.config_path)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt = {k.replace("_forward_module.", ""): v for k, v in ckpt.items()}
    diffusion_model.load_state_dict(ckpt, strict=False)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    if args.with_optical_flow:
        inf_pipe = InferenceIP2PVideoOpticalFlow(
            unet=diffusion_model.unet,
            num_ddim_steps=20,
            scheduler="ddpm",
        )
    else:
        inf_pipe = InferenceIP2PVideo(
            unet=diffusion_model.unet,
            num_ddim_steps=20,
            scheduler="ddpm",
        )

    text_uncond = diffusion_model.encode_text([""])
    if isinstance(text_uncond, torch.Tensor):
        text_uncond = text_uncond.to(device)

    for item in subset:
        base = resolve_item_basename(item)
        output_video_path = os.path.join(args.output_dir, f"{base}_gen.mp4")
        input_path = os.path.join(args.output_dir, f"{base}_input.mp4")
        compare_path = os.path.join(args.output_dir, f"{base}_compare.mp4")
        info_path = os.path.join(args.output_dir, f"{base}_gen_info.txt")
        os.makedirs(args.output_dir, exist_ok=True)

        video_path = item.get("source_video_path")
        if video_path is None:
            print(f"[GPU {rank}] Skipping {base}: missing source_video_path")
            continue
        if not os.path.exists(video_path):
            print(f"[GPU {rank}] Skipping {base}: video path not found {video_path}")
            continue

        prompt, edit_text = build_prompt(item, args.videoedit_reasoning)

        print(f"[GPU {rank}] Processing {base} ({video_path})")
        try:
            video_tensor, original_height, original_width, fps = load_video_frames(
                video_path,
                args.num_frames,
            )
        except Exception as exc:
            print(f"[GPU {rank}] Failed to load video {video_path}: {exc}")
            continue

        # override output fps with detected FPS if available
        output_fps = fps if fps > 0 else args.output_fps
        video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]

        cond = diffusion_model.encode_image_to_latent(video_tensor) / 0.18215
        text_cond = diffusion_model.encode_text(prompt)
        if isinstance(text_cond, torch.Tensor):
            text_cond = text_cond.to(device)

        conds, num_ref_frames_each_batch = split_batch(
            cond,
            frames_in_batch=args.frames_in_batch,
            num_ref_frames=args.num_ref_frames,
        )
        splitted_frames, _ = split_batch(
            video_tensor,
            frames_in_batch=args.frames_in_batch,
            num_ref_frames=args.num_ref_frames,
        )

        cond1 = conds[0]
        latent_pred_list: List[torch.Tensor] = []
        init_latent = torch.randn_like(cond1)
        latent_pred = inf_pipe(
            latent=init_latent,
            text_cond=text_cond,
            text_uncond=text_uncond,
            img_cond=cond1,
            text_cfg=args.text_cfg,
            img_cfg=args.video_cfg,
        )["latent"]
        latent_pred_list.append(latent_pred)

        for prev_cond, cond_, prev_frame, curr_frame, num_ref_frames_ in zip(
            conds[:-1],
            conds[1:],
            splitted_frames[:-1],
            splitted_frames[1:],
            num_ref_frames_each_batch,
        ):
            init_latent = torch.cat(
                [init_latent[:, -num_ref_frames_:], torch.randn_like(cond_)],
                dim=1,
            )
            cond_ = torch.cat([prev_cond[:, -num_ref_frames_:], cond_], dim=1)
            additional_kwargs: Dict[str, Any] = {}
            if args.with_optical_flow:
                ref_images = prev_frame[:, -num_ref_frames_:]
                query_images = curr_frame
                additional_kwargs = {
                    "ref_images": ref_images,
                    "query_images": query_images,
                }
            latent_pred = inf_pipe.second_clip_forward(
                latent=init_latent,
                text_cond=text_cond,
                text_uncond=text_uncond,
                img_cond=cond_,
                latent_ref=latent_pred[:, -num_ref_frames_:],
                noise_correct_step=0.5,
                text_cfg=args.text_cfg,
                img_cfg=args.video_cfg,
                **additional_kwargs,
            )["latent"]
            latent_pred_list.append(latent_pred[:, num_ref_frames_:])

        latent_pred = torch.cat(latent_pred_list, dim=1)
        image_pred = diffusion_model.decode_latent_to_image(latent_pred).clip(-1, 1)

        original_images = video_tensor.detach().cpu()
        transferred_images = image_pred.float().cpu()

        try:
            save_video_tensor(original_images, input_path, output_fps)
            save_video_tensor(transferred_images, output_video_path, output_fps)
            save_side_by_side(original_images, transferred_images, compare_path, output_fps)
        except Exception as exc:
            print(f"[GPU {rank}] Failed to save results for {base}: {exc}")
            continue

        with open(info_path, "w", encoding="utf-8") as info_file:
            info_file.write(prompt)

        print(
            f"[GPU {rank}] Completed {base} "
            f"(frames={transferred_images.shape[1]}, size={original_width}x{original_height})"
        )
        torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        print("Inference finished for all items.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

