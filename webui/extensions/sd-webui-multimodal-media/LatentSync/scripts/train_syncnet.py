# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

from tqdm.auto import tqdm
import os, argparse, datetime, math
import logging
from omegaconf import OmegaConf
import shutil

from latentsync.data.syncnet_dataset import SyncNetDataset
from latentsync.models.stable_syncnet import StableSyncNet
from latentsync.models.wav2lip_syncnet import Wav2LipSyncNet
from latentsync.utils.util import gather_loss, plot_loss_chart
from accelerate.utils import set_seed

import torch
from diffusers import AutoencoderKL
from diffusers.utils.logging import get_logger
from einops import rearrange
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from latentsync.utils.util import init_dist, cosine_loss, dummy_context

logger = get_logger(__name__)


def main(config):
    # Initialize distributed training
    local_rank = init_dist()
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = config.run.seed + global_rank
    set_seed(seed)

    # Logging folder
    folder_name = "train" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/loss_charts", exist_ok=True)
        shutil.copy(config.config_path, output_dir)

    device = torch.device(local_rank)

    if config.data.latent_space:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        vae.requires_grad_(False)
        vae.to(device)
    else:
        vae = None

    # Dataset and Dataloader setup
    train_dataset = SyncNetDataset(config.data.train_data_dir, config.data.train_fileslist, config)
    val_dataset = SyncNetDataset(config.data.val_data_dir, config.data.val_fileslist, config)

    train_distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.run.seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        sampler=train_distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    num_samples_limit = 640

    val_batch_size = min(
        num_samples_limit // config.data.num_frames, config.data.batch_size
    )  # limit batch size to avoid CUDA OOM

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=val_dataset.worker_init_fn,
    )

    # Model
    syncnet = StableSyncNet(OmegaConf.to_container(config.model)).to(device)
    # syncnet = Wav2LipSyncNet().to(device)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, syncnet.parameters())), lr=config.optimizer.lr
    )

    global_step = 0
    train_step_list = []
    train_loss_list = []
    val_step_list = []
    val_loss_list = []

    if config.ckpt.resume_ckpt_path != "":
        if is_main_process:
            logger.info(f"Load checkpoint from: {config.ckpt.resume_ckpt_path}")
        ckpt = torch.load(config.ckpt.resume_ckpt_path, map_location=device, weights_only=True)

        syncnet.load_state_dict(ckpt["state_dict"])

        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
            train_step_list = ckpt["train_step_list"]
            train_loss_list = ckpt["train_loss_list"]
            val_step_list = ckpt["val_step_list"]
            val_loss_list = ckpt["val_loss_list"]

    # DDP wrapper
    syncnet = DDP(syncnet, device_ids=[local_rank], output_device=local_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel & distributed & accumulation) = {config.data.batch_size * num_processes * config.data.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")

    first_epoch = global_step // num_update_steps_per_epoch
    num_val_batches = config.data.num_val_samples // (num_processes * config.data.batch_size)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, config.run.max_train_steps), initial=global_step, desc="Steps", disable=not is_main_process
    )

    # Support mixed-precision training
    scaler = torch.amp.GradScaler("cuda") if config.run.mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        syncnet.train()
        step_loss = 0
        optimizer.zero_grad()

        for index, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            frames = batch["frames"].to(device, dtype=torch.float16)
            audio_samples = batch["audio_samples"].to(device, dtype=torch.float16)
            y = batch["y"].to(device, dtype=torch.float32)

            if config.data.latent_space:
                max_batch_size = (
                    num_samples_limit // config.data.num_frames
                )  # due to the limited cuda memory, we split the input frames into parts
                if frames.shape[0] > max_batch_size:
                    assert (
                        frames.shape[0] % max_batch_size == 0
                    ), f"max_batch_size {max_batch_size} should be divisible by batch_size {frames.shape[0]}"
                    frames_part_results = []
                    for i in range(0, frames.shape[0], max_batch_size):
                        frames_part = frames[i : i + max_batch_size]
                        frames_part = rearrange(frames_part, "b f c h w -> (b f) c h w")
                        with torch.no_grad():
                            frames_part = vae.encode(frames_part).latent_dist.sample() * 0.18215
                        frames_part_results.append(frames_part)
                    frames = torch.cat(frames_part_results, dim=0)
                else:
                    frames = rearrange(frames, "b f c h w -> (b f) c h w")
                    with torch.no_grad():
                        frames = vae.encode(frames).latent_dist.sample() * 0.18215

                frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=config.data.num_frames)
            else:
                frames = rearrange(frames, "b f c h w -> b (f c) h w")

            if config.data.lower_half:
                height = frames.shape[2]
                frames = frames[:, :, height // 2 :, :]

            # Disable gradient sync for the first N-1 steps, enable sync on the final step
            with syncnet.no_sync() if (index + 1) % config.data.gradient_accumulation_steps != 0 else dummy_context():
                # Mixed-precision training
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training
                ):
                    vision_embeds, audio_embeds = syncnet(frames, audio_samples)

                loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), y).mean()
                loss = loss / config.data.gradient_accumulation_steps

                # Backpropagate
                scaler.scale(loss).backward()

            step_loss += gather_loss(loss, device)

            # Update parameters when the accumulation steps are reached
            if (index + 1) % config.data.gradient_accumulation_steps == 0:
                """>>> gradient clipping >>>"""
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(syncnet.parameters(), config.optimizer.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                train_step_list.append(global_step)
                train_loss_list.append(step_loss)

                if is_main_process and global_step % config.run.validation_steps == 0:
                    logger.info(f"Validation at step {global_step}")
                    val_loss = validation(
                        val_dataloader,
                        device,
                        syncnet,
                        config.data.latent_space,
                        config.data.lower_half,
                        vae,
                        num_val_batches,
                    )
                    val_step_list.append(global_step)
                    val_loss_list.append(val_loss)
                    logger.info(f"Validation loss at step {global_step} is {val_loss:0.3f}")
                    plot_loss_chart(
                        os.path.join(output_dir, f"loss_charts/loss_chart-{global_step}.png"),
                        ("Train loss", train_step_list, train_loss_list),
                        ("Val loss", val_step_list, val_loss_list),
                    )

                if is_main_process and global_step % config.ckpt.save_ckpt_steps == 0:
                    checkpoint_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                    torch.save(
                        {
                            "state_dict": syncnet.module.state_dict(),  # to unwrap DDP
                            "global_step": global_step,
                            "train_step_list": train_step_list,
                            "train_loss_list": train_loss_list,
                            "val_step_list": val_step_list,
                            "val_loss_list": val_loss_list,
                        },
                        checkpoint_save_path,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_save_path}")

                progress_bar.set_postfix({"step_loss": step_loss, "epoch": epoch})
                step_loss = 0

                if global_step >= config.run.max_train_steps:
                    break

    progress_bar.close()
    dist.destroy_process_group()


@torch.no_grad()
def validation(val_dataloader, device, syncnet, latent_space, lower_half, vae, num_val_batches):
    syncnet.eval()

    losses = []
    val_step = 0
    while True:
        for index, batch in enumerate(val_dataloader):
            ### >>>> Validation >>>> ###

            frames = batch["frames"].to(device, dtype=torch.float16)
            audio_samples = batch["audio_samples"].to(device, dtype=torch.float16)
            y = batch["y"].to(device, dtype=torch.float32)

            if latent_space:
                num_frames = frames.shape[1]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
                frames = vae.encode(frames).latent_dist.sample() * 0.18215
                frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=num_frames)
            else:
                frames = rearrange(frames, "b f c h w -> b (f c) h w")

            if lower_half:
                height = frames.shape[2]
                frames = frames[:, :, height // 2 :, :]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                vision_embeds, audio_embeds = syncnet(frames, audio_samples)

            loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), y).mean()

            losses.append(loss.item())

            val_step += 1
            if val_step > num_val_batches:
                syncnet.train()
                if len(losses) == 0:
                    raise RuntimeError("No validation data")
                return sum(losses) / len(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to train the SyncNet")
    parser.add_argument("--config_path", type=str, default="configs/syncnet/syncnet_16_pixel.yaml")
    args = parser.parse_args()

    # Load a configuration file
    config = OmegaConf.load(args.config_path)
    config.config_path = args.config_path

    main(config)
