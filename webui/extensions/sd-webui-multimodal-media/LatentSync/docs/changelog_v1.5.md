# LatentSync 1.5

## What's new in LatentSync 1.5?

1. Add temporal layer: our previous claim that the [temporal layer](https://arxiv.org/abs/2307.04725) severely impairs lip-sync accuracy was incorrect; the issue was actually caused by a bug in the code implementation. We have corrected our [paper](https://arxiv.org/abs/2412.09262) and updated the code. After incorporating the temporal layer, LatentSync 1.5 demonstrates significantly improved temporal consistency compared to version 1.0.

2. Improves performance on Chinese videos: many issues reported poor performance on Chinese videos, so we added Chinese data to the training of the new model version.

3. Reduce the VRAM requirement of the stage2 training to **20 GB** through the following optimizations:

   1. Implement gradient checkpointing in U-Net, VAE, SyncNet and VideoMAE
   2. Replace xFormers with PyTorch's native implementation of FlashAttention-2.
   3. Clear the CUDA cache after loading checkpoints.
   4. The stage2 training only requires training the temporal layer and audio cross-attention layer, which significantly reduces VRAM requirement compared to the previous full-parameter fine-tuning.

   Now you can train LatentSync on a single **RTX 3090**! Start the stage2 training with `configs/unet/stage2_efficient.yaml`.

4. Other code optimizations:

   1. Remove the dependency on xFormers and Triton.
   2. Upgrade the diffusers version to `0.32.2`.

## LatentSync 1.5 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/b0c8d1da-3fdc-4946-9800-1b2fd0ef9c7f controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/25dd1733-44c7-42fe-805a-d612d4bc30e0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/4e48e501-64b4-4b4f-a69c-ed18dd987b1f controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/e690d91b-9fe5-4323-a60e-2b7f546f01bc controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/e84e2c13-1deb-41f7-8382-048ba1922b71 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/5a5ba09f-590b-4eb3-8dfb-a199d8d1e276 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/11e4b2b6-64f4-4617-b005-059209fcaea5 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/38437475-3c90-4d08-b540-c8e819e93e0d controls preload></video>
    </td>
  </tr>
</table>