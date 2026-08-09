import torch
import os
import soundfile as sf
from diffusers.models import AutoencoderOobleck
from tqdm import tqdm
import torch.nn.functional as F

def process_audio(audio_path, target_sr=48000):
    try:
        # Load audio using soundfile
        audio_np, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to torch: [samples, channels] or [samples] -> [channels, samples]
        if audio_np.ndim == 1:
            audio = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            audio = torch.from_numpy(audio_np.T)
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        
        audio = audio[:2]
        
        # Resample if needed
        if sr != target_sr:
            ratio = target_sr / sr
            new_length = int(audio.shape[-1] * ratio)
            audio = F.interpolate(audio.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio.unsqueeze(0) # Add batch dim: [1, 2, samples]
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    print("Initializing Calibration Data Preparation...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "quant_data")
    output_path = os.path.join(project_root, "data", "calibration_latents.pt")
    vae_path = os.path.join(project_root, "checkpoints", "vae")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    print(f"Loading VAE from {vae_path}...")
    try:
        vae = AutoencoderOobleck.from_pretrained(vae_path)
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check for XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    
    print(f"Using device: {device}")
    vae = vae.to(device)
    vae.eval()

    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.flac')]
    print(f"Found {len(audio_files)} audio files.")
    
    all_chunks = []
    chunk_size = 512 # Latent frames
    samples_per_latent = 1920
    audio_chunk_size = chunk_size * samples_per_latent
    
    pbar = tqdm(audio_files, desc="Processing audio")
    for audio_file in pbar:
        file_path = os.path.join(data_dir, audio_file)
        full_audio = process_audio(file_path)
        
        if full_audio is None:
            continue
            
        # Split audio into chunks
        num_samples = full_audio.shape[-1]
        
        for start_idx in range(0, num_samples, audio_chunk_size):
            end_idx = start_idx + audio_chunk_size
            if end_idx > num_samples:
                break # Skip incomplete chunks
                
            audio_input = full_audio[:, :, start_idx:end_idx].to(device)
            
            try:
                with torch.no_grad():
                    # Encode
                    # VAE encode expects [Batch, Channels, Samples]
                    # Returns DiagonalGaussianDistribution
                    posterior = vae.encode(audio_input).latent_dist
                    latents = posterior.sample() # [1, 64, LatentLength]
                    
                    # It should be exactly chunk_size, but let's be safe
                    if latents.shape[-1] >= chunk_size:
                        all_chunks.append(latents[:, :, :chunk_size].cpu())
                    
                    pbar.set_postfix({"chunks": len(all_chunks)})
                    
            except Exception as e:
                print(f"Error encoding chunk {start_idx}-{end_idx} of {audio_file}: {e}")
                torch.cuda.empty_cache()
                if device == "xpu":
                    torch.xpu.empty_cache()
    
    print(f"Collected {len(all_chunks)} chunks of size {chunk_size}.")
    
    if len(all_chunks) > 0:
        print(f"Saving to {output_path}...")
        torch.save(all_chunks, output_path)
        print("Done.")
    else:
        print("No chunks collected.")

if __name__ == "__main__":
    main()
