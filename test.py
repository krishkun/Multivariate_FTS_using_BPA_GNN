#!/usr/bin/env python3
"""Textual Inversion Training using EGGROLL Evolution Strategies with Quantization"""

import os

# Set GPU to use
gpu_id = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
from diffusers import DiffusionPipeline, DDPMScheduler, StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import safetensors
import wandb
import matplotlib.pyplot as plt
from optimum.quanto import quantize, freeze, qint8

# Configuration
config = {
    'pretrained_model': 'sd-local-sf',
    'data_dir': 'dataset/dog6',
    'output_dir': 'results_opt/1024_pertb_b16_high_rank_quant',
    'placeholder_token': 'sks',
    'initializer_token': 'dog',
    'resolution': 512,
    'train_batch_size': 16,
    'num_train_epochs': 50,
    'max_train_steps': 5000,
    'learning_rate': 1e-2,
    'lr_warmup_steps': 500,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': torch.float16,
    'wandb_project': 'Textual Inversion EGGROLL',
    'wandb_name': 'dog6-eggti-1024_pertb_b32_fullrez_quant',
    'wandb_notes': 'Improved Textual Inversion training with quantization and better personalization',
    'num_envs': 1024,
    'rank': 512,
    'initial_sigma': 0.15,
    'min_sigma': 0.01,
    'perturb_batch_size': 1024,
}

print(f"Using device: {config['device']}")
print(f"Dataset directory: {config['data_dir']}")
print(f"Output directory: {config['output_dir']}")
print(f"Placeholder token: {config['placeholder_token']}")

# Initialize Weights & Biases with error handling
try:
    os.environ['WANDB_NOTEBOOK_MODE'] = 'FALSE'
    wandb.init(
        project=config['wandb_project'],
        config=config,
        name=config['wandb_name'],
        notes=config['wandb_notes'],
        reinit="return_previous",
    )
    print(f"✓ Weights & Biases initialized: {wandb.run.url}")
except Exception as e:
    print(f"⚠ Warning: wandb initialization had an issue: {e}")
    print("Continuing without wandb tracking...")

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)

# Dataset templates for prompts
imagenet_templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of a {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
]

class TextualInversionDataset(Dataset):
    """Custom dataset for textual inversion with CPU preprocessing (GPU transforms in main process)"""
    
    def __init__(self, data_root, tokenizer, placeholder_token, size=512, 
                 flip_p=0.5, center_crop=False, repeats=100):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.flip_p = flip_p
        self.center_crop = center_crop
        
        # Get image files
        self.image_paths = [
            os.path.join(self.data_root, file_path) 
            for file_path in os.listdir(self.data_root)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        ]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats
        
        # CPU transformations - resize on CPU to ensure consistent batch sizes
        self.cpu_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # PIL -> Tensor [0,1], shape (C, H, W)
        ])
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx % self.num_images]
        image = Image.open(image_path)
        
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Center crop on CPU if needed
        if self.center_crop:
            w, h = image.size
            crop = min(w, h)
            left = (w - crop) // 2
            top = (h - crop) // 2
            right = left + crop
            bottom = top + crop
            image = image.crop((left, top, right, bottom))
        
        # Apply CPU transformations (resize + to tensor)
        image_tensor = self.cpu_transforms(image)  # Shape: (C, H, W), range [0, 1], already resized
        
        # Return as CPU tensor - GPU preprocessing (color jitter, flip, normalize) will be done in main process
        image_tensor = image_tensor.float()
        
        # Create text prompt
        text = random.choice(imagenet_templates).format(self.placeholder_token)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        return {"pixel_values": image_tensor, "input_ids": input_ids}

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model'], subfolder="tokenizer")

# Create dataset and dataloader
train_dataset = TextualInversionDataset(
    data_root=config['data_dir'],
    tokenizer=tokenizer,
    placeholder_token=config['placeholder_token'],
    size=config['resolution'],
    repeats=100,  # More repeats for better coverage
    flip_p=0.5,
    center_crop=False,
)

# Create GPU transformations for preprocessing in main process (no resize needed, done in CPU)
gpu_color_jitter = transforms.ColorJitter(
    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['train_batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,  # Pre-fetch 2 batches per worker for better GPU utilization
)

print(f"✓ Dataset created: {len(train_dataset)} samples")
print(f"✓ Number of reference images: {train_dataset.num_images}")
print(f"✓ DataLoader created with batch size: {config['train_batch_size']}")

# Load models from Stable Diffusion v1.5 with fp16 precision
print(f"Loading Stable Diffusion v1.5 models with {config['dtype']} precision...")

dtype = config['dtype']

# Load text encoder
text_encoder = CLIPTextModel.from_pretrained(
    config['pretrained_model'], subfolder="text_encoder", torch_dtype=dtype
).to(config['device'])

# Load VAE
vae = AutoencoderKL.from_pretrained(
    config['pretrained_model'], subfolder="vae", torch_dtype=dtype
).to(config['device'])

# Load UNet
unet = UNet2DConditionModel.from_pretrained(
    config['pretrained_model'], subfolder="unet", torch_dtype=dtype
).to(config['device'])

# Load noise scheduler
noise_scheduler = DDPMScheduler.from_pretrained(
    config['pretrained_model'], subfolder="scheduler"
)

print("✓ Models loaded successfully with fp16 precision")

# Add placeholder token to tokenizer
placeholder_tokens = [config['placeholder_token']]
num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

if num_added_tokens != 1:
    raise ValueError(f"The tokenizer already contains the token {config['placeholder_token']}")

# Get token IDs
initializer_token_id = tokenizer.encode(config['initializer_token'], add_special_tokens=False)[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(config['placeholder_token'])

# Resize token embeddings
text_encoder.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

print(f"✓ Added placeholder token: {config['placeholder_token']}")
print(f"✓ Initialized with token: {config['initializer_token']}")

# Freeze VAE and UNet
vae.requires_grad_(False)
unet.requires_grad_(False)

# Freeze text encoder except for token embeddings
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

print("\n" + "="*60)
print("Starting Textual Inversion Training with EGGROLL")
print("="*60)
num_envs = config['num_envs']
rank = config['rank']
learning_rate = config['learning_rate']
initial_sigma = config['initial_sigma']
min_sigma = config['min_sigma']
num_epochs = config['num_train_epochs']
perturb_batch_size = config['perturb_batch_size']

# Setup text encoder and models in eval mode except embeddings
text_encoder.train()
vae.eval()
unet.eval()

# Get the embedding dimension
embedding_dim = text_encoder.get_input_embeddings().weight.shape[1]

# =============================================================================
# Quantize and freeze modules
# =============================================================================
print("Quantizing models with qint8...")
quantize(text_encoder, weights=qint8)
quantize(unet, weights=qint8)
quantize(vae, weights=qint8)

print("Freezing quantized models...")
freeze(text_encoder)
freeze(unet)
freeze(vae)

print("Converting text_encoder to half precision...")
text_encoder.half()

# Compile models
print("Compiling models with torch.compile...")
unet = torch.compile(unet, dynamic=False, backend='eager')
vae = torch.compile(vae, dynamic=False, backend='eager')
text_encoder = torch.compile(text_encoder, dynamic=False, backend='eager')

torch.cuda.empty_cache()
print("✓ Models quantized, frozen, and compiled successfully")

# Keep original embeddings as reference
orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()
base_embeds = orig_embeds_params[placeholder_token_id].clone()

def generate_lora_perturbations(num_perturbs, embedding_dim, rank, sigma, seed=None, device=config['device'], dtype=torch.float16):
    """Generate low-rank perturbations using vectorized matrix operations"""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Generate A: (embedding_dim, rank)
    A = torch.randn(embedding_dim, rank, device=device, dtype=dtype)
    
    # Generate B: (rank, num_perturbs // 2)
    B = torch.randn(rank, num_perturbs // 2, device=device, dtype=dtype)
    
    # Compute perturbations: A @ B -> shape: (embedding_dim, num_perturbs // 2)
    pos_perturbs = (A @ B) * sigma
    
    # Create opposite sign perturbations
    neg_perturbs = -pos_perturbs
    
    # Stack positive and negative: (embedding_dim, num_perturbs)
    all_perturbs = torch.cat([pos_perturbs, neg_perturbs], dim=1)
    
    # Transpose to get (num_perturbs, embedding_dim)
    return all_perturbs.T

# Loss history
losses = []
weight_dtype = config['dtype']

print(f"Embedding dimension: {embedding_dim}")
print(f"Number of perturbations per epoch: {num_envs}")
print(f"Rank of perturbations: {rank}")
print(f"Initial perturbation strength (sigma): {initial_sigma}")
print(f"Minimum sigma: {min_sigma}")
print(f"Number of epochs: {num_epochs}")

# Log EGGROLL hyperparameters to wandb
try:
    wandb.config.update({
        'num_envs': config['num_envs'],
        'rank': config['rank'],
        'initial_sigma': config['initial_sigma'],
        'min_sigma': config['min_sigma'],
        'learning_rate_eggroll': config['learning_rate'],
        'num_train_epochs': config['num_train_epochs'],
        'perturb_batch_size': config['perturb_batch_size'],
        'embedding_dim': embedding_dim,
        'dtype': 'fp16',
        'improved': True,
        'resolution': config['resolution'],
        'CUDA_VISIBLE_DEVICES': gpu_id,
        'quantization': 'qint8',
        'compiled': True,
    })
except:
    pass

progress_bar = tqdm(total=num_epochs * len(train_dataloader), desc="Training with Improved EGGROLL (Quantized)")

sigma = initial_sigma

for epoch in range(num_epochs):
    epoch_losses = []
    
    # Generate EGGROLL perturbations for this epoch (GPU-optimized)
    perturbations = generate_lora_perturbations(
        num_envs, 
        embedding_dim, 
        rank, 
        sigma, 
        seed=epoch,
        device=config['device'],
        dtype=weight_dtype
    )
    
    for step, batch in enumerate(train_dataloader):
        # Move batch to device with non-blocking transfer for better performance
        pixel_values = batch["pixel_values"].to(config['device'], dtype=torch.float32, non_blocking=True)
        input_ids = batch["input_ids"].to(config['device'], non_blocking=True)
        
        # Apply GPU-accelerated preprocessing in main process
        # Resize is already done on CPU, so we only do augmentations here
        pixel_values = gpu_color_jitter(pixel_values)  # Color jitter on GPU
        
        # Random horizontal flip on GPU
        if random.random() < 0.5:
            pixel_values = torch.flip(pixel_values, [-1])
        
        # Normalize from [0, 1] to [-1, 1] on GPU
        pixel_values = pixel_values * 2.0 - 1.0
        
        # Convert to target dtype
        pixel_values = pixel_values.to(dtype=weight_dtype)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Sample noise for diffusion
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        #currently training for the full timestep
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Evaluate each perturbation
        raw_fitnesses = torch.empty(num_envs, device=config['device'], dtype=weight_dtype)

        for pert_start in range(0, num_envs, perturb_batch_size):
            pert_end = min(pert_start + perturb_batch_size, num_envs)
            num_pert_current = pert_end - pert_start

            with torch.no_grad():
                perturbed_embeds_stack = base_embeds.unsqueeze(0) + perturbations[pert_start:pert_end]

                for i in range(num_pert_current):
                    text_encoder.get_input_embeddings().weight[placeholder_token_id] = perturbed_embeds_stack[i]
                    encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    perturb_loss = F.mse_loss(model_pred.float(), target.float())
                    raw_fitnesses[pert_start + i] = -perturb_loss.to(weight_dtype)
        
        # Normalize fitnesses
        normalized_fitnesses = (raw_fitnesses - raw_fitnesses.mean()) / (raw_fitnesses.std() + 1e-8)

        # EGGROLL-style update
        update = (normalized_fitnesses.unsqueeze(-1) * perturbations).sum(0) / num_envs
        base_embeds = base_embeds + learning_rate * update
        
        # Update text encoder with new embeddings
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[placeholder_token_id] = base_embeds.to(config['device'], dtype=weight_dtype)
        
        # Restore other embeddings
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[placeholder_token_id] = False
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[index_no_updates] = (
                orig_embeds_params[index_no_updates]
            )
        
        avg_fitness = raw_fitnesses.mean().item()
        epoch_losses.append(avg_fitness)
        losses.append(avg_fitness)
        
        progress_bar.update(1)
        progress_bar.set_postfix({"fitness": avg_fitness, "epoch": epoch, "sigma": sigma})
        
        # Log to wandb
        try:
            global_step = epoch * len(train_dataloader) + step
            wandb.log({
                'fitness': avg_fitness,
                'epoch': epoch,
                'step': global_step,
                'sigma': sigma,
                'min_fitness_batch': raw_fitnesses.min().item(),
                'max_fitness_batch': raw_fitnesses.max().item(),
            }, step=global_step)
        except:
            pass
        
        # Save checkpoint periodically
        if global_step % 100 == 0 and global_step > 0:
            learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]
            learned_embeds_dict = {config['placeholder_token']: learned_embeds.detach().cpu()}
            
            checkpoint_path = os.path.join(config['output_dir'], f"learned_embeds-eggroll-{global_step}.safetensors")
            safetensors.torch.save_file(learned_embeds_dict, checkpoint_path)
            
            try:
                wandb.save(checkpoint_path)
            except:
                pass
    
    avg_epoch_fitness = np.mean(epoch_losses)
    print(f"\nEpoch {epoch+1}/{num_epochs}: Avg Fitness (neg-loss) = {avg_epoch_fitness:.6f}, Sigma = {sigma:.4f}")
    
    # Log epoch summary to wandb
    try:
        wandb.log({
            'epoch_avg_fitness': avg_epoch_fitness,
            'epoch': epoch,
            'sigma': sigma,
        }, step=(epoch + 1) * len(train_dataloader))
    except:
        pass
    
    # Adaptive sigma decay
    progress = epoch / num_epochs
    sigma = max(min_sigma, initial_sigma * (1 - progress * 0.8))

progress_bar.close()

# Save final embeddings
learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]
learned_embeds_dict = {config['placeholder_token']: learned_embeds.detach().cpu()}
final_path = os.path.join(config['output_dir'], "learned_embeds.safetensors")
safetensors.torch.save_file(learned_embeds_dict, final_path)

print("\n" + "="*60)
print("✓ Training completed!")
print(f"✓ Final embeddings saved to: {final_path}")
print(f"✓ Best fitness: {max(losses):.6f}, Final fitness: {losses[-1]:.6f}")
print("="*60)

# Log final metrics to wandb
try:
    wandb.log({
        'best_fitness': max(losses),
        'final_fitness': losses[-1],
        'final_path': final_path,
    })
except:
    pass

# Visualize training fitness curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Fitness')
plt.xlabel('Batch Step')
plt.ylabel('Fitness (Neg-Loss)')
plt.title('Fitness Progression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
window = min(20, len(losses) // 5)
if window > 1:
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(smoothed, label='Smoothed Fitness')
else:
    plt.plot(losses, label='Fitness')
plt.xlabel('Batch Step')
plt.ylabel('Fitness (Neg-Loss)')
plt.title('Smoothed Fitness')
plt.legend()
plt.grid(True)

plt.tight_layout()
fitness_plot_path = os.path.join(config['output_dir'], 'training_fitness.png')
plt.savefig(fitness_plot_path)
plt.show()

print(f"Initial fitness: {losses[0]:.6f}")
print(f"Final fitness: {losses[-1]:.6f}")
print(f"Fitness improvement: {losses[-1] - losses[0]:.6f} (higher is better)")
print(f"Average fitness: {np.mean(losses):.6f}")

# Inference: Generate images with the trained textual inversion
print("\nLoading trained embeddings for inference...")

# Load the trained embeddings
learned_embeds_dict = safetensors.torch.load_file(final_path)
learned_embeds = learned_embeds_dict[config['placeholder_token']].to(config['device'])

# Load embeddings into text encoder
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[placeholder_token_id] = learned_embeds

# Create the pipeline for inference
pipeline = StableDiffusionPipeline.from_pretrained(
    config['pretrained_model'],
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    vae=vae,
    torch_dtype=weight_dtype,
).to(config['device'])

# Test prompts using the learned token
test_prompts = [
    f"a photo of {config['placeholder_token']}",
    f"a painting in the style of {config['placeholder_token']}",
    f"a close-up photo of {config['placeholder_token']}",
]

print("\n" + "="*60)
print("Generating test images with trained textual inversion...")
print("="*60)

with torch.no_grad():
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {idx}: {prompt}")
        image = pipeline(prompt, num_inference_steps=50).images[0]
        
        # Save generated image
        save_path = os.path.join(config['output_dir'], f"generated_{idx}.png")
        image.save(save_path)
        print(f"✓ Saved to {save_path}")
        
        # Log generated image to wandb
        try:
            wandb.log({
                f"generated_image_{idx}": wandb.Image(image, caption=prompt),
                f"prompt_{idx}": prompt,
            })
        except:
            pass
        
        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(prompt)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

print("\n✓ Complete!")

# Finish wandb run
try:
    wandb.finish()
except:
    pass
