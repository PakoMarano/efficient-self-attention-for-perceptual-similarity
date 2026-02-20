import json
import os
import zipfile

import torch
import torch.nn.functional as F
from torchvision import transforms
import peft
from peft import PeftModel, LoraConfig, get_peft_model

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .feature_extraction.extractor import ViTExtractor
from .config import dreamsim_args, dreamsim_weights
from .efficient_modules import validate_attention_module


class PerceptualModel(torch.nn.Module):
    def __init__(self, model_type: str = "dino_vitb16", feat_type: str = "cls", stride: int = 16,
                 load_dir: str = "./models", normalize_embeds: bool = False,
                 device: str = "cuda", attention_module: str = "benchmark"):
        """ Initializes a perceptual model that returns the perceptual distance between two image tensors.
        Extracts features from a DINO ViT-B/16 model.

        :param model_type: Base ViT model. Only 'dino_vitb16' is supported.
        :param feat_type: Which ViT feature to extract. Accepted values are:
            'cls': The CLS token
            'last_layer': The final layer tokens
            'cls_patch': The CLS token and global-pooled patch tokens, concatenated
        :param stride: Stride of first convolution layer (should match patch size, typically 16).
        :param load_dir: Path to pretrained ViT checkpoints.
        :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
        :param device: Device for model (e.g., 'cuda' or 'cpu').
        :param attention_module: Attention backend for ViT blocks. 'benchmark' keeps standard MHA.
        """
        super().__init__()
        assert model_type == "dino_vitb16", f"Only dino_vitb16 is supported, got {model_type}"
        
        self.model_type = model_type
        self.feat_type = feat_type
        self.stride = stride
        self.is_patch = feat_type == "cls_patch"
        self.normalize_embeds = normalize_embeds
        self.device = device
        self.attention_module = validate_attention_module(attention_module)
        
        # Initialize single extractor
        self.extractor = ViTExtractor(
            model_type,
            stride,
            load_dir,
            device=device,
            attention_module=self.attention_module,
        )
        
        # Get extraction function
        self.extract_fn = self._get_extract_fn(feat_type)

    def forward(self, img_a, img_b):
        """
        :param img_a: An RGB image passed as a (1, 3, 224, 224) tensor with values [0-1].
        :param img_b: Same as img_a.
        :return: A distance score for img_a and img_b. Higher means further/more different.
        """
        embed_a = self.embed(img_a)
        embed_b = self.embed(img_b)

        if self.feat_type == 'cls_patch':
            cls_a = embed_a[:, 0]
            patch_a = embed_a[:, 1:]
            cls_b = embed_b[:, 0]
            patch_b = embed_b[:, 1:]

            n = patch_a.shape[0]
            s = int(patch_a.shape[1] ** 0.5)
            patch_a_pooled = F.adaptive_avg_pool2d(patch_a.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)).squeeze()
            if len(patch_a_pooled.shape) == 1:
                patch_a_pooled = patch_a_pooled.unsqueeze(0)
            patch_b_pooled = F.adaptive_avg_pool2d(patch_b.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)).squeeze()
            if len(patch_b_pooled.shape) == 1:
                patch_b_pooled = patch_b_pooled.unsqueeze(0)

            embed_a = torch.cat((cls_a, patch_a_pooled), dim=-1)
            embed_b = torch.cat((cls_b, patch_b_pooled), dim=-1)
        return 1 - F.cosine_similarity(embed_a, embed_b, dim=-1)

    def embed(self, img):
        """
        Returns an embedding of img. The perceptual distance is the cosine distance between two embeddings. If the
        embeddings are normalized then L2 distance can also be used.
        """
        # Preprocess and extract features
        prep_img = self._preprocess(img)
        embed = self.extract_fn(prep_img).squeeze()
        
        # Ensure proper shape
        if len(embed.shape) <= 1:
            embed = embed.unsqueeze(0)
        if len(embed.shape) <= 2 and self.is_patch:
            embed = embed.unsqueeze(0)

        # Normalize if requested
        if self.normalize_embeds:
            embed = normalize_embedding_patch(embed) if self.is_patch else normalize_embedding(embed)
            
        return embed

    def _get_extract_fn(self, feat_type):
        """Get the extraction function for the specified feature type."""
        if feat_type == "cls":
            return self._extract_cls
        elif feat_type == "last_layer":
            return self._extract_last_layer
        elif feat_type == "cls_patch":
            return self._extract_cls_and_patch
        else:
            raise ValueError(f"Feature type {feat_type} is not supported for dino_vitb16. Use 'cls', 'last_layer', or 'cls_patch'.")

    def _extract_cls_and_patch(self, img):
        """Extract CLS token and patch tokens from layer 11."""
        layer = 11
        return self.extractor.extract_descriptors(img, layer)

    def _extract_cls(self, img):
        """Extract only the CLS token from layer 11."""
        layer = 11
        return self._extract_cls_and_patch(img)[:, :, :, 0, :]

    def _extract_last_layer(self, img):
        """Extract features from the last layer."""
        return self.extractor.forward(img, is_proj=False)

    def _preprocess(self, img):
        """Normalize image using ImageNet mean and std (DINO standard)."""
        return transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(img)


def download_weights(cache_dir, use_patch_model=False):
    """
    Downloads and unzips DreamSim weights for dino_vitb16.
    
    :param cache_dir: Directory to cache downloaded weights.
    :param use_patch_model: If True, downloads weights for cls_patch model, otherwise for cls model.
    """
    model_type = "dino_vitb16_patch" if use_patch_model else "dino_vitb16"
    
    dreamsim_required_ckpts = {
        "dino_vitb16": ["dino_vitb16_pretrain.pth", "dino_vitb16_single_lora"],
        "dino_vitb16_patch": ["dino_vitb16_pretrain.pth", "dino_vitb16_patch_lora"],
    }

    def check(path):
        for required_ckpt in dreamsim_required_ckpts[model_type]:
            if not os.path.exists(os.path.join(path, required_ckpt)):
                return False
        return True

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if check(cache_dir):
        print(f"Using cached {cache_dir}")
    else:
        print("Downloading checkpoint")
        torch.hub.download_url_to_file(url=dreamsim_weights[model_type],
                                       dst=os.path.join(cache_dir, "pretrained.zip"))
        print("Unzipping...")
        with zipfile.ZipFile(os.path.join(cache_dir, "pretrained.zip"), 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def dreamsim(pretrained: bool = True, device="cuda", cache_dir="./models", normalize_embeds: bool = True,
             use_patch_model: bool = False, attention_module: str = "benchmark"):
    """ Initializes the DreamSim model with dino_vitb16 backbone. When first called, downloads/caches model weights.

    :param pretrained: If True, downloads and loads DreamSim weights.
    :param device: Device for model (e.g., 'cuda' or 'cpu').
    :param cache_dir: Location for downloaded weights.
    :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
    :param use_patch_model: If True, uses model trained with CLS and patch features, otherwise just CLS.
    :param attention_module: Attention backend for ViT blocks. 'benchmark' keeps standard MHA.
    :return:
        - PerceptualModel with DreamSim settings and weights.
        - Preprocessing function that converts a PIL image to a (1, 3, 224, 224) tensor with values [0-1].
    """
    # Determine model configuration
    config_key = "dino_vitb16_patch" if use_patch_model else "dino_vitb16"
    
    # Download weights if needed
    download_weights(cache_dir=cache_dir, use_patch_model=use_patch_model)

    # Initialize PerceptualModel
    model = PerceptualModel(
        **dreamsim_args['model_config'][config_key], 
        device=device, 
        load_dir=cache_dir,
        normalize_embeds=normalize_embeds,
        attention_module=attention_module,
    )

    # Setup LoRA configuration
    lora_tag = 'dino_vitb16_patch_lora' if use_patch_model else 'dino_vitb16_single_lora'
    with open(os.path.join(cache_dir, lora_tag, 'adapter_config.json'), 'r') as f:
        adapter_config = json.load(f)
    lora_keys = ['r', 'lora_alpha', 'lora_dropout', 'bias', 'target_modules']
    lora_config = LoraConfig(**{k: adapter_config[k] for k in lora_keys})
    model = get_peft_model(model, lora_config)

    # Load pretrained weights if requested
    if pretrained:
        load_dir = os.path.join(cache_dir, lora_tag)
        model = PeftModel.from_pretrained(model.base_model.model, load_dir).to(device)

    model.eval().requires_grad_(False)

    # Define preprocessing function
    img_size = dreamsim_args['img_size']
    t = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    def preprocess(pil_img):
        pil_img = pil_img.convert('RGB')
        return t(pil_img).unsqueeze(0)

    return model, preprocess


def normalize_embedding(embed):
    embed = (embed.T - torch.mean(embed, dim=1)).T
    return (embed.T / torch.norm(embed, dim=1)).T
    
def normalize_embedding_patch(embed):
    mean_matrix = torch.mean(embed, dim=2).unsqueeze(1)
    embed = (embed.mT - mean_matrix).mT
    normed_matrix = torch.norm(embed, dim=2).unsqueeze(1)
    return (embed.mT / normed_matrix).mT
