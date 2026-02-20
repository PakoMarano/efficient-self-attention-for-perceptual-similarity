dreamsim_args = {
    "model_config": {
        "dino_vitb16": {
            "feat_type": 'cls',
            "model_type": "dino_vitb16",
            "stride": 16
        },
        "dino_vitb16_patch": {
            "feat_type": 'cls_patch',
            "model_type": "dino_vitb16",
            "stride": 16
        },
    },
    "img_size": 224
}

dreamsim_weights = {
    "dino_vitb16": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_dino_vitb16_checkpoint.zip",
    "dino_vitb16_patch": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.1-checkpoints/dreamsim_dino_vitb16_patch_checkpoint.zip",
}
