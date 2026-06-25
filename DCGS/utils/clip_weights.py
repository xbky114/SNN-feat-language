import os
import torch


def load_clip_state_dict(checkpoint_path):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    with open(checkpoint_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location='cpu').eval()
            state_dict = model.state_dict()
        except RuntimeError:
            opened_file.seek(0)
            state_dict = torch.load(opened_file, map_location='cpu')

    if isinstance(state_dict, dict):
        for model_key in ['model', 'module']:
            if model_key in state_dict:
                state_dict = state_dict[model_key]
                break
    return state_dict


def extract_visual_state_dict(checkpoint_path):
    state_dict = load_clip_state_dict(checkpoint_path)
    visual_sd = {}
    for key, value in state_dict.items():
        if key.startswith('visual.'):
            visual_sd[key[len('visual.'):]] = value.float()
    return visual_sd


def load_clip_visual_weights(model, checkpoint_path, device):
    visual_sd = extract_visual_state_dict(checkpoint_path)
    model.load_state_dict(visual_sd)
    model.to(device)
    model.float()
    return model


def default_rn50_checkpoint():
    return os.path.expanduser('~/.cache/clip/RN50.pt')
