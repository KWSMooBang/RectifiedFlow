import os
import logging
import torch
import tensorflow as tf


def load_checkpoint(checkpoint_dir, state, device):
    if not tf.io.gfile.exists(checkpoint_dir):
        tf.io.gfile.makedirs(os.path.dirname(checkpoint_dir))
        logging.warning(f"No checkpoint found at {checkpoint_dir}. Returned the same state as input")
    else:
        loaded_state = torch.load(checkpoint_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state
    

def save_checkpoint(checkpoint_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, checkpoint_dir)
        