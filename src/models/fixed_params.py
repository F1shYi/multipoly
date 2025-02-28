CHORD_ENCODER_PARAMS_FIXED = {
    "input_dim":36,
    "hidden_dim":512,
    "z_dim":512
}

POLYFFUSION_PARAMS_FIXED = {
    "in_channels": 2,
    "out_channels": 2,
    "channels": 64,
    "attention_levels":[2,3],
    "n_res_blocks": 2,
    "channel_multipliers":[1,2,4,4],
    "n_heads": 4,
    "tf_layers": 1,
    "d_cond": 512
}

DIFFUSION_PARAMS_FIXED = {
    "linear_start": 0.00085,
    "linear_end": 0.012,
    "n_steps": 1000,
}
