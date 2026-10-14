"""Developer-only numerical fixtures; requires the pinned reference mlx-audio fork.

Runtime inference does not use Python. Run with mlx-audio commit
 e7488935ec2ebfafd38ef2b58e290d79f3d05dc7 to regenerate tiny synthetic weights.
Never reads, writes, or removes downloaded model checkpoints.
"""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_audio.tts.models.fish_omni.config import ModelConfig
from mlx_audio.tts.models.fish_omni.fish_omni import Model

root = Path(__file__).resolve().parents[1] / 'Tests/SayItBackendTests/Fixtures/Drama3'
root.mkdir(parents=True, exist_ok=True)
config = {
    'model_type': 'fish_omni', 'eos_token_id': 31,
    'semantic_start_token_id': 32, 'semantic_end_token_id': 47,
    'text_config': {
        'model_type': 'qwen3_5_moe', 'hidden_size': 64, 'num_hidden_layers': 4,
        'intermediate_size': 64, 'num_attention_heads': 2, 'num_key_value_heads': 1,
        'head_dim': 32, 'vocab_size': 64, 'linear_num_value_heads': 2,
        'linear_num_key_heads': 1, 'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
        'linear_conv_kernel_dim': 4, 'num_experts': 4, 'num_experts_per_tok': 2,
        'shared_expert_intermediate_size': 64, 'moe_intermediate_size': 64,
        'full_attention_interval': 4, 'rms_norm_eps': 1e-6,
        'rope_parameters': {'rope_theta': 10000000.0, 'partial_rotary_factor': .25, 'rope_type': 'default'},
    },
    'audio_decoder_config': {
        'dim': 64, 'vocab_size': 16, 'num_codebooks': 4, 'n_head': 2,
        'n_local_heads': 1, 'head_dim': 32, 'intermediate_size': 64,
        'norm_eps': 1e-6, 'rope_base': 10000000.0, 'rotary_percent': .25,
        'shared_enrichment_layers': 1, 'vq_backbone_codebooks': 1,
    },
}
prompt = mx.array([[[2,0,0,0,0], [32,0,1,2,3], [33,1,2,3,4], [3,0,0,0,0]]])
for bits in [4, 8]:
    mx.random.seed(42)
    model = Model(ModelConfig.from_dict(config))
    quant = {'group_size': 32, 'bits': 8, 'mode': 'affine'}
    def predicate(path, module):
        if hasattr(module, 'to_quantized') and (path == 'backbone.embed_tokens' or '.switch_mlp.' in path):
            value = {'group_size': 32, 'bits': 8 if path == 'backbone.embed_tokens' else bits}
            quant[path] = value
            return value
        return False
    nn.quantize(model, class_predicate=predicate)
    config['quantization'] = quant
    (root / f'q{bits}-config.json').write_text(json.dumps(config, indent=2) + '\n')
    mx.save_safetensors(str(root / f'q{bits}-weights.safetensors'), dict(tree_flatten(model.parameters())))
    bc, tc = model.backbone.make_cache(), model.tower.make_cache()
    hidden = model._backbone_step(prompt, bc)
    model.tower.prefill(hidden[:, :-1], prompt[:, :-1], prompt[:, 1:], tc)
    main = mx.array([[34]])
    codes = model.tower.decode_step(main, hidden[:, -1:], prompt[:, -1:], tc, temperature=1, top_k=1, top_p=1)
    frame = mx.concatenate([main[:, :, None], codes], axis=-1)
    next_hidden = model._backbone_step(frame, bc)
    expected = {'prompt': prompt, 'hidden': hidden, 'codes': codes, 'next_hidden': next_hidden,
                'logits': model.lm_head(hidden)}
    mx.eval(list(expected.values()))
    mx.save_safetensors(str(root / f'q{bits}-expected.safetensors'), expected)
print('Generated deterministic Q4/Q8 fixtures.')
