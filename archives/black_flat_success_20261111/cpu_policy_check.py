"""CPU checkpoint load and inference check; no simulation or training."""
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

root = Path(__file__).resolve().parent
checkpoint = root / 'logs/black_teacherfree/mixed_30000_20261111/updates_30000/model_33999.pt'
expected = '55be4aff5a54232a45f27e6df567d0de49d53e4aeb9096c1e96c307b343749f7'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
# Load only the trusted, hash-verified project checkpoint.
payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
obs = TensorDict({'actor': torch.zeros(3, 72), 'critic': torch.zeros(3, 72)}, batch_size=[3])
results = {}
for kind, size in [('actor', 12), ('critic', 1)]:
    model = MLPModel(obs, {'actor': ['actor'], 'critic': ['critic']}, kind, size,
                     hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0,
                                       'std_type': 'scalar'} if kind == 'actor' else None)
    state = payload[f'{kind}_state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        output = model(obs)
    assert output.shape == (3, size) and torch.isfinite(output).all()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    results[kind] = {'output_shape': list(output.shape), 'finite': True,
                     'strict_load': True, 'state_unchanged': True}
print(json.dumps({'sha256': expected, 'scope': 'CPU inference only', 'results': results}))
