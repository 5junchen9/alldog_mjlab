"""Public strategy adapter: no machine paths, checkpoints or automatic launch."""
from __future__ import annotations

def _need(ok, message):
    if not ok:
        raise ValueError(message)

def require_native_frame(obs):
    import torch
    x, c, p = (obs[k] for k in ("actor", "critic", "p45"))
    n = len(x)
    for a, w in ((x, 446), (c, 446), (p, 45)):
        _need(a.shape == (n, w) and a.dtype == torch.float32 and torch.isfinite(a).all(), "real native input schema")
    _need(torch.equal(x[:, 45:48], p[:, 30:33]), "same-clock applied command")

def install_weights(alg, actor_state, critic_state):
    """Use on the privately constructed original learner before any update."""
    _need(not alg.optimizer.state, "fresh optimizer only")
    alg.actor.load_state_dict(actor_state, strict=True)
    alg.critic.load_state_dict(critic_state, strict=True)
    _need(not alg.optimizer.state, "do not import old optimizer or simulator state")

def initial_course(state):
    """Checked immediately after the true native initial reset, allowing births only."""
    c = state.passage_course
    _need(c is not None and c.current == c.votes == 0 and not (c.history or c.promotion or c.outcomes),
          "only new genuine passage evidence")
    _need(all(row[1] == 0 for row in c.active.values()), "only initial level births")
    _need(not bool(state.passage_seen.pass_once.any()), "no prior passage latch")

def tell_trainable(alg):
    groups = {
        "raw446_producer51": list(alg.actor.producer51.parameters()),
        "shared96_to_12_mean": list(alg.actor.mlp.parameters()),
        "gaussian_distribution": list(alg.actor.distribution.parameters()),
        "privileged446_critic": list(alg.critic.parameters()),
    }
    seen = [p for g in groups.values() for p in g]
    flat = [p for pg in alg.optimizer.param_groups for p in pg["params"]]
    _need(all(groups.values()) and all(p.requires_grad for p in seen) and
          len({id(p) for p in seen}) == len(seen) and
          len(seen) == len(flat) and {id(p) for p in seen} == {id(p) for p in flat},
          "only and all declared parameter groups train")
    return {k: sum(p.numel() for p in g) for k, g in groups.items()}
