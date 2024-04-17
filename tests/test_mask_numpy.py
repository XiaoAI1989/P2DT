"""Dependency-light (numpy-only) checks of the prompt-visible attention mask logic.

Mirrors the exact arithmetic of model.MaskedCausalAttention / model.expert_Block
to verify the two properties the fix must guarantee:

1. suffix task prompt tokens DO influence trajectory outputs (they are learned
   parameters, so this is required for the mechanism to exist at all);
2. causality among trajectory tokens is preserved (a future timestep never
   influences the prediction at an earlier timestep).

It also documents the failure mode of a plain causal mask over appended
prompts: the prompts become invisible to every trajectory position, i.e. the
mechanism silently degenerates to a no-op. Run with `python tests/test_mask_numpy.py`
or pytest.
"""
import numpy as np

rng = np.random.default_rng(0)
H = 64        # hidden_dim
T_TRAJ = 30   # 3 * context window of 10
TOK = 5       # prompt length


def linear(x, W, b):
    return x @ W.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def make_params(seed):
    prng = np.random.default_rng(seed)
    p = {}
    for name in ["q", "k", "v", "proj"]:
        p[f"{name}_W"] = prng.standard_normal((H, H)) * 0.05
        p[f"{name}_b"] = prng.standard_normal(H) * 0.01
    p["mlp1_W"] = prng.standard_normal((4 * H, H)) * 0.05
    p["mlp1_b"] = prng.standard_normal(4 * H) * 0.01
    p["mlp2_W"] = prng.standard_normal((H, 4 * H)) * 0.05
    p["mlp2_b"] = prng.standard_normal(H) * 0.01
    p["ln1_g"] = np.ones(H); p["ln1_b"] = np.zeros(H)
    p["ln2_g"] = np.ones(H); p["ln2_b"] = np.zeros(H)
    return p


def attention(x, p, n_suffix=0):
    # mirrors MaskedCausalAttention.forward (single head, dropout off)
    B, T, C = x.shape
    q = linear(x, p["q_W"], p["q_b"])
    k = linear(x, p["k_W"], p["k_b"])
    v = linear(x, p["v_W"], p["v_b"])
    w = q @ k.transpose(0, 2, 1) / np.sqrt(C)
    mask = np.tril(np.ones((T, T)))
    if n_suffix > 0:
        mask = mask.copy()
        mask[:, T - n_suffix:] = 1   # every position may attend to the suffix prompts
    w = np.where(mask == 0, -np.inf, w)
    return linear(softmax(w) @ v, p["proj_W"], p["proj_b"])


def expert_block(x, task_token, p, prompt_visible=True):
    # mirrors expert_Block.forward
    B = x.shape[0]
    tok = np.repeat(task_token[None], B, axis=0)
    x = np.concatenate([x, tok], axis=1)
    x = x + attention(x, p, n_suffix=TOK if prompt_visible else 0)
    x = layernorm(x, p["ln1_g"], p["ln1_b"])
    x = x + linear(gelu(linear(x, p["mlp1_W"], p["mlp1_b"])), p["mlp2_W"], p["mlp2_b"])
    x = layernorm(x, p["ln2_g"], p["ln2_b"])
    return x[:, :-TOK]


PARAMS = [make_params(s) for s in (1, 2, 3)]   # 3 expert blocks
X = rng.standard_normal((4, T_TRAJ, H))


def forward(token_seed, x=X, prompt_visible=True):
    trng = np.random.default_rng(token_seed)
    out = x.copy()
    for p in PARAMS:
        tok = trng.standard_normal((TOK, H)) * 0.02
        out = expert_block(out, tok, p, prompt_visible=prompt_visible)
    return out


def test_prompts_influence_outputs():
    out_a = forward(token_seed=10)
    out_b = forward(token_seed=20)
    diff = np.abs(out_a - out_b).max()
    assert diff > 0, "with the prompt-visible mask, different prompts must give different outputs"
    print(f"[ok] prompts influence outputs (max diff {diff:.3e})")


def test_trajectory_causality_preserved():
    x2 = X.copy()
    k = T_TRAJ - 1
    x2[:, k:] += 100.0  # perturb only the last trajectory position
    out_orig = forward(token_seed=10)
    out_pert = forward(token_seed=10, x=x2)
    assert np.array_equal(out_orig[:, :k], out_pert[:, :k]), \
        "future trajectory tokens must not influence earlier positions"
    assert np.abs(out_orig[:, k:] - out_pert[:, k:]).max() > 0
    print("[ok] trajectory causality preserved (past outputs bit-identical under future perturbation)")


def test_plain_causal_mask_makes_prompts_inert():
    out_a = forward(token_seed=10, prompt_visible=False)
    out_b = forward(token_seed=20, prompt_visible=False)
    assert np.array_equal(out_a, out_b), \
        "documentation of the failure mode: plain causal mask hides appended prompts entirely"
    print("[ok] regression doc: plain causal mask over suffix prompts is a no-op (hence the fix)")


if __name__ == "__main__":
    test_prompts_influence_outputs()
    test_trajectory_causality_preserved()
    test_plain_causal_mask_makes_prompts_inert()
    print("all numpy mask-logic tests passed")
