import math

import torch
from torch import nn
import torch.nn.functional as F

from attnres import BlockAttnResReference, FullAttnResReference
from hyper_connections import get_init_and_expand_reduce_stream_functions
from value_residual import ValueResidualState


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.v_residual = config.v_residual
        if self.v_residual:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
            self.lamb2 = nn.Parameter(torch.tensor(0.5))
        else:
            self.lamb1 = 1.0
            self.lamb2 = 0.0

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            bias = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "bias", bias.view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, vrl_state=None):
        b, t, c = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(b, t, self.n_head, self.head_dim)
        k = k.view(b, t, self.n_head, self.head_dim)
        v = v.view(b, t, self.n_head, self.head_dim)

        if self.v_residual:
            if vrl_state is None:
                raise ValueError("v_residual requires vrl_state")
            v = vrl_state.mix(v, self.lamb1, self.lamb2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class AttnBranch(nn.Module):
    def __init__(self, norm, attn):
        super().__init__()
        self.norm = norm
        self.attn = attn

    def forward(self, x, vrl_state=None):
        x = self.norm(x)
        return self.attn(x, vrl_state=vrl_state)


class AttnResBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.attn_branch = AttnBranch(self.ln_1, self.attn)

    def attn_output(self, x, vrl_state=None):
        return self.attn_branch(x, vrl_state=vrl_state)

    def mlp_output(self, x):
        return self.mlp(self.ln_2(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx, init_hc):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.attn_branch = AttnBranch(self.ln_1, self.attn)

        hc_kwargs = dict(
            mhc=config.mhc,
            sinkhorn_iters=config.sinkhorn_iters,
            sinkhorn_tau=config.sinkhorn_tau,
            mhc_h_res_proj=config.mhc_h_res_proj,
            ns_steps=config.ns_steps,
            ns_eps=config.ns_eps,
            ns_coeffs=config.ns_coeffs,
            mhc_residual_identity_mix=config.mhc_residual_identity_mix,
            mhc_residual_alpha=config.mhc_residual_alpha,
        )

        self.hc_attn = init_hc(
            dim=config.n_embd,
            branch=self.attn_branch,
            layer_index=layer_idx * 2,
            **hc_kwargs,
        )

        self.hc_mlp = init_hc(
            dim=config.n_embd,
            branch=nn.Sequential(self.ln_2, self.mlp),
            layer_index=layer_idx * 2 + 1,
            **hc_kwargs,
        )

    def forward(self, x, vrl_state=None):
        x = self.hc_attn(x, vrl_state=vrl_state)
        x = self.hc_mlp(x)
        return x


class GPTConfig:
    def __init__(self, **kwargs):
        self.block_size = kwargs.pop("block_size", 1024)
        self.vocab_size = kwargs.pop("vocab_size", 50304)
        self.n_layer = kwargs.pop("n_layer", 12)
        self.n_head = kwargs.pop("n_head", 12)
        self.n_embd = kwargs.pop("n_embd", 768)
        self.dropout = kwargs.pop("dropout", 0.0)
        self.bias = kwargs.pop("bias", True)

        self.hc_num_streams = kwargs.pop("hc_num_streams", 1)
        self.hc_num_fracs = kwargs.pop("hc_num_fracs", 1)
        self.hc_disable = kwargs.pop("hc_disable", False)
        self.mhc = kwargs.pop("mhc", False)
        self.sinkhorn_iters = kwargs.pop("sinkhorn_iters", 20)
        self.sinkhorn_tau = kwargs.pop("sinkhorn_tau", 0.05)
        self.mhc_h_res_proj = kwargs.pop("mhc_h_res_proj", "sinkhorn")
        self.ns_steps = kwargs.pop("ns_steps", 5)
        self.ns_eps = kwargs.pop("ns_eps", 1e-7)
        self.ns_coeffs = kwargs.pop("ns_coeffs", (3.0, -3.2, 1.2))
        self.mhc_residual_identity_mix = kwargs.pop("mhc_residual_identity_mix", False)
        self.mhc_residual_alpha = kwargs.pop("mhc_residual_alpha", 0.01)
        self.v_residual = kwargs.pop("v_residual", False)
        self.v_residual_lamb_lr = kwargs.pop("v_residual_lamb_lr", 1e-2)
        self.attnres_variant = kwargs.pop("attnres_variant", "none")
        self.attnres_block_size = kwargs.pop("attnres_block_size", 1)
        self.attnres_eps = kwargs.pop("attnres_eps", 1e-8)

        for key, value in kwargs.items():
            setattr(self, key, value)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.vrl_state = ValueResidualState() if config.v_residual else None
        self.use_attnres = config.attnres_variant != "none"
        self.attnres_num_logical_layers = 2 * config.n_layer
        self.attnres_mixer = None

        self._validate_config()

        if self.use_attnres:
            if config.attnres_variant == "full":
                self.attnres_mixer = FullAttnResReference(
                    num_layers=self.attnres_num_logical_layers,
                    dim=config.n_embd,
                    eps=config.attnres_eps,
                )
            elif config.attnres_variant == "block":
                self.attnres_mixer = BlockAttnResReference(
                    num_layers=self.attnres_num_logical_layers,
                    dim=config.n_embd,
                    block_size=config.attnres_block_size,
                    eps=config.attnres_eps,
                )
            else:
                raise NotImplementedError(
                    f"attnres_variant={config.attnres_variant!r} is not integrated into GPT yet"
                )

            self.expand_stream = None
            self.reduce_stream = None
            blocks = [AttnResBlock(config) for _ in range(config.n_layer)]
        else:
            init_hc, expand_stream, reduce_stream = (
                get_init_and_expand_reduce_stream_functions(
                    config.hc_num_streams,
                    num_fracs=config.hc_num_fracs,
                    disable=config.hc_disable,
                )
            )

            self.expand_stream = expand_stream
            self.reduce_stream = reduce_stream
            blocks = [Block(config, i, init_hc) for i in range(config.n_layer)]

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(blocks),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _validate_config(self):
        valid_attnres_variants = {"none", "full", "block"}
        if self.config.attnres_variant not in valid_attnres_variants:
            raise ValueError(
                f"attnres_variant must be one of {sorted(valid_attnres_variants)}, got {self.config.attnres_variant!r}"
            )

        if not self.use_attnres:
            return

        if self.config.mhc:
            raise ValueError("Attention Residuals cannot be combined with mhc")
        if not self.config.hc_disable:
            raise ValueError("Attention Residuals requires hc_disable=True")
        if self.config.hc_num_streams != 1:
            raise ValueError("Attention Residuals requires hc_num_streams=1")
        if self.config.hc_num_fracs != 1:
            raise ValueError("Attention Residuals requires hc_num_fracs=1")
        if self.config.attnres_block_size <= 0:
            raise ValueError("attnres_block_size must be positive")
        if self.config.attnres_eps <= 0:
            raise ValueError("attnres_eps must be positive")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_input(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        return self.transformer.drop(tok_emb + pos_emb)

    def _forward_full_attnres_hidden(self, embedding, vrl_state=None):
        logical_outputs = []

        for block_index, block in enumerate(self.transformer.h):
            attn_layer_index = 2 * block_index
            attn_input = self.attnres_mixer.layer_input(
                embedding=embedding,
                prior_layer_outputs=logical_outputs,
                layer_index=attn_layer_index,
            )
            attn_output = block.attn_output(attn_input, vrl_state=vrl_state)
            logical_outputs.append(attn_output)

            mlp_layer_index = attn_layer_index + 1
            mlp_input = self.attnres_mixer.layer_input(
                embedding=embedding,
                prior_layer_outputs=logical_outputs,
                layer_index=mlp_layer_index,
            )
            mlp_output = block.mlp_output(mlp_input)
            logical_outputs.append(mlp_output)

        if len(logical_outputs) != self.attnres_num_logical_layers:
            raise RuntimeError(
                f"expected {self.attnres_num_logical_layers} logical outputs, got {len(logical_outputs)}"
            )

        return self.attnres_mixer.final_output(
            embedding=embedding,
            layer_outputs=logical_outputs,
        )

    def _forward_block_attnres_hidden(self, embedding, vrl_state=None):
        state = self.attnres_mixer.init_state(embedding)

        for block_index, block in enumerate(self.transformer.h):
            attn_layer_index = 2 * block_index
            attn_input = self.attnres_mixer.layer_input(
                state=state,
                layer_index=attn_layer_index,
            )
            attn_output = block.attn_output(attn_input, vrl_state=vrl_state)
            self.attnres_mixer.append_layer_output(state, attn_output)

            mlp_layer_index = attn_layer_index + 1
            mlp_input = self.attnres_mixer.layer_input(
                state=state,
                layer_index=mlp_layer_index,
            )
            mlp_output = block.mlp_output(mlp_input)
            self.attnres_mixer.append_layer_output(state, mlp_output)

        if state.completed_layers != self.attnres_num_logical_layers:
            raise RuntimeError(
                f"expected {self.attnres_num_logical_layers} logical outputs, got {state.completed_layers}"
            )

        return self.attnres_mixer.final_output(state)

    def forward(self, idx, targets=None):
        x = self._embed_input(idx)

        vrl_state = self.vrl_state
        if vrl_state is not None:
            vrl_state.reset()

        if self.use_attnres:
            if self.config.attnres_variant == "full":
                x = self._forward_full_attnres_hidden(x, vrl_state=vrl_state)
            elif self.config.attnres_variant == "block":
                x = self._forward_block_attnres_hidden(x, vrl_state=vrl_state)
            else:
                raise NotImplementedError(
                    f"attnres_variant={self.config.attnres_variant!r} is not implemented in forward"
                )
            x = self.transformer.ln_f(x)
        else:
            x = self.expand_stream(x)

            for block in self.transformer.h:
                x = block(x, vrl_state=vrl_state)

            x = self.transformer.ln_f(x)
            x = self.reduce_stream(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                    decay.add(full_name)
                else:
                    no_decay.add(full_name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        lamb_params = {pn for pn in param_dict if "lamb" in pn}

        decay -= lamb_params
        no_decay -= lamb_params

        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay | lamb_params)) == 0

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if lamb_params:
            lamb_lr = self.config.v_residual_lamb_lr
            optim_groups.append(
                {
                    "params": [param_dict[pn] for pn in sorted(lamb_params)],
                    "weight_decay": 0.0,
                    "lr": lamb_lr,
                    "lr_scale": lamb_lr / learning_rate,
                }
            )

        use_fused = (
            device_type == "cuda"
        ) and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )

        return optimizer
