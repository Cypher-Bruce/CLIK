import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from src.utils.comm import (
    all_to_all_with_pad,
    gather_sequence,
    get_sp_group,
    is_sp_enabled,
    split_sequence,
)


def gradient_checkpoint_forward(module, *args, **kwargs):
    return torch.utils.checkpoint.checkpoint(
        module,
        *args,
        **kwargs,
        use_reentrant=False,
    )


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_cross_rope(
    xq: Tensor, xk: Tensor, q_freqs: Tensor, k_freqs: Tensor
) -> tuple[Tensor, Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = q_freqs[..., 0] * xq_[..., 0] + q_freqs[..., 1] * xq_[..., 1]
    xk_out = k_freqs[..., 0] * xk_[..., 0] + k_freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class PosEmbedND(nn.Module):
    def __init__(self, dim: int, theta: int):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        ids: Tensor = None,
        axes_dim: list[int] = None,
        axes_lengths: list[int] = None,
        device: torch.device = None,
    ) -> Tensor:
        if sum(axes_dim) != self.dim:
            raise ValueError(
                f"sum(axes_dim) must equal dim, got sum({axes_dim}) = {sum(axes_dim)} but dim = {self.dim}"
            )
        if ids is None:
            ids = self._create_ids(axes_dim, axes_lengths, device)

        emb = torch.cat(
            [rope(ids[..., i], axes_dim[i], self.theta) for i in range(len(axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)

    def _create_ids(
        self, axes_dim: list[int], axes_lengths: list[int], device: torch.device = None
    ) -> Tensor:
        if len(axes_lengths) != len(axes_dim):
            raise ValueError(
                f"axes_lengths and axes_dim must have the same length, got {len(axes_lengths)} vs {len(axes_dim)}"
            )
        ranges = [torch.arange(length, device=device) for length in axes_lengths]
        grids = torch.meshgrid(*ranges, indexing="ij")
        ids = torch.stack([g.flatten() for g in grids], dim=-1)
        return ids.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got {d_model} % {num_heads} = {d_model % num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        q, k = apply_rope(q, k, freqs)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.d_model
        )
        return self.W_o(output)


class SPMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with sequence-parallel all-to-all.

    Swaps heads <-> sequence so each rank attends over the full sequence
    with a subset of heads, then swaps back after attention.
    """

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        if is_sp_enabled():
            q = all_to_all_with_pad(q, get_sp_group(), scatter_dim=1, gather_dim=2)
            k = all_to_all_with_pad(k, get_sp_group(), scatter_dim=1, gather_dim=2)
            v = all_to_all_with_pad(v, get_sp_group(), scatter_dim=1, gather_dim=2)

        q, k = apply_rope(q, k, freqs)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        if is_sp_enabled():
            attn_output = all_to_all_with_pad(
                attn_output, get_sp_group(), scatter_dim=2, gather_dim=1
            )

        output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.d_model
        )
        return self.W_o(output)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got {d_model} % {num_heads} = {d_model % num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_freqs: Tensor,
        kv_freqs: Tensor,
        kv_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, q_seq_length, _ = q.shape
        _, kv_seq_length, _ = kv.shape

        q = self.W_q(q)
        k = self.W_k(kv)
        v = self.W_v(kv)

        q = q.view(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        q, k = apply_cross_rope(q, k, q_freqs, kv_freqs)

        # kv_mask: (B, kv_len) -> broadcast to (B, 1, 1, kv_len) for SDPA
        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask[:, None, None, :].bool()

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = attn_output.transpose(1, 2).reshape(
            batch_size, q_seq_length, self.d_model
        )
        output = self.W_o(output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seq_len_n: int,
        seq_len_hw: int,
    ):
        super(TransformerBlock, self).__init__()
        self.self_attn_n = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attn_hw = SPMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attn_cross = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm3 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm4 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)

        self.gradient_checkpointing = False

    def forward(
        self,
        x: Tensor,
        encoder_hidden_states: Tensor,
        idx: int,
        hw_freqs: Tensor,
        n_freqs: Tensor,
        l_freqs: Tensor,
        all_freqs: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, n, hw, _ = x.shape
        norm_x = self.norm1(x)
        norm_x = rearrange(norm_x, "b n hw c -> (b hw) n c")
        out = self.self_attn_n(norm_x, freqs=n_freqs)
        out = rearrange(out, "(b hw) n c -> b n hw c", b=batch, hw=hw)
        x = x + out

        norm_x = self.norm2(x)
        norm_x = rearrange(norm_x, "b n hw c -> (b n) hw c")
        out = self.self_attn_hw(norm_x, freqs=hw_freqs)
        out = rearrange(out, "(b n) hw c -> b n hw c", b=batch, n=n)
        x = x + out

        norm_x = self.norm3(x)
        if is_sp_enabled():
            norm_x = gather_sequence(norm_x, get_sp_group(), dim=2, grad_scale="up")
        norm_x = rearrange(norm_x, "b n hw c -> b (n hw) c")
        out = self.self_attn_cross(
            norm_x,
            encoder_hidden_states,
            q_freqs=all_freqs,
            kv_freqs=l_freqs,
            kv_mask=encoder_attention_mask,
        )
        out = rearrange(out, "b (n hw) c -> b n hw c", b=batch, n=n)
        if is_sp_enabled():
            out = split_sequence(out, get_sp_group(), dim=2, grad_scale="down")
        x = x + out

        norm_x = self.norm4(x)
        out = self.feed_forward(norm_x)
        x = x + out

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_base_model_layers: int,
        num_token_per_layer: int,
        lora_rank: int,
        dim_accumulation: int,
        output_dim: int,
        head_dim: int,
        num_blocks: int,
        lora_A_token_count: int,
        lora_B_token_count: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_n = num_base_model_layers
        self.seq_h = num_token_per_layer
        self.seq_w = lora_rank
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.dim_accumulation = dim_accumulation
        if self.d_model % self.dim_accumulation != 0:
            raise ValueError(
                f"d_model must be divisible by dim_accumulation, got {self.d_model} % {self.dim_accumulation} = {self.d_model % self.dim_accumulation}"
            )
        self.dim_wo_accumulation = self.d_model // self.dim_accumulation
        if self.d_model % self.head_dim != 0:
            raise ValueError(
                f"d_model must be divisible by head_dim, got {self.d_model} % {self.head_dim} = {self.d_model % self.head_dim}"
            )
        self.num_heads = self.d_model // self.head_dim
        self.lora_A_token_count = lora_A_token_count
        self.lora_B_token_count = lora_B_token_count

        self.pos_embed = PosEmbedND(dim=self.head_dim, theta=10000)

        self.norm_input = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)
        self.norm_tokens = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)

        self.hw_pos = nn.Parameter(
            torch.randn((num_token_per_layer * lora_rank, self.dim_wo_accumulation))
        )  # [40x32, 3072]
        self.layer_pos = nn.Parameter(
            torch.randn((num_base_model_layers, self.dim_wo_accumulation))
        )  # [60, 3072]

        self.norm_final = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)

        module_list = []
        for _ in range(num_blocks):
            module_list.append(
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_model * 4,
                    seq_len_n=self.seq_n,
                    seq_len_hw=self.seq_h * self.seq_w,
                )
            )
        self.blocks = nn.ModuleList(module_list)

        self.proj_out_A = nn.Linear(self.dim_wo_accumulation, self.output_dim)
        self.proj_out_B = nn.Linear(self.dim_wo_accumulation, self.output_dim)

        self._init_weights()

        self.gradient_checkpointing = False

    def _init_weights(self):
        def _init_linear(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_init_linear)

        torch.nn.init.zeros_(self.proj_out_B.weight)
        torch.nn.init.zeros_(self.proj_out_B.bias)

    @torch.no_grad()
    def _get_pos_embed(self, x: Tensor) -> Tensor:
        # hw attention: 2D grid (h, w)
        hw_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_h, self.seq_w // self.dim_accumulation],
            device=x.device,
        )
        # n attention: 1D sequence
        n_freqs = self.pos_embed(
            axes_dim=[self.head_dim], axes_lengths=[self.seq_n], device=x.device
        )

        # cross attention Q: 2D grid (n, hw)
        all_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_n, self.seq_h * self.seq_w // self.dim_accumulation],
            device=x.device,
        )

        # cross attention KV: diagonal positions
        diag_ids = torch.arange(x.shape[1], device=x.device)
        diag_ids = torch.stack([diag_ids, diag_ids], dim=-1).unsqueeze(0)  # (1, L, 2)
        diag_ids = diag_ids + max(self.seq_h, self.seq_w // self.dim_accumulation)
        l_freqs = self.pos_embed(
            ids=diag_ids,
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
        )
        return hw_freqs, n_freqs, l_freqs, all_freqs

    def get_tokens(self, batch_size: int) -> Tensor:
        hw_pos = repeat(
            self.hw_pos, "hw d -> b n hw d", b=batch_size, n=self.seq_n
        ).clone()  # need clone
        layer_pos = repeat(
            self.layer_pos, "n d -> b n hw d", b=batch_size, hw=self.seq_h * self.seq_w
        ).clone()
        tokens = hw_pos + layer_pos
        tokens = rearrange(
            tokens,
            "b n (h w a) d -> b n (h w) (a d)",
            a=self.dim_accumulation,
            h=self.seq_h,
            w=self.seq_w // self.dim_accumulation,
        )

        if is_sp_enabled():
            tokens = split_sequence(
                tokens, get_sp_group(), dim=2, grad_scale="down"
            )  # [b, n, hw, c] -> [b, n, hw/sp, c]

        tokens = self.norm_tokens(tokens)
        return tokens

    def post_process(self, x: Tensor) -> Tensor:
        x = self.norm_final(x)

        if is_sp_enabled():
            x = gather_sequence(
                x, get_sp_group(), dim=2, grad_scale="up"
            )  # [b, n, hw/sp, c] -> [b, n, hw, c]

        x = rearrange(
            x,
            "b n (h w) (a c) -> b n h (w a) c",
            a=self.dim_accumulation,
            h=self.seq_h,
            w=self.seq_w // self.dim_accumulation,
        )  # (B, 60, 40, 34, 3072)

        x_A, x_B = torch.split(
            x, [self.lora_A_token_count, self.lora_B_token_count], dim=2
        )
        x_A = self.proj_out_A(x_A)
        x_B = self.proj_out_B(x_B)
        x = torch.cat([x_A, x_B], dim=2)

        x = x * 1e-2

        return x

    def forward(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_hidden_states = encoder_hidden_states.squeeze(1)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            encoder_hidden_states
        )
        encoder_hidden_states = self.norm_input(encoder_hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = gradient_checkpoint_forward(
                self.get_tokens,
                batch_size=encoder_hidden_states.shape[0],
            )
        else:
            x = self.get_tokens(encoder_hidden_states.shape[0])

        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    idx=idx,
                    hw_freqs=hw_freqs,
                    n_freqs=n_freqs,
                    l_freqs=l_freqs,
                    all_freqs=all_freqs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x = block(
                    x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    idx=idx,
                    hw_freqs=hw_freqs,
                    n_freqs=n_freqs,
                    l_freqs=l_freqs,
                    all_freqs=all_freqs,
                    encoder_attention_mask=encoder_attention_mask,
                )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = gradient_checkpoint_forward(
                self.post_process,
                x=x,
            )
        else:
            x = self.post_process(x)

        return x
