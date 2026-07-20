from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from .model import TransformerModel
from .tokenizer import Tokenizer2DBatchedLoRA


class ParameterGenerator(nn.Module):
    def __init__(
        self,
        pg_config: Dict[str, Any],
    ):
        super().__init__()
        self.tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=pg_config["token_dim"],
            rank=pg_config["rank"],
            alpha=pg_config["alpha"],
            pg_mapping=pg_config["pg_mapping"],
        )
        self.lora_A_token_count = self.tokenizer.lora_A_token_count
        self.lora_B_token_count = self.tokenizer.lora_B_token_count

        self.model_config = pg_config["model"]
        self.model = TransformerModel(
            d_model=self.model_config["d_model"],
            num_base_model_layers=self.model_config["num_base_model_layers"],
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=pg_config["rank"],
            output_dim=self.model_config["output_dim"],
            head_dim=self.model_config["head_dim"],
            num_blocks=self.model_config["num_pg_layers"],
            dim_accumulation=self.model_config["dim_accumulation"],
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
        )

        self.layer_num = self.model_config["num_base_model_layers"]
        self.prefix = pg_config["prefix"]
        self.lora_rank = pg_config["rank"]

        hidden_input_dim = self.model_config["hidden_input_dim"]
        self.hidden_in = nn.Linear(hidden_input_dim, self.model_config["d_model"])

        # Optional learnable aggregation across all base-model hidden states
        # (ELMo: gamma * sum_j softmax(w_j) * h_j). When enabled, ``forward``
        # expects ``hidden`` of shape (B, L+1, T, D).
        self.hidden_aggregation = pg_config.get("hidden_aggregation", "single")
        if self.hidden_aggregation == "elmo":
            n_states = self.model_config["num_input_hidden_states"]
            self.elmo_weights = nn.Parameter(torch.zeros(n_states))
            # FSDP v2 (``fully_shard``) does not support 0-D parameters,
            # so gamma is stored as a (1,) tensor and broadcasts in forward.
            self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.register_buffer(
            "_lora_scale",
            torch.tensor(self.tokenizer.lora_scale),
            persistent=False,
        )

    def get_lora_count(self):
        return (
            self.model_config["output_dim"]
            * (self.lora_A_token_count + self.lora_B_token_count)
            * self.model_config["num_base_model_layers"]
            * self.lora_rank
        )

    def enable_gradient_checkpointing(self):
        for _, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True

    def forward(
        self, hidden: Tensor, attention_mask: Tensor | None = None
    ) -> Dict[str, Tensor]:
        """Generate per-layer LoRA weights from context hidden states.

        Args:
            hidden: When ``hidden_aggregation == "elmo"``, a stacked tensor of
                    shape (batch, num_input_hidden_states, seq_len, hidden_dim)
                    holding every base-model hidden state (embedding output +
                    each transformer layer). Otherwise, (batch, seq_len,
                    hidden_dim) — a single (or pre-averaged) hidden state.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.
        """
        if self.hidden_aggregation == "elmo":
            # hidden: (B, L+1, T, D)
            w = torch.softmax(self.elmo_weights.to(hidden.dtype), dim=0)
            hidden = self.elmo_gamma.to(hidden.dtype) * torch.einsum(
                "l,bltd->btd", w, hidden
            )

        embeddings = self.hidden_in(hidden).unsqueeze(1)
        output = self.model(
            encoder_hidden_states=embeddings,
            encoder_attention_mask=attention_mask,
        )

        all_layer_state_dict = {}
        shape_state_dict = self.tokenizer.shape_state_dict
        for layer_index in range(self.layer_num):
            layer_output = output[:, layer_index]
            layer_state_dict = self.tokenizer.detokenize(shape_state_dict, layer_output)
            for key, value in layer_state_dict.items():
                new_key = f"{self.prefix}{layer_index}.{key}"
                all_layer_state_dict[new_key] = value

        all_layer_state_dict["scale"] = self._lora_scale.to(dtype=output.dtype)

        if self.training:
            all_layer_state_dict["param_norm"] = (output**2).mean()

        return all_layer_state_dict
