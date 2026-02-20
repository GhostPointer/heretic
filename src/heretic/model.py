# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Type, cast

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

from .config import QuantizationMethod, RowNormalization, Settings
from .utils import Prompt, batchify, empty_cache, print


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    # Prioritize the architecture defined in the config. If it explicitly
    # claims to be a CausalLM, trust that over the presence of vision_config.
    # Some text-only models (e.g. MiniMax M2.5) or VL models used in text mode
    # may contain vision_config but should still be loaded as CausalLM.
    for config in configs:
        if isinstance(config, dict) and "architectures" in config:
            if any("CausalLM" in arch for arch in config["architectures"]):
                return AutoModelForCausalLM

    if any([("vision_config" in config) for config in configs]):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(settings.model),
                    **extra_kwargs,
                )

                # If we reach this point and the model requires trust_remote_code,
                # either the user accepted, or settings.trust_remote_code is True.
                if self.trusted_models.get(settings.model) is None:
                    self.trusted_models[settings.model] = True

                # Detect FP8 layers (kept in FP8 for efficient inference).
                self._detect_fp8()

                # Some trust_remote_code models wrap forward() with decorators
                # that reject standard HuggingFace kwargs (e.g., input_ids).
                self._unwrap_restrictive_forward_decorators()

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                # It also validates that FP8 inference works correctly if applicable.
                inputs, outputs = self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=500,
                )

                if settings.print_responses:
                    response = self.tokenizer.decode(
                        outputs[0, cast(Tensor, inputs["input_ids"]).shape[1] :],
                        skip_special_tokens=True,
                    )
                    print(f"\n* Sanity check response: {response}")
            except Exception as error:
                self.model = None  # ty:ignore[invalid-assignment]
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("[green]Ok[/] (quantized to 4-bit precision)")
            else:
                print("[green]Ok[/]")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, modules in self.get_layer_modules(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
            )

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect the actual leaf module names from get_layer_modules() so that PEFT
        # targets the right modules regardless of architecture. For example, MoE models
        # may use "w2" instead of "down_proj" for their expert down-projection.
        # Unrelated modules with the same leaf name (e.g. "conv.o_proj") will also get
        # LoRA adapters, but this is harmless as we only abliterate the modules we
        # target in abliterate(), leaving the others at their default (identity) state.
        module_id_to_name = {id(mod): name for name, mod in self.model.named_modules()}
        target_modules = set()
        for modules in self.get_layer_modules(0).values():
            for module in modules:
                full_name = module_id_to_name.get(id(module))
                if full_name is not None:
                    target_modules.add(full_name.rsplit(".", 1)[-1])
        target_modules = list(target_modules)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        print(f"* LoRA adapters initialized (targets: {', '.join(target_modules)})")

    def _detect_fp8(self) -> None:
        """
        Detects FP8Linear layers and logs their presence.

        FP8Linear layers (e.g., from MiniMax M2.5) are kept as-is for inference —
        PEFT's standard LoRA wrapper calls FP8Linear.forward() which uses efficient
        Triton FP8 kernels. Dequantization only happens on-the-fly in abliterate()
        when weight matrices need to be read for LoRA computation.
        """
        fp8_count = sum(
            1
            for _, module in self.model.named_modules()
            if "FP8Linear" in module.__class__.__name__
        )

        if fp8_count > 0:
            print(f"* Detected [bold]{fp8_count}[/] FP8 layers (kept in FP8 for inference)")

    def _convert_fp8_to_bf16_for_merge(self) -> None:
        """
        Converts FP8Linear layers to standard BFloat16 Linear layers.

        Only used when merging LoRA adapters into base weights for saving,
        since the saved model needs standard weight tensors.
        """
        fp8_module_names = [
            name
            for name, module in self.model.named_modules()
            if "FP8Linear" in module.__class__.__name__
        ]

        if not fp8_module_names:
            return

        print(f"* Converting {len(fp8_module_names)} FP8 layers to BF16 for merge...")

        for name in fp8_module_names:
            module = self.model.get_submodule(name)
            new_layer = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=torch.bfloat16,
                device=module.weight.device,
            )

            with torch.no_grad():
                # Reuse the dequantization logic. _dequantize_weight expects a
                # LoRA-wrapped module with a base_layer attribute, so we create
                # a lightweight shim that points to the raw FP8 module.
                shim = Module()
                shim.base_layer = module  # ty:ignore[unresolved-attribute]
                weight_data = Model._dequantize_weight(shim)
                new_layer.weight.copy_(weight_data.to(torch.bfloat16))

                if module.bias is not None:
                    new_layer.bias.copy_(module.bias.data.float().to(torch.bfloat16))

            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = self.model.get_submodule(parent_name)
            else:
                child_name = name
                parent = self.model
            setattr(parent, child_name, new_layer)

        empty_cache()
        print("* FP8 to BF16 conversion complete")

    @staticmethod
    def _dequantize_weight(module: Module) -> Tensor:
        """
        Get the float32 weight matrix from a LoRA-wrapped module.

        Handles three cases:
        - FP8 with block-wise scaling (e.g., MiniMax M2.5)
        - BitsAndBytes 4-bit quantization
        - Standard float weights (BF16/FP16/FP32)
        """
        base_layer = module.base_layer  # ty:ignore[unresolved-attribute]
        base_weight = cast(Tensor, base_layer.weight)

        # FP8 with block-wise scaling.
        # Identify the scale attribute name by inspecting the module.
        scale_attr = None
        for attr in ["weight_scale", "scale", "w_scale", "weight_scale_inv"]:
            if hasattr(base_layer, attr):
                scale_attr = attr
                break

        if scale_attr is not None:
            weight_data = base_weight.data.float()
            scale = getattr(base_layer, scale_attr).float().to(weight_data.device)

            # Determine if the scale needs to be inverted for dequantization.
            # Most attributes store the dequantization factor directly (multiply).
            # "weight_scale_inv" is ambiguous across libraries:
            #   - transformer_engine, msamp: it IS the dequant factor (multiply)
            #   - HuggingFace finegrained_fp8: it IS the dequant factor (multiply)
            #   - Other conventions: it may be 1/dequant_factor (need to invert)
            if scale_attr == "weight_scale_inv":
                module_origin = base_layer.__class__.__module__ or ""
                if not any(
                    lib in module_origin
                    for lib in ("transformer_engine", "msamp", "finegrained_fp8", "transformers")
                ):
                    scale = 1.0 / scale

            # Handle block-wise scaling (e.g., 128x128 blocks).
            if scale.dim() == 2 and weight_data.dim() == 2 and scale.shape != weight_data.shape:
                block_rows = math.ceil(weight_data.shape[0] / scale.shape[0])
                block_cols = math.ceil(weight_data.shape[1] / scale.shape[1])

                scale_expanded = torch.repeat_interleave(scale, block_rows, dim=0)
                scale_expanded = torch.repeat_interleave(scale_expanded, block_cols, dim=1)

                if scale_expanded.shape != weight_data.shape:
                    scale_expanded = scale_expanded[:weight_data.shape[0], :weight_data.shape[1]]

                return weight_data * scale_expanded
            else:
                return weight_data * scale

        # BitsAndBytes 4-bit quantization.
        quant_state = getattr(base_weight, "quant_state", None)
        if quant_state is not None:
            return cast(
                Tensor,
                bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                    base_weight.data,
                    quant_state,
                ).to(torch.float32),
            )

        # Standard float weights.
        return base_weight.to(torch.float32)

    def _unwrap_restrictive_forward_decorators(self) -> None:
        """
        Some models with trust_remote_code (e.g., MiniMax M2.5) wrap forward()
        with decorators like check_model_inputs whose wrapper function rejects
        standard HuggingFace kwargs (input_ids, attention_mask, etc.).
        Detect and unwrap such decorators to allow normal generation.
        """
        for model_obj in [self.model, getattr(self.model, "model", None)]:
            if model_obj is None:
                continue

            forward = getattr(type(model_obj), "forward", None)
            if forward is None:
                continue

            qualname = getattr(forward, "__qualname__", "")
            if "check_model_inputs" not in qualname:
                continue

            # Try __wrapped__ first (set by functools.wraps).
            if hasattr(forward, "__wrapped__"):
                type(model_obj).forward = forward.__wrapped__
                print("* Unwrapped check_model_inputs decorator from forward()")
                continue

            # Fallback: extract the original function from the closure.
            # Decorators that don't use functools.wraps typically capture the
            # original function in their closure cells.
            if hasattr(forward, "__closure__") and forward.__closure__:
                import types

                for cell in forward.__closure__:
                    try:
                        cell_contents = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(cell_contents, types.FunctionType) and cell_contents is not forward:
                        # Verify this looks like the real forward method
                        # (its qualname should contain "forward" but not "check_model_inputs").
                        inner_qualname = getattr(cell_contents, "__qualname__", "")
                        if "forward" in inner_qualname and "check_model_inputs" not in inner_qualname:
                            type(model_obj).forward = cell_contents
                            print("* Unwrapped check_model_inputs decorator from forward() (via closure)")
                            break

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def _has_fp8_layers(self) -> bool:
        """Check if the model contains FP8Linear layers."""
        model = self.model
        if isinstance(model, PeftModel):
            model = model.base_model.model
        return any(
            "FP8Linear" in module.__class__.__name__
            for _, module in model.named_modules()
        )

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Quantized models (BNB 4-bit) and FP8 models need special handling:
        # we must reload the base model in full precision to merge the LoRA adapters,
        # because merge_and_unload() adds the delta directly to base weights, which
        # doesn't work with quantized or FP8 weight formats.
        if self.settings.quantization == QuantizationMethod.BNB_4BIT or self._has_fp8_layers():
            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            original_model = self.model
            self.model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=original_model.dtype,
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.settings.model),
            )

            # Convert FP8 layers to BF16 for merging (the saved model needs standard weights).
            self._convert_fp8_to_bf16_for_merge()
            self._unwrap_restrictive_forward_decorators()

            base_model = self.model
            self.model = original_model

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """
        current_model = getattr(self.model.config, "name_or_path", None)
        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation)
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        dtype = self.model.dtype

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        empty_cache()

        quantization_config = self._get_quantization_config(str(dtype).split(".")[-1])

        # Build kwargs, only include quantization_config if it's not None
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.settings.model),
            **extra_kwargs,
        )

        # Detect FP8 layers (kept in FP8 for efficient inference).
        self._detect_fp8()
        self._unwrap_restrictive_forward_decorators()

        self._apply_lora()

        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_moe_info(self, layer_index: int) -> tuple[Module, ModuleList] | None:
        """Returns (gate, experts) for a MoE layer, or None for dense layers."""
        layer = self.get_layers()[layer_index]

        # Qwen3 MoE.
        with suppress(Exception):
            return (layer.mlp.gate, layer.mlp.experts)

        # Phi-3.5-MoE, MiniMax.
        with suppress(Exception):
            return (layer.block_sparse_moe.gate, layer.block_sparse_moe.experts)

        # Granite MoE.
        with suppress(Exception):
            return (layer.moe.gate, layer.moe.experts)

        return None

    def get_layer_modules(
        self,
        layer_index: int,
        expert_mask: dict[int, list[int]] | None = None,
    ) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]
        selected_experts = expert_mask.get(layer_index) if expert_mask else None
        selected = set(selected_experts) if selected_experts is not None else None

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for i, expert in enumerate(layer.mlp.experts):  # ty:ignore[possibly-missing-attribute, not-iterable]
                if selected is None or i in selected:
                    try_add("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for i, expert in enumerate(layer.block_sparse_moe.experts):  # ty:ignore[possibly-missing-attribute, not-iterable]
                if selected is None or i in selected:
                    try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for i, expert in enumerate(layer.moe.experts):  # ty:ignore[possibly-missing-attribute, not-iterable]
                if selected is None or i in selected:
                    try_add("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_modules(0).keys())

    def get_expert_activations(
        self, prompts: list[Prompt]
    ) -> tuple[dict[int, Tensor], int]:
        """
        Profiles MoE router activations to determine which experts are selected
        for the given prompts. Returns ({layer_index: Tensor(num_experts)}, total_tokens)
        where the tensor contains per-expert activation counts.
        """
        layers = self.get_layers()
        activation_counts: dict[int, Tensor] = {}
        token_counts: list[int] = [0]
        hooks = []

        for layer_index in range(len(layers)):
            moe_info = self.get_moe_info(layer_index)
            if moe_info is None:
                continue
            gate, experts = moe_info
            num_experts = len(experts)
            activation_counts[layer_index] = torch.zeros(num_experts)

            # Read top-k and routing bias from the MoE block
            # (attribute names vary by architecture).
            layer = layers[layer_index]
            top_k = None
            routing_bias = None
            for block_attr in ["block_sparse_moe", "mlp", "moe"]:
                with suppress(Exception):
                    block = getattr(layer, block_attr)
                    for attr in ["top_k", "num_experts_per_tok"]:
                        with suppress(Exception):
                            top_k = getattr(block, attr)
                            break
                    # Some models (e.g. MiniMax) apply a correction bias after
                    # the activation function but before top-k selection.
                    with suppress(Exception):
                        routing_bias = block.e_score_correction_bias
                    break
            if top_k is None:
                top_k = 8  # Fallback default.

            def make_hook(li: int, ne: int, k: int, bias: Tensor | None):  # ty:ignore[no-any-explicit]
                def hook_fn(module: Module, args: tuple, output: Tensor) -> None:  # ty:ignore[no-any-explicit]
                    # Gate output shape: (batch*seq, num_experts) - raw logits.
                    # When a routing bias exists (e.g. MiniMax e_score_correction_bias),
                    # we must apply sigmoid + bias to replicate the actual top-k selection.
                    # Without bias, raw logits suffice since activation functions are monotonic.
                    scores = output.float()
                    if bias is not None:
                        scores = torch.sigmoid(scores) + bias.float().to(scores.device)
                    n_tokens = scores.shape[0]
                    if li == 0:
                        token_counts[0] += n_tokens
                    _, top_indices = torch.topk(scores, k, dim=-1)
                    counts = torch.bincount(
                        top_indices.reshape(-1).to(torch.int64), minlength=ne
                    ).float().cpu()
                    activation_counts[li] += counts

                return hook_fn

            hooks.append(gate.register_forward_hook(make_hook(layer_index, num_experts, top_k, routing_bias)))

        if not hooks:
            return {}, 0

        try:
            self.generate(prompts, max_new_tokens=1)
        finally:
            for hook in hooks:
                hook.remove()

        return activation_counts, token_counts[0]

    def get_expert_activations_batched(
        self, prompts: list[Prompt]
    ) -> tuple[dict[int, Tensor], int]:
        """Batched version of get_expert_activations."""
        combined_counts: dict[int, Tensor] = {}
        total_tokens = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_counts, batch_tokens = self.get_expert_activations(batch)
            total_tokens += batch_tokens
            for layer_index, counts in batch_counts.items():
                if layer_index not in combined_counts:
                    combined_counts[layer_index] = torch.zeros_like(counts)
                combined_counts[layer_index] += counts

        return combined_counts, total_tokens

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        expert_mask: dict[int, list[int]] | None = None,
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index, expert_mask).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = cast(float, abs(layer_index - params.max_weight_position))

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # Defensive fallback for modules without LoRA adapters.
                    # Under normal flow, _apply_lora() wraps
                    # all targeted modules, so this should not trigger. It exists as a
                    # safeguard in case PEFT fails to match a module by name.
                    if not hasattr(module, "base_layer"):
                        # Direct weight modification fallback (slower, triggers reload).
                        # Unlike the LoRA path which reads from immutable base_layer weights,
                        # this modifies weights in-place. A second call without reload would
                        # compound on already-modified weights, producing wrong results.
                        assert not self.needs_reload, (
                            "Module without LoRA adapter has already been directly modified. "
                            "Call reset_model() before re-abliterating."
                        )
                        # FP8 modules cannot be modified in-place (need scale factors
                        # for dequantization and re-quantization is lossy). This fallback
                        # only works for standard float modules.
                        assert not any(
                            hasattr(module, attr)
                            for attr in ("weight_scale_inv", "weight_scale", "w_scale")
                        ), (
                            f"FP8 module {module.__class__.__name__} has no LoRA adapter. "
                            "Direct weight modification is not supported for FP8 layers."
                        )
                        self.needs_reload = True
                        v = layer_refusal_direction.to(module.weight.device)
                        W = module.weight.data.float()

                        if self.settings.row_normalization == RowNormalization.FULL:
                            W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                            W_norm = F.normalize(W, p=2, dim=1)
                            proj = v @ W_norm
                            delta = -weight * torch.outer(v, proj)
                            W_adjusted = W_norm + delta
                            W_adjusted = F.normalize(W_adjusted, p=2, dim=1)
                            W = W_adjusted * W_row_norms
                        elif self.settings.row_normalization == RowNormalization.PRE:
                            W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                            W_norm = F.normalize(W, p=2, dim=1)
                            proj = v @ W_norm
                            delta = -weight * torch.outer(v, proj)
                            W = W + W_row_norms * delta
                        else:
                            proj = v @ W
                            delta = -weight * torch.outer(v, proj)
                            W = W + delta

                        module.weight.data = W.to(module.weight.dtype)
                        continue

                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    # Handles FP8 (block-wise scaling), BNB 4-bit, and standard weights.
                    W = self._dequantize_weight(module)

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = self.peft_config.r
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [prompt + self.response_prefix for prompt in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        layers = self.get_layers()
        collected: list[Tensor] = []
        hooks = []
        # Number of entries expected from the prefill pass:
        # 1 embedding (pre-hook) + len(layers) layer outputs (post-hooks).
        n_expected = len(layers) + 1
        # Flag to stop collecting after the first complete forward pass,
        # preventing decode steps or chunked prefill from corrupting the list.
        collecting = True

        # Register forward hooks on decoder layers to capture hidden states.
        # This is needed as a fallback for models that use OutputRecorder
        # (set up by check_model_inputs) instead of manually collecting
        # hidden states in their forward method (e.g., MiniMax M2.5).
        # Use the first layer's device as the common device for collected tensors.
        collect_device = next(layers[0].parameters()).device

        # Pre-hook on first layer captures the embedding output.
        def pre_hook(module: Module, args: tuple[Any, ...]) -> None:
            if collecting:
                collected.append(args[0][:, -1, :].detach().to(collect_device))

        hooks.append(layers[0].register_forward_pre_hook(pre_hook))

        # Post-hooks capture each layer's output.
        def make_post_hook() -> Any:
            def post_hook(module: Module, args: tuple[Any, ...], output: Any) -> None:
                nonlocal collecting
                if not collecting:
                    return
                # Layer output may be a bare tensor, a tuple, or a dataclass
                # (e.g. BaseModelOutputWithPast) that supports [0] indexing.
                hs = output if isinstance(output, torch.Tensor) else output[0]
                collected.append(hs[:, -1, :].detach().to(collect_device))
                # Stop collecting once we have all prefill entries.
                if len(collected) >= n_expected:
                    collecting = False

            return post_hook

        for layer in layers:
            hooks.append(layer.register_forward_hook(make_post_hook()))

        try:
            _, outputs = self.generate(
                prompts,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # This cast is valid because GenerateDecoderOnlyOutput is the return type
            # of model.generate with return_dict_in_generate=True.
            outputs = cast(GenerateDecoderOnlyOutput, outputs)

            # Check if generate() returned usable hidden states.
            # Some models return a non-None tuple whose elements are None.
            has_hidden_states = (
                outputs.hidden_states is not None
                and len(outputs.hidden_states) > 0
                and outputs.hidden_states[0] is not None
            )

            if has_hidden_states:
                # Standard path: model natively supports output_hidden_states.
                # Move all tensors to a common device for multi-GPU compatibility.
                hidden_states = cast(
                    tuple[tuple[FloatTensor]], outputs.hidden_states
                )[0]
                residuals = torch.stack(
                    [
                        layer_hidden_states[:, -1, :].to(collect_device)
                        for layer_hidden_states in hidden_states
                    ],
                    dim=1,
                )
            else:
                # Fallback path: use hook-collected hidden states.
                residuals = torch.stack(collected[:n_expected], dim=1)
        finally:
            for hook in hooks:
                hook.remove()

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            return torch.clamp(residuals, -thresholds, thresholds)

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # This cast is valid because we passed output_scores=True above.
        logits = cast(tuple[FloatTensor], outputs.scores)[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
