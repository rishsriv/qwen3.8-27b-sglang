#!/usr/bin/env python3
"""Launch SGLang while keeping the API key out of the process arguments."""

import os
import sys

from sglang.launch_server import run_server
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.models.dspark import DSparkDraftMixin, gather_and_crop_vocab
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def _enable_quantized_dspark_lm_head() -> None:
    """Route DSpark logits through SGLang quantization for packed NVFP4 heads.

    SGLang 0.5.17 performs a raw matmul in DSparkDraftMixin even when the
    shared target LM head is packed NVFP4. This mirrors the LogitsProcessor
    quantized-head branch and also runs in spawned worker processes.
    """

    def compute_base_logits(self, hidden):
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )

        multiplier = getattr(self, "logits_mup_width_multiplier", None)
        if multiplier:
            hidden = hidden / multiplier

        quant_method = getattr(self.lm_head, "quant_method", None)
        if should_apply_lm_head_quant_method(self.lm_head, quant_method):
            local_logits = quant_method.apply(self.lm_head, hidden)
        else:
            weight = self.lm_head.weight
            local_logits = hidden.to(weight.dtype) @ weight.T

        return gather_and_crop_vocab(local_logits, self.lm_head), None

    DSparkDraftMixin.compute_base_logits = compute_base_logits


_enable_quantized_dspark_lm_head()


def main() -> None:
    api_key = os.environ.pop("SGLANG_API_KEY")
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    server_args.override(source="environment-secret", api_key=api_key)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
