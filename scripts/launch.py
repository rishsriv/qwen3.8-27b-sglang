#!/usr/bin/env python3
"""Launch SGLang while keeping the API key out of the process arguments."""

import os
import sys
from functools import wraps

from sglang.launch_server import run_server
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def _enable_qwen_reasoning_effort_compatibility() -> None:
    """Translate OpenAI ``high`` effort to the Qwen template tier ``xhigh``.

    Codex and the OpenAI SDK used by SGLang 0.5.17 can serialize ``high`` in
    Responses events, while the Qwen3.8 chat template names the same tier
    ``xhigh``. Limit the translation to prompt rendering, then restore the
    request so response metadata continues to use the SDK-compatible value.
    """

    original_apply_jinja_template = OpenAIServingChat._apply_jinja_template

    @wraps(original_apply_jinja_template)
    def apply_jinja_template(self, request, tools, is_multimodal):
        original_effort = request.reasoning_effort
        if original_effort == "high":
            request.reasoning_effort = "xhigh"

        try:
            return original_apply_jinja_template(
                self, request, tools, is_multimodal
            )
        finally:
            request.reasoning_effort = original_effort

    OpenAIServingChat._apply_jinja_template = apply_jinja_template


_enable_qwen_reasoning_effort_compatibility()


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
