#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_path="${unit_dir}/qwen3.8-sglang.service"
escaped_root="${repo_root//|/\\|}"

mkdir -p "${unit_dir}"
sed "s|@REPO_ROOT@|${escaped_root}|g" \
  "${repo_root}/systemd/qwen3.8-sglang.service.in" \
  | install -m 0644 /dev/stdin "${unit_path}"

systemctl --user daemon-reload
systemctl --user enable --now qwen3.8-sglang.service

echo "Installed and started qwen3.8-sglang.service"
echo "Follow startup with: journalctl --user -u qwen3.8-sglang.service -f"
