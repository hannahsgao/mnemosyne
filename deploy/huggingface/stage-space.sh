#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EMPTY_STAGE_DIRECTORY" >&2
  exit 64
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/../.." && pwd)
stage_directory=$1

if [[ -e "$stage_directory" && ! -d "$stage_directory" ]]; then
  echo "stage target exists and is not a directory: $stage_directory" >&2
  exit 65
fi

if [[ -d "$stage_directory" ]] && [[ -n "$(find "$stage_directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "stage target must be empty: $stage_directory" >&2
  exit 65
fi

mkdir -p "$stage_directory/service/mnemosyne_search"
cp "$script_directory/README.md" "$stage_directory/README.md"
cp "$script_directory/Dockerfile" "$stage_directory/Dockerfile"
cp "$script_directory/.dockerignore" "$stage_directory/.dockerignore"
cp "$script_directory/start.py" "$stage_directory/start.py"
cp "$repository_root/service/pyproject.toml" "$stage_directory/service/pyproject.toml"

service_sources=("$repository_root"/service/mnemosyne_search/*.py)
if [[ ! -e "${service_sources[0]}" ]]; then
  echo "no service Python sources found" >&2
  exit 66
fi
cp "${service_sources[@]}" "$stage_directory/service/mnemosyne_search/"

echo "Staged the Space allowlist at: $stage_directory"
find "$stage_directory" -type f -print | LC_ALL=C sort
