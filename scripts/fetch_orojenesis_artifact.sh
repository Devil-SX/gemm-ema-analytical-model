#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${root_dir}/third_party"
out_file="${out_dir}/orojenesis.zip"
url="https://zenodo.org/records/12600121/files/orojenesis.zip?download=1"

mkdir -p "${out_dir}"

if [[ -f "${out_file}" ]]; then
  echo "exists: ${out_file}"
  exit 0
fi

curl -L "${url}" -o "${out_file}"
echo "downloaded: ${out_file}"

