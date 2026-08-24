#!/usr/bin/env bash
# Run procagen3d make on every image + GLB pair under assets/3d_glb.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_all_3d_glb.sh [options] [-- extra make args]

Discover every image + GLB pair under assets/3d_glb and run
`uv run procagen3d make` for each case.

Defaults:
  --backend      codex
  --granularity  surface
  --output       <repo>/outputs
  --root         <repo>/assets/3d_glb

Options:
  -h, --help              show this help
  -n, --dry-run           print commands without running them
  -o, --output DIR        workspace output directory
  -r, --root DIR          example collection root
  -b, --backend NAME      coding-agent backend
  -g, --granularity LEVEL geometry detail profile
      --skip-existing     skip cases whose workspace already has run_report.json
      --                  extra arguments forwarded to every `procagen3d make`

Environment overrides: BACKEND, GRANULARITY, OUTPUT, ASSETS_ROOT.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

backend="${BACKEND:-codex}"
granularity="${GRANULARITY:-surface}"
output="${OUTPUT:-$repo_root/outputs}"
assets_root="${ASSETS_ROOT:-$repo_root/assets/3d_glb}"
dry_run=0
skip_existing=0
extra_make_args=()

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -n|--dry-run)
      dry_run=1
      shift
      ;;
    -o|--output)
      output="$2"
      shift 2
      ;;
    -r|--root)
      assets_root="$2"
      shift 2
      ;;
    -b|--backend)
      backend="$2"
      shift 2
      ;;
    -g|--granularity)
      granularity="$2"
      shift 2
      ;;
    --skip-existing)
      skip_existing=1
      shift
      ;;
    --)
      shift
      extra_make_args=("$@")
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "unexpected argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$assets_root" ]]; then
  echo "example root not found: $assets_root" >&2
  exit 1
fi

cd "$repo_root"

cases_tsv="$(
  uv run procagen3d examples --root "$assets_root" | python3 -c '
import json
import sys

pairs = json.load(sys.stdin)
if not pairs:
    raise SystemExit("no image + GLB pairs found")
for pair in pairs:
    print("\t".join((pair["name"], pair["image"], pair["glb"])))
'
)"

total="$(printf '%s\n' "$cases_tsv" | grep -c .)"
passed_names=()
review_names=()
failed_names=()
skipped_names=()
index=0

echo "Running $total case(s) from $assets_root"
echo "backend=$backend granularity=$granularity output=$output"
if ((${#extra_make_args[@]})); then
  echo "extra make args: ${extra_make_args[*]}"
fi
echo

while IFS=$'\t' read -r name image glb; do
  index=$((index + 1))
  echo "==> [$index/$total] $name"

  if ((skip_existing)); then
    existing="$(
      python3 -c '
import json
import sys
from pathlib import Path

name, output = sys.argv[1], Path(sys.argv[2])
prefix = name.replace("_", "-")
if not output.is_dir():
    raise SystemExit(0)
for path in sorted(output.iterdir()):
    report = path / "run_report.json"
    if not report.is_file():
        continue
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    workspace = Path(str(payload.get("workspace") or path))
    if workspace.name.startswith(prefix):
        print(path)
        break
' "$name" "$output" || true
    )"
    if [[ -n "$existing" ]]; then
      echo "skip existing workspace: $existing"
      skipped_names+=("$name")
      echo
      continue
    fi
  fi

  set -- uv run procagen3d make \
    "$image" \
    "$glb" \
    --backend "$backend" \
    --granularity "$granularity" \
    --output "$output"
  if ((${#extra_make_args[@]})); then
    set -- "$@" "${extra_make_args[@]}"
  fi

  printf '    '
  printf '%q ' "$@"
  echo

  if ((dry_run)); then
    echo "dry-run: skipped"
    skipped_names+=("$name")
    echo
    continue
  fi

  set +e
  "$@"
  status=$?
  set -e

  case "$status" in
    0)
      echo "ok: $name"
      passed_names+=("$name")
      ;;
    2)
      echo "needs-review: $name"
      review_names+=("$name")
      ;;
    *)
      echo "failed ($status): $name"
      failed_names+=("$name")
      ;;
  esac
  echo
done <<<"$cases_tsv"

echo "Summary"
echo "  passed:       ${#passed_names[@]} ${passed_names[*]:+(${passed_names[*]})}"
echo "  needs-review: ${#review_names[@]} ${review_names[*]:+(${review_names[*]})}"
echo "  failed:       ${#failed_names[@]} ${failed_names[*]:+(${failed_names[*]})}"
echo "  skipped:      ${#skipped_names[@]} ${skipped_names[*]:+(${skipped_names[*]})}"

if ((${#failed_names[@]})); then
  exit 1
fi
if ((${#review_names[@]})); then
  exit 2
fi
exit 0
