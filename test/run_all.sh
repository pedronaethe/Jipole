#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GREEN=$'\033[0;32m'
RED=$'\033[0;31m'

#NC defines the no color, in order to reset after printing PASS or FAIL.
NC=$'\033[0m'

passed=()
failed=()
for run in */run.sh; do
    name="$(dirname "$run")"
    if bash "$run"; then
        passed+=("$name")
    else
        failed+=("$name")
    fi
done

echo "Summary:"
for name in "${passed[@]}"; do echo -e "${GREEN}PASS${NC}  $name"; done
for name in "${failed[@]}"; do echo -e "${RED}FAIL${NC}  $name"; done
[[ ${#failed[@]} -eq 0 ]]