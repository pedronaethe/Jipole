#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for test_dir in "$SCRIPT_DIR"/*/; do
    if [[ -x "$test_dir/run.sh" ]]; then
        echo
        echo "Running: $(basename "$test_dir")"

        (
            cd "$test_dir"
            ./run.sh
        )
    fi
done

echo
echo "All tests passed"
