#!/bin/bash
# Usage: ./run_regtest.sh /path/to/erf_exec
# Runs all four decks then checks the result. SKIP_RUN=1 skips straight to
# the check, if the decks have already been run.
set -e
EXE=$1
DECKS="albini albini_bmst rothermel rothermel_bmst"
export FI_PROVIDER=tcp   # MPICH's default OFI provider fails MPI_Init here even for one rank
if [ -z "$SKIP_RUN" ]; then
    for d in $DECKS; do
        echo "=== running inputs_$d ==="
        "$EXE" "inputs_$d"
    done
fi
python3 check_regtest.py
