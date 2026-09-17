#!/bin/bash
# Run the OpenBCRidgeCorner decks (fire mesh off/on) and check the corner
# behavior at the mixed inflow/outflow corners of the Open-BC ridge case.
#
#   [MPIRUN="mpirun -np 2"] [PYTHON=python3] ./run_openbccorner.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_openbccorner.sh x      # check only, on existing output

set -u
EXE=${1:?usage: run_openbccorner.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="nofire fire"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_openbccorner.py $VARIANTS
