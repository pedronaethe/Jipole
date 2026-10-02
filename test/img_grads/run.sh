#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

#ensure we have python with necessary libraries and julia
#these functions are defined in common.sh
ensure_python
ensure_julia
dump="$(ensure_file sample_dump_SANE_a+0.94_MKS_0900.h5 \
    https://dataverse.harvard.edu/api/access/datafile/12137142 dc8cf9b45136cd9b28bdb8ed197cfd69)"


#redirect logs and images to output folder
rm -rf output
mkdir -p output
ln -s "$dump" output/dump.h5

log "Running Jipole to generate all gradients through AD"
run_jipole grads.toml

#When doing finite differences, dP = h * P, where h = 1.e-5
h_rel=1e-8

# Check what parameters to differentiate to. This will be useful when adding new parameters in the toml, it will automatically detect them.
#as long as we also add them to the KEY array below
params="$(grep -m1 '^wrt *=' grads.toml | sed -e 's/.*\[//' -e 's/\].*//' -e 's/[",]/ /g')"

declare -A KEY=([M_unit]=m_unit [MBH]=MBH [Rhigh]=Rhigh [Rlow]=Rlow [beta_crit]=beta_crit
                [ro]=ro [th]=theta_o [phi]=phi [sourceD]=source_distance_pc)

#here we make a new directory for the finite difference runs
mkdir -p output/fd
: > output/fd/steps.txt

#Loops over the parameters to differentiate
for P in $params; do
    echo "Processing parameter: $P"

    #Using the key table, we find out what the parameter's key is in the toml file
    #if it doesn't exist, shout out
    key="${KEY[$P]:?no par-file key known for gradient parameter $P}"
    #Read the current value from the toml file disregarding extra symbols that might show up there
    value="$(grep -m1 "^$key *=" grads.toml | cut -d= -f2 | cut -d'#' -f1 | tr -d ' ')"

    #Use awk to compute +- with h=1.e-6 * P
    read -r up down step < <(awk -v x="$value" -v h="$h_rel" 'BEGIN {
        s = (x == 0) ? h : h * (x < 0 ? -x : x)
        printf "%.17g %.17g %.17g\n", x + s, x - s, s }')
    echo "$P $step" >> output/fd/steps.txt

    #Now compute the files +- 
    for side in plus minus; do
        v="$up"; [[ $side == minus ]] && v="$down"
        sed -e "s|^$key *=.*|$key = $v|" \
            -e "s|^on *=.*|on = false|" \
            -e "s|^filename *=.*|filename = \"output/fd/${P}_$side.h5\"|" \
            grads.toml > "output/fd/${P}_$side.toml"
        log "FD run: $P $side ($key = $v)"
        run_jipole "output/fd/${P}_$side.toml"
    done
done
"$PYTHON" check.py output/ad.h5 output/fd output/comparison.png
