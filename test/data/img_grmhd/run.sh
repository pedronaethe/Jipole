#!/bin/bash
set -e

# Clone ipole
git clone https://github.com/AFD-Illinois/ipole.git

# Compile ipole
cd ipole
make -j 

