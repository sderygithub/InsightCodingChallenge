#!/usr/bin/env bash

# Run script for the challenge

# Proper permissions
chmod a+x src/wordcount.c

# @TODO: Would need to check if compiler exist

# Compile
gcc -g src/wordcount.c -o wordcount

# Execute both task
./wordcount