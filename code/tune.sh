#!/bin/bash

# Tune
echo "Begin out_channels tunning..."
for((oc=4; oc<=512; oc=oc*2))
do
    python main_run.py --subdir="out_channels" --epochs=1 --out-channels=${oc}
    echo "Finished out_channels"
done