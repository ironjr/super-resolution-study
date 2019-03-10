#!/bin/bash
set -x

TODAY=$(date +%y%m%d)
DIRNAME=archive/${TODAY}

ITER=0
while [ -d $DIRNAME-$ITER ]; do
    ITER=$(($ITER + 1))
done
DIRNAME=$DIRNAME-$ITER

[[ -d $DIRNAME ]] || mkdir $DIRNAME
[[ -d $DIRNAME/logs ]] || mkdir $DIRNAME/logs
mv checkpoints/* $DIRNAME
mv logs/* $DIRNAME/logs
