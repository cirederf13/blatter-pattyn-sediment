#!/bin/sh
# All transient runs of the paper (the base runs, then the sensitivity tests).  Each run takes about 10 minutes.
# Usage: ./run_all_transient.sh        (runs up to four at a time)
run() { python run_transient.py "$@" > logs/transient_$(echo "$@" | tr ' -' '__').log 2>&1; }
mkdir -p logs
for g in arolla franzjosef gorner; do run $g & done; wait
run gorner --dE 0 --tag _unforced &
run arolla --dE 0 --tag _unforced &
run franzjosef --dE 0 --tag _unforced &
run gorner --dE 100 --tag _dE100 &
wait
run gorner --dE 200 --tag _dE200 &
run gorner --fine-um 20 --tag _D20um &
run gorner --fine-um 125 --tag _D125um &
wait
run arolla --ratio 0.5 --tag _r0.5 &
run arolla --ratio 0.75 --tag _r0.75 &
run gorner --ratio 0.5 --tag _r0.5 &
run gorner --ratio 0.75 --tag _r0.75 &
wait
for l in 1.0 3.0; do
  for g in arolla franzjosef gorner; do run $g --exponent $l --tag _l$l & done
  wait
done
