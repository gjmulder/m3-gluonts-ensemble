#!/bin/sh

if [ $# -eq 2 ]
then
	WAIT=$1
	REP=$2
else
	WAIT=0
	REP=1
fi
for X in `seq 1 $REP`
do
	echo "==============================================================================="
	date
	(nvidia-smi ;ssh yokojitsu nvidia-smi ; ssh asushimu nvidia-smi )| egrep "python3|Default" | sort -n -k 9,9
	sleep $WAIT
done
