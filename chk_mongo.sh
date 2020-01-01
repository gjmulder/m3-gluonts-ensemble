#!/bin/sh

if [ $# -eq 2 ]
then
	MODEL=$2
	echo "db.jobs.find({\"result.status\" : \"ok\", \"result.cfg.model.type\" : \"${MODEL}\"}).sort( { \"result.loss\": 1} ).limit(1).pretty()"
else
	echo 'db.jobs.find({"result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()'
fi | mongo --host heika $1 | awk 'BEGIN {skip=1} /result/ {skip=0} /misc/ {skip=1} {if (! skip) {print}}'

