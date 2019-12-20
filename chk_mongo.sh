#!/bin/sh

echo 'db.jobs.find({"result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika $1 | awk 'BEGIN {skip=1} /result/ {skip=0} /misc/ {skip=1} {if (! skip) {print}}'
