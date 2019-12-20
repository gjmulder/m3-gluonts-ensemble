#!/bin/sh

STATUS="ok"

DBS=`echo "show dbs" | mongo --host heika | awk '/GB$/ {print $1}' | egrep -v "^(admin|config|local)"`
for DB in $DBS
do
	echo "DB name: $DB"
	echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB
#| awk 'BEGIN {skip=1} /result/ {skip=0} /misc/ {skip=1} {if (! skip) {print}}'
done
