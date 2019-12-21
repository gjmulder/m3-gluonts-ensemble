#!/bin/sh

if [ $# -eq 1 ]
then
	STATUS="ok"
else
	STATUS=$2
fi

echo "model.type, train.MASE, train.sMAPE, test.MASE, test.sMAPE, experiment, start.time, end.time"
#for DB in plos1-m3-001g plos1-m3-002g plos1-m3-003g
for DB in $1
do
	echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB | awk '
/"MASE"/ {
	printf("%10.5f,", $NF)
}

/"sMAPE"/ {
	printf("%10.5f,", $NF)
}

/"exp_key"/ {
	printf("%14s", $NF)
}

/"type"/ {
	printf("%30s,", $NF)
}

/"book_time"/ {
	printf("%37s", $NF)
}

/"refresh_time"/ {
	printf("%36s\n", $NF)
}'
done | sed 's/,,/, /g' | sed 's/ISODate//g' | tr -d "()\""
