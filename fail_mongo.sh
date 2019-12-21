#!/bin/sh

echo 'db.jobs.find({"result.status" : "fail"}).toArray()' | mongo --host heika $1 | egrep -A 1 '"type"|exception' | sed 's/\\n/\n/g'
