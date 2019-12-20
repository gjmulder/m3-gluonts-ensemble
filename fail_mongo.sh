#!/bin/sh -x

echo 'db.jobs.find({"result.status" : "fail"}).toArray()' | mongo --host heika $1 | grep -A 1 exception #| sed 's/\\n/\n/g'
