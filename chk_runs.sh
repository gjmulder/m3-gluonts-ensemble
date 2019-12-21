#!/bin/bash

MD5_OLD=`cat /tmp/run_old | md5sum`
. /var/tmp/dataset_version.sh
/home/mulderg/bin/chk_mongo.sh "${DATASET}-${VERSION}" 2>&1 | grep -v MongoDB > /tmp/run_new
MD5_NEW=`cat /tmp/run_new | md5sum`
if [ "$MD5_NEW" != "$MD5_OLD" ]
then
	SUBJECT=`awk '/MASE/ {printf("%.3f "), $NF}' /tmp/run_new`
	mail -s "HyperOpt: $SUBJECT" -a "From: mulder.g@unic.ac.cy" gjmulder@gmail.com < /tmp/run_new
	mv /tmp/run_new /tmp/run_old
fi
