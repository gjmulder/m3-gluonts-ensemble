#!/bin/sh

. /var/tmp/dataset_version.sh
/home/mulderg/bin/chk_mongo.sh "${DATASET}-${VERSION}" | grep -v MongoDB
