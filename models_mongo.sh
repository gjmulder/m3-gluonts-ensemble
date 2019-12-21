#!/bin/sh

echo 'db.jobs.find({"result.status" : "new"}).sort( { "result.loss": 1} ).pretty()' | mongo --host heika $1 | awk '
BEGIN {
	skip=1
}

/vals/ {
	skip=0
}

/exp_key/ {
	skip=1;
	print "\n"
}

/"model"/ {
	if (! skip) {
		#printf("%s ", $1)
		printf("%s ", $0);
		getline;
		print $NF;
	}
}' | tr -d '[]}"'
