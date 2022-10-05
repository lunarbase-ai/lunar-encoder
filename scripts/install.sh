#!/usr/bin/env bash


PYTHON=$(which python)
if [ $0 != 0 ]; then
	PYTHON=$(which python3)
fi

SCRIPTSDIR="$(dirname "$(readlink -f "$0")")"
ROOTDIR="$(dirname "$SCRIPTSDIR")"
SETUPSCRIPT="${ROOTDIR}/setup.py"

cd "$ROOTDIR"
$PYTHON "$SETUPSCRIPT" install
