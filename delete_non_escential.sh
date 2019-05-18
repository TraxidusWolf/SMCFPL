#!/bin/bash

find . \
	! -name 'delete_non_escential.sh' \
	! -path "./.git*" \
	! -path "./TempData_39Bus_v7*" \
        ! -path "./InputData*" \
	! -path "./docs*" \
        ! -name "requirements.txt" \
        ! -name 'LICENSE.md' \
        ! -name 'README.md' \
        ! -name 'script_SMCFPL_run.py' \
        ! -name 'filter_intracong_log.py' \
        ! -name 'setup.py' \
        ! -name "TODO.todo" \
	-delete
