#!/bin/bash

find .  ! -name 'delete_non_escential.sh' \
        ! -name 'script_SMCFPL_run.py' \
        ! -name 'filter_intracong_log.py' \
        ! -name 'LICENSE.md' \
        ! -name 'README.md' \
        ! -name 'setup.py' \
        ! -name 'requirements.txt' \
        ! -name 'TODO.todo' \
        ! -path 'InputData*' \
        ! -path 'docs*' \
        ! -path 'TempData_39Bus_v7*' \
        -delete