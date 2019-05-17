#!/bin/bash

find .  -maxdepth 1 \
        ! -name 'delete_non_escential.sh' \
        ! -name 'docs' \
        ! -name 'filter_intracong_log.py' \
        ! -name 'InputData' \
        ! -name 'LICENSE.md' \
        ! -name 'README.md' \
        ! -name 'requirements.txt' \
        ! -name 'script_SMCFPL_run.py' \
        ! -name 'setup.py' \
        ! -name 'TempData_39Bus_v7' \
        ! -name 'TODO.todo' \
        -delete