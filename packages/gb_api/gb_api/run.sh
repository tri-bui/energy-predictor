#!/bin/bash
exec gunicorn --bind 0.0.0.0:5000 --timeout 0 --worker-tmp-dir /dev/shm --workers=4 --threads=4 --worker-class=gthread --log-file=- run:app

