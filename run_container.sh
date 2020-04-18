#!/bin/bash

# Build image
docker build -t gbm .

# Run container
docker run --name gb_model -p 8000:5000 -d --rm gbm

# Show logs
docker logs gb_model

