#!/bin/bash

# Export the host user's UID and GID to be picked up by Docker Compose
# These variables will be visible to the docker-compose command
export UID=$(id -u)
export GID=$(id -g)

# Execute the docker compose command
docker compose up "$@"