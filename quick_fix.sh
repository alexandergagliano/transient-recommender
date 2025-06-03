#!/bin/bash

echo "=== Quick Algorithm Config Diagnostic ==="
echo "Copying diagnostic script into container and running it..."

# Copy the diagnostic script into the running container
docker cp debug_algorithm_config.py transient-recommender_web_1:/app/debug_algorithm_config.py

# Run the diagnostic inside the container
echo "Running diagnostic..."
docker-compose exec web python debug_algorithm_config.py

echo ""
echo "If the diagnostic shows PyYAML import errors, run:"
echo "  docker-compose down"
echo "  docker-compose build --no-cache web"
echo "  docker-compose up -d" 