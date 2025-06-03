#!/bin/bash

# Fix Algorithm Configuration Issue
# This script helps diagnose and fix the 500 error on the algorithms page

echo "=== Transient Recommender Algorithm Config Fix ==="
echo "This script will help fix the 500 error on the algorithms page"
echo

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found"
    echo "   Please run this script from the transient_recommender_server directory"
    exit 1
fi

echo "✓ Found docker-compose.yml"

# Step 1: Check current container status
echo
echo "=== Step 1: Checking Container Status ==="
if docker-compose ps | grep -q "Up"; then
    echo "✓ Containers are running"
    CONTAINERS_RUNNING=true
else
    echo "• Containers are not running"
    CONTAINERS_RUNNING=false
fi

# Step 2: Run diagnostic script
echo
echo "=== Step 2: Running Diagnostic ==="
if [ "$CONTAINERS_RUNNING" = true ]; then
    echo "Running diagnostic inside Docker container..."
    docker-compose exec web python debug_algorithm_config.py
    DIAGNOSTIC_EXIT_CODE=$?
else
    echo "Starting containers temporarily for diagnostic..."
    docker-compose up -d
    sleep 10
    docker-compose exec web python debug_algorithm_config.py
    DIAGNOSTIC_EXIT_CODE=$?
    docker-compose down
fi

# Step 3: Determine fix needed
echo
echo "=== Step 3: Applying Fix ==="

if [ $DIAGNOSTIC_EXIT_CODE -eq 0 ]; then
    echo "✅ Diagnostic passed - the issue might be intermittent"
    echo "   Try refreshing the algorithms page"
else
    echo "❌ Diagnostic failed - rebuilding containers with latest dependencies"
    
    # Rebuild containers
    echo "Stopping containers..."
    docker-compose down
    
    echo "Rebuilding with fresh dependencies..."
    docker-compose build --no-cache web
    
    echo "Starting containers..."
    docker-compose up -d
    
    # Wait for startup
    echo "Waiting for containers to start..."
    sleep 15
    
    # Test again
    echo "Running post-fix diagnostic..."
    docker-compose exec web python debug_algorithm_config.py
    POST_FIX_EXIT_CODE=$?
    
    if [ $POST_FIX_EXIT_CODE -eq 0 ]; then
        echo "✅ Fix successful!"
        echo "   The algorithms page should now work properly"
    else
        echo "❌ Fix failed - manual intervention required"
        echo "   Please check the logs with: docker-compose logs app"
    fi
fi

# Step 4: Show access information
echo
echo "=== Step 4: Access Information ==="
echo "Your application should be accessible at:"
echo "  - Main app: http://your-server-ip:8080"
echo "  - Algorithms page: http://your-server-ip:8080/algorithms"
echo
echo "If you're still seeing issues:"
echo "  1. Check logs: docker-compose logs web"
echo "  2. Restart: docker-compose restart"
echo "  3. Full rebuild: docker-compose down && docker-compose up --build -d"
echo
echo "=== Fix Complete ===" 