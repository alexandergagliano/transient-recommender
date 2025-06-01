#!/bin/bash

# Transient Recommender Linode VPS Deployment Script
# For transientrecommender.org

set -e

echo "Transient Recommender - Linode VPS Deployment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "Error: Please run this script from the transient_recommender_server directory"
    exit 1
fi

echo "Choose your deployment method:"
echo "1) Docker Compose (Recommended)"
echo "2) Manual Python deployment"
echo "3) Local testing"

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Setting up Docker Compose deployment..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "Docker is not installed. Installing Docker..."
            echo "Run these commands:"
            echo "sudo apt update && sudo apt install -y docker.io docker-compose"
            echo "sudo systemctl start docker && sudo systemctl enable docker"
            echo "sudo usermod -aG docker \$USER"
            echo "Then log out and back in, and run this script again."
            exit 1
        fi
        
        # Check if Docker Compose is installed
        if ! command -v docker-compose &> /dev/null; then
            echo "Docker Compose not found. Please install it first."
            exit 1
        fi
        
        # Build and run
        echo "Building Docker image..."
        docker-compose build
        
        echo "Starting services..."
        docker-compose up -d
        
        echo "Services started successfully!"
        echo "Application running on port 8080"
        echo ""
        echo "Next steps:"
        echo "1. Point transientrecommender.org DNS to this server's IP"
        echo "2. Configure nginx reverse proxy for SSL/domain routing"
        echo "3. Visit: http://your-server-ip:8080"
        ;;
    2)
        echo "Setting up manual Python deployment..."
        
        # Create virtual environment if needed
        if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv .venv
        fi
        
        echo "Installing dependencies..."
        source .venv/bin/activate
        pip install -r requirements.txt
        
        echo "Starting application..."
        echo "Application will run on port 5000"
        echo "Press Ctrl+C to stop"
        python app.py
        ;;
    3)
        echo "Starting local testing environment..."
        
        # Install dependencies if needed
        if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt
        else
            source .venv/bin/activate
        fi
        
        echo "Starting development server..."
        python app.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Setup Notes:"
echo "============"
echo "• Database: app.db contains your transient data"
echo "• DNS: Point transientrecommender.org to your server IP"
echo "• SSL: Use nginx with Let's Encrypt for HTTPS"
echo "• Monitoring: Check logs with 'docker-compose logs' (Docker) or application logs"
echo ""
echo "For nginx configuration and SSL setup, see DEPLOYMENT.md" 