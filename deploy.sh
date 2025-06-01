#!/bin/bash

# Transient Recommender One-Click Deployment Script
# For transientrecommender.org

set -e

echo "🚀 Transient Recommender Deployment Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the transient_recommender_server directory"
    exit 1
fi

echo "📋 Choose your deployment method:"
echo "1) Railway.app (Recommended - Easy)"
echo "2) Render.com (Alternative)"
echo "3) DigitalOcean App Platform"
echo "4) Docker Compose (Self-hosted)"
echo "5) Local testing"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚂 Setting up Railway.app deployment..."
        echo ""
        echo "1. Install Railway CLI: npm install -g @railway/cli"
        echo "2. Login: railway login"
        echo "3. Create project: railway init"
        echo "4. Deploy: railway up"
        echo "5. Add custom domain in Railway dashboard: transientrecommender.org"
        echo ""
        echo "Railway config file: railway.toml (already created)"
        ;;
    2)
        echo "🎨 Setting up Render.com deployment..."
        echo ""
        echo "1. Push code to GitHub"
        echo "2. Connect GitHub repo to Render.com"
        echo "3. Render will automatically use render.yaml config"
        echo "4. Add custom domain transientrecommender.org in Render dashboard"
        echo ""
        echo "Render config file: render.yaml (already created)"
        ;;
    3)
        echo "🌊 Setting up DigitalOcean App Platform..."
        echo ""
        echo "1. Push code to GitHub"
        echo "2. Create app in DigitalOcean App Platform"
        echo "3. Connect GitHub repo"
        echo "4. DigitalOcean will auto-detect Dockerfile"
        echo "5. Add domain transientrecommender.org in app settings"
        ;;
    4)
        echo "🐳 Setting up Docker Compose (self-hosted)..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        # Build and run
        echo "Building Docker image..."
        docker-compose build
        
        echo "Starting services..."
        docker-compose up -d
        
        echo "✅ Services started!"
        echo "🌐 Visit: http://localhost:8080"
        echo ""
        echo "To set up domain transientrecommender.org:"
        echo "1. Point your DNS A record to your server IP"
        echo "2. Use nginx proxy (config included in docker-compose.yml)"
        ;;
    5)
        echo "🧪 Starting local testing environment..."
        
        # Install dependencies if needed
        if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt
        else
            source .venv/bin/activate
        fi
        
        echo "Starting development server..."
        python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🔧 Additional Setup Notes:"
echo "========================="
echo "• Make sure your app.db file contains your data"
echo "• Update domain DNS settings to point to your deployment"
echo "• Consider setting up SSL/TLS for production"
echo "• Monitor logs and performance after deployment"
echo ""
echo "📞 Need help? Check the documentation or deployment logs" 