#!/bin/bash

# Transient Recommender - Update Deployment Script
# For updating the system with new feature extraction improvements

set -e

echo "🚀 Transient Recommender - Update Deployment"
echo "=============================================="
echo "This will update your system with the new feature extraction improvements"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    print_error "Please run this script from the transient_recommender_server directory"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Starting update deployment process..."

# Step 1: Create backup
print_status "Step 1: Creating backup..."
BACKUP_DIR=~/backups/$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

if [ -f "app.db" ]; then
    cp app.db $BACKUP_DIR/app.db.backup
    print_success "Database backed up to $BACKUP_DIR/app.db.backup"
else
    print_warning "No app.db found in current directory"
fi

# Step 2: Stop current services
print_status "Step 2: Stopping current services..."
docker-compose down || print_warning "No services were running"

# Step 3: Run database migration
print_status "Step 3: Running database migration..."
if [ -f "safe_migration.py" ]; then
    python3 safe_migration.py
    if [ $? -eq 0 ]; then
        print_success "Database migration completed successfully"
    else
        print_error "Database migration failed"
        exit 1
    fi
else
    print_error "safe_migration.py not found. Please ensure the migration script is in the current directory."
    exit 1
fi

# Step 4: Build new Docker image
print_status "Step 4: Building new Docker image..."
docker-compose build --no-cache
if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 5: Start services
print_status "Step 5: Starting updated services..."
docker-compose up -d
if [ $? -eq 0 ]; then
    print_success "Services started successfully"
else
    print_error "Failed to start services"
    exit 1
fi

# Step 6: Health check
print_status "Step 6: Running health check..."
sleep 10  # Give services time to start

# Check if the service is responding
if curl -f http://localhost:8080/ > /dev/null 2>&1; then
    print_success "✅ Application is responding on port 8080"
else
    print_warning "Application may still be starting up. Check with: docker-compose logs -f"
fi

# Step 7: Show status
print_status "Step 7: Showing service status..."
docker-compose ps

echo ""
print_success "🎉 Update deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Test the new features:"
echo "   - Feature extraction with lookback days parameter"
echo "   - Automatic/manual extraction visibility"
echo "   - Admin detailed progress monitoring"
echo ""
echo "2. Monitor logs:"
echo "   docker-compose logs -f"
echo ""
echo "3. Access your application:"
echo "   http://your-domain.com (if DNS configured)"
echo "   http://$(curl -s ifconfig.me):8080 (direct IP access)"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
print_warning "Important: Test all functionality before removing backup!" 