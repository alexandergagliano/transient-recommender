# Transient Recommender

**Repository**: https://github.com/alexandergagliano/transient-recommender

Web application for astronomical transient discovery and target selection using machine learning recommendations.

## Features

- Machine learning-powered transient recommendations (0.03s response time)
- User authentication and target management
- Automated finder chart generation
- Audio annotation support for observations
- SQLite database with 25,515+ transient objects
- Production deployment with Docker and nginx

## Installation

### Local Development
```bash
git clone https://github.com/alexandergagliano/transient-recommender.git
cd transient-recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Production Deployment
```bash
# Docker deployment (recommended)
docker-compose --profile production up -d

# Interactive deployment script
chmod +x deploy.sh && ./deploy.sh
```

## System Requirements

- **Memory**: 2GB minimum, 4GB recommended
- **Storage**: 10GB minimum
- **OS**: Ubuntu 20.04 LTS or newer

## Configuration

### Environment Variables
```bash
LOG_LEVEL=INFO
FEATURE_BANK_PATH=data/feature_bank.csv
PORT=5000
SECRET_KEY=your-secret-key
```

### Email (Optional)
Configure SMTP settings in `app.py` for password reset functionality.

## API Endpoints

### Core Functions
- `/`: Main dashboard
- `/recommendations`: ML transient recommendations
- `/vote`: Submit target classifications
- `/targets`: View target list
- `/finder_chart/<ztfid>`: Generate finder charts

### Authentication
- `/register`, `/login`, `/logout`: User management
- `/reset_password_request`: Password reset

### Admin
- `/admin`: User administration
- `/update_feature_bank`: Update ML features

## Architecture

- **Backend**: Flask with SQLAlchemy ORM
- **Database**: SQLite with bcrypt password hashing
- **ML Pipeline**: Real-time feature extraction and scoring
- **Security**: CSRF protection, session management, rate limiting
- **Deployment**: Docker containerization with nginx reverse proxy

## Documentation

See `DEPLOYMENT.md` for detailed deployment instructions. 