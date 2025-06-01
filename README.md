# Transient Recommender - Web Application

🔗 **GitHub Repository**: https://github.com/alexandergagliano/transient-recommender

A production-ready web application for astronomical transient discovery and recommendation, featuring machine learning-powered suggestions, target management, and collaborative user features.

## 🌟 Features

- **ML-Powered Recommendations**: Real-time transient recommendations using machine learning (0.03s response time)
- **User Authentication**: Complete registration, login, and password reset system
- **Target Management**: Personal target lists with tagging and notes
- **Finder Charts**: Automated finder chart generation for observations
- **Audio Notes**: Voice memo support for transient observations  
- **Admin Dashboard**: User management and system administration
- **Production Ready**: Docker deployment with nginx and SSL support

## 🚀 Quick Start

### Clone Repository
```bash
git clone https://github.com/alexandergagliano/transient-recommender.git
cd transient-recommender
```

### Local Development
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Visit `http://localhost:5000` to access the application.

## 📊 System Specifications

**Current Production Setup:**
- **Database**: SQLite (299MB) with 25,515+ transient objects
- **Performance**: ML recommendations in ~0.03 seconds
- **Features**: Complete user authentication, target management, finder charts
- **Deployment**: Docker-ready with nginx and SSL configuration

## 🎯 Linode VPS Deployment

### Option A: Docker Deployment (Recommended)
```bash
docker-compose --profile production up -d
```

### Option B: Interactive Deployment
```bash
chmod +x deploy.sh
./deploy.sh
```

### System Requirements
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 10GB minimum (for database and logs)
- **CPU**: 1 core minimum, 2 cores recommended
- **OS**: Ubuntu 20.04 LTS or newer

For detailed deployment instructions, see `DEPLOYMENT.md`.

## 🔧 Configuration

### Environment Variables
```bash
LOG_LEVEL=INFO
FEATURE_BANK_PATH=data/feature_bank.csv
PORT=5000
SECRET_KEY=your-secret-key-here
```

### Email Configuration (Optional)
For password reset functionality, configure SMTP settings in `app.py`:
```python
app.config['MAIL_SERVER'] = 'smtp.your-provider.com'
app.config['MAIL_USERNAME'] = 'your-email@domain.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
```

## 📡 API Endpoints

### Authentication
- `GET /`: Main dashboard (requires login)
- `GET /register`: User registration page
- `POST /register`: Process registration
- `GET /login`: Login page
- `POST /login`: Process login
- `GET /logout`: Logout user
- `GET /profile`: User profile management
- `POST /reset_password_request`: Request password reset
- `POST /reset_password/<token>`: Reset password with token

### Recommendations & Voting
- `GET /recommendations`: Get ML-powered transient recommendations
- `POST /vote`: Submit vote (like, dislike, target) for transient
- `POST /skip`: Mark transient as skipped
- `GET /targets`: View user's target list
- `POST /remove_target`: Remove object from targets

### Data & Analysis
- `GET /finder_chart/<ztfid>`: Generate/view finder chart
- `POST /audio_note/<ztfid>`: Upload audio note
- `GET /audio_note/<ztfid>`: Play audio note
- `GET /stats`: User statistics and analytics

### Admin (Admin users only)
- `GET /admin`: Admin dashboard
- `POST /admin/users`: User management
- `POST /update_feature_bank`: Update ML feature bank

## 🔒 Security Features

- **Password Security**: bcrypt hashing with salt
- **Session Management**: Flask-Login with secure session cookies
- **CSRF Protection**: Built-in CSRF token validation
- **SQL Injection Prevention**: SQLAlchemy ORM parameter binding
- **File Upload Security**: Restricted file types and validation
- **Rate Limiting**: Protection against abuse (configurable)

## 📈 Performance & Monitoring

**Optimizations Included:**
- Database indexing for fast queries
- Efficient ML feature processing
- Static file caching
- Gzip compression (nginx)
- Connection pooling

**Monitoring Ready:**
- Comprehensive logging
- Error tracking
- Performance metrics
- Resource usage monitoring

## 🛠️ Development

### Project Structure
```
transient_recommender/
├── app/                    # Flask application
│   ├── static/            # CSS, JS, images
│   ├── templates/         # HTML templates
│   └── models.py          # Database models
├── data/                  # Feature bank and data files
├── finder_charts/         # Generated finder charts
├── docker-compose.yml     # Container orchestration
├── Dockerfile            # Container definition
├── nginx.conf            # Web server configuration
├── deploy.sh             # Deployment script
└── requirements.txt      # Python dependencies
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## 🚀 Production Deployment

The application includes complete production infrastructure:

- ✅ **Docker**: Containerized deployment
- ✅ **nginx**: Reverse proxy with SSL
- ✅ **SSL/HTTPS**: Let's Encrypt integration
- ✅ **Database**: SQLite with backup strategies
- ✅ **Monitoring**: Logging and error tracking
- ✅ **Security**: Production security headers

## 📚 Documentation

- `DEPLOYMENT.md` - Comprehensive deployment guide
- `USER_GUIDE.md` - User interface and features guide
- API documentation available at `/docs` when running

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to deploy?** 
1. Clone the repository
2. Run `./deploy.sh` 
3. Follow the interactive prompts
4. Access your application at your domain with HTTPS! 