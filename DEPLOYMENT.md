# 🚀 Transient Recommender - Linode VPS Deployment Guide
## Deploy to Your Domain with Full Control

This guide provides deployment options for the Transient Recommender on your Linode VPS server.

## ⚡ Quick Start (Recommended)

Run the deployment script:
```bash
./deploy.sh
```

Choose your preferred deployment method and follow the interactive prompts!

## 🎯 Linode VPS Deployment Options

### 1. Docker Deployment (Recommended)

**Why Docker:** Containerized, consistent environment, easy scaling, production-ready.

**Steps:**
1. Clone repository to your Linode server
2. Ensure Docker and Docker Compose are installed
3. Run: `docker-compose --profile production up -d`
4. Configure nginx for SSL and domain routing

**Requirements:**
- Linode VPS with Docker installed
- Domain pointing to your server IP
- SSL certificate (Let's Encrypt recommended)

### 2. Direct Python Deployment

**Why Direct:** Full control, easier debugging, simpler for smaller deployments.

**Steps:**
1. Clone repository to your server
2. Set up Python virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure nginx as reverse proxy
5. Use systemd for process management

## 🌐 Domain & SSL Configuration

### DNS Setup
Point your domain DNS to your Linode server:

```
Type: A
Name: yourdomain.com
Value: [your Linode server IP address]
```

### SSL/TLS with Let's Encrypt
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔧 Server Setup Prerequisites

### System Requirements
- **RAM:** 2GB minimum, 4GB recommended
- **Storage:** 10GB minimum (for database and logs)
- **CPU:** 1 core minimum, 2 cores recommended
- **OS:** Ubuntu 20.04 LTS or newer

### Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nginx
sudo apt install nginx -y

# Install Python (if using direct deployment)
sudo apt install python3 python3-pip python3-venv -y
```

## 🔧 Environment Configuration

### Required Environment Variables
```bash
LOG_LEVEL=INFO
FEATURE_BANK_PATH=data/feature_bank.csv
PORT=8080
SECRET_KEY=your-secret-key-here
```

### Production Optimizations
- Use production database (included: `app.db`)
- Configure proper logging with log rotation
- Set up monitoring and alerts
- Enable gzip compression (included in nginx config)
- Set up automated backup strategy

## 📊 Database Considerations

Your `app.db` file (299MB) contains:
- User accounts and votes
- Feature bank with 25,515 objects
- Audio notes and comments
- Target lists and tags

**Backup Strategy:**
```bash
# Daily database backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp /path/to/app.db /backup/app_backup_$DATE.db
# Keep only last 7 days
find /backup -name "app_backup_*.db" -mtime +7 -delete
```

## 🚀 Performance Optimization

**Current Setup Handles:**
- 25,515 transient objects in feature bank
- Real-time recommendations with machine learning
- File uploads (finder charts, audio notes)
- Multiple concurrent users

**Linode Optimization:**
- Use Linode's high-performance instances
- Enable SSD storage for database
- Configure nginx caching for static files
- Monitor resource usage with Linode's monitoring tools

## 🛠️ Post-Deployment Checklist

1. **Test Core Features:**
   - [ ] User registration/login
   - [ ] Recommendations engine
   - [ ] Target list functionality
   - [ ] Finder chart generation
   - [ ] Audio notes

2. **Security:**
   - [ ] HTTPS enabled with Let's Encrypt
   - [ ] Firewall configured (ufw)
   - [ ] SSH key authentication
   - [ ] Security headers configured in nginx

3. **Performance:**
   - [ ] Page load times < 3 seconds
   - [ ] Database queries optimized
   - [ ] Static files cached and compressed

4. **Monitoring:**
   - [ ] Error logging configured
   - [ ] Uptime monitoring
   - [ ] Resource usage monitoring
   - [ ] Backup automation

## 🆘 Troubleshooting

### Common Issues:

**Domain not working:**
- Check DNS propagation: `nslookup yourdomain.com`
- Verify nginx configuration: `sudo nginx -t`
- Check firewall: `sudo ufw status`

**Application errors:**
- Check Docker logs: `docker-compose logs -f`
- Verify file permissions
- Ensure database file is accessible

**Performance issues:**
- Monitor resource usage: `htop`, `df -h`
- Check nginx access logs
- Optimize database queries

### Useful Commands:
```bash
# Check service status
systemctl status nginx
docker ps

# View logs
tail -f /var/log/nginx/error.log
docker-compose logs -f app

# Restart services
sudo systemctl restart nginx
docker-compose restart
```

## 📈 Monitoring & Maintenance

### Recommended Monitoring:
- **System monitoring:** htop, netdata, or Prometheus
- **Application logs:** centralized logging
- **Uptime monitoring:** simple ping scripts or external services
- **SSL certificate expiry:** certbot handles auto-renewal

### Regular Maintenance:
- Weekly system updates
- Daily database backups
- Monthly log cleanup
- Quarterly security reviews

---

**Ready to deploy?** Run `./deploy.sh` and follow the prompts!

For Linode-specific questions, consult [Linode's documentation](https://www.linode.com/docs/) or their community support. 