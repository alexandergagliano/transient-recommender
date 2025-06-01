# 🚀 Transient Recommender Deployment Guide
## Deploy to transientrecommender.org with One Click

This guide provides multiple options for deploying the Transient Recommender to your domain `transientrecommender.org`.

## ⚡ Quick Start (Recommended)

Run the deployment script:
```bash
./deploy.sh
```

Choose your preferred deployment method and follow the interactive prompts!

## 🎯 Deployment Options

### 1. Railway.app (Easiest - Recommended)

**Why Railway:** One-click deployment, automatic SSL, easy domain setup, great for scientific applications.

**Steps:**
1. Push your code to GitHub
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize: `railway init`
5. Deploy: `railway up`
6. Add custom domain in Railway dashboard: `transientrecommender.org`

**Cost:** ~$5-20/month depending on usage

### 2. Render.com (Alternative)

**Why Render:** Simple, automatic SSL, good performance, scientific-friendly.

**Steps:**
1. Push code to GitHub
2. Connect repository to Render.com
3. Render auto-detects `render.yaml` configuration
4. Add custom domain `transientrecommender.org` in dashboard

**Cost:** ~$7-25/month

### 3. DigitalOcean App Platform

**Why DigitalOcean:** Reliable, good for data-heavy applications, flexible scaling.

**Steps:**
1. Push code to GitHub
2. Create app in DigitalOcean App Platform
3. Connect GitHub repository
4. Add domain in app settings
5. DigitalOcean handles SSL automatically

**Cost:** ~$5-50/month

### 4. Self-Hosted with Docker

**Why Self-Host:** Full control, cost-effective for high usage, customizable.

**Steps:**
1. Get a VPS (DigitalOcean Droplet, AWS EC2, etc.)
2. Clone repository to server
3. Run: `docker-compose --profile production up -d`
4. Configure DNS to point to your server

**Cost:** ~$5-20/month for VPS

## 🌐 Domain Configuration

### DNS Setup
Point your domain DNS to your deployment:

**For Cloud Providers (Railway, Render, etc.):**
```
Type: CNAME
Name: transientrecommender.org
Value: [provided by your cloud provider]
```

**For Self-Hosted:**
```
Type: A
Name: transientrecommender.org
Value: [your server IP address]
```

### SSL/TLS
- **Cloud providers:** Automatic SSL included
- **Self-hosted:** Use Let's Encrypt or add SSL certificates to nginx

## 🔧 Environment Configuration

### Required Environment Variables
```bash
LOG_LEVEL=INFO
FEATURE_BANK_PATH=data/feature_bank.csv
PORT=8080
```

### Production Optimizations
- Use production database (included: `app.db`)
- Configure proper logging
- Set up monitoring and alerts
- Enable gzip compression (included in nginx config)
- Set up backup strategy for database

## 📊 Database Considerations

Your `app.db` file (299MB) contains:
- User accounts and votes
- Feature bank with 25,515 objects
- Audio notes and comments
- Target lists and tags

**Backup Strategy:**
- Cloud providers: Use their backup services
- Self-hosted: Regular database backups to cloud storage

## 🚀 Performance Notes

**Current Setup Handles:**
- 25,515 transient objects in feature bank
- Real-time recommendations with machine learning
- File uploads (finder charts, audio notes)
- Multiple concurrent users

**Scaling Recommendations:**
- Start with basic plan
- Monitor CPU/memory usage
- Scale up as user base grows
- Consider database optimization for >100K objects

## 🛠️ Post-Deployment Checklist

1. **Test Core Features:**
   - [ ] User registration/login
   - [ ] Recommendations engine
   - [ ] Target list functionality
   - [ ] Finder chart generation
   - [ ] Audio notes

2. **Security:**
   - [ ] HTTPS enabled
   - [ ] Security headers configured
   - [ ] User data protection

3. **Performance:**
   - [ ] Page load times < 3 seconds
   - [ ] Database queries optimized
   - [ ] Static files cached

4. **Monitoring:**
   - [ ] Error logging configured
   - [ ] Uptime monitoring
   - [ ] Performance metrics

## 🆘 Troubleshooting

### Common Issues:

**Domain not working:**
- Check DNS propagation (24-48 hours)
- Verify domain configuration in provider dashboard

**Application errors:**
- Check deployment logs
- Verify environment variables
- Ensure database file is accessible

**Performance issues:**
- Monitor resource usage
- Scale up instance size
- Optimize database queries

### Support Resources:
- Railway: [docs.railway.app](https://docs.railway.app)
- Render: [render.com/docs](https://render.com/docs)
- DigitalOcean: [docs.digitalocean.com](https://docs.digitalocean.com)

## 📈 Monitoring & Analytics

Consider adding:
- Application performance monitoring (APM)
- User analytics (respecting privacy)
- Error tracking (Sentry, Rollbar)
- Uptime monitoring (UptimeRobot, Pingdom)

---

**Ready to deploy?** Run `./deploy.sh` and follow the prompts!

For questions or issues, check the deployment logs or consult the provider documentation. 