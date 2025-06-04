#!/usr/bin/env python3
"""Script to check user statistics from command line."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models import User
from app.database import SessionLocal
from datetime import datetime, timedelta

def main():
    db = SessionLocal()
    try:
        # Total users
        total = db.query(User).count()
        
        # Active users
        active = db.query(User).filter(User.is_active == True).count()
        
        # Admin users
        admins = db.query(User).filter(User.is_admin == True).count()
        
        # Recent users
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent = db.query(User).filter(User.created_at >= week_ago).count()
        
        # Users with data consent
        consenting = db.query(User).filter(User.data_sharing_consent == True).count()
        
        # Users with science interests
        with_interests = db.query(User).filter(User.science_interests.isnot(None)).count()
        
        print("=" * 40)
        print("        USER STATISTICS")
        print("=" * 40)
        print(f"📊 Total users:              {total}")
        print(f"✅ Active users:             {active}")
        print(f"❌ Inactive users:           {total - active}")
        print(f"👑 Admin users:              {admins}")
        print(f"🆕 New users (last 7 days):  {recent}")
        print(f"🤝 Users with data consent:  {consenting}")
        print(f"🔬 Users with sci interests: {with_interests}")
        print("=" * 40)
        
        # Show some user details if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--details":
            print("\n📋 USER DETAILS:")
            print("-" * 40)
            users = db.query(User).limit(10).all()
            for user in users:
                status = "✅ Active" if user.is_active else "❌ Inactive"
                admin = " 👑" if user.is_admin else ""
                print(f"{user.username:<20} {status}{admin}")
            if total > 10:
                print(f"... and {total - 10} more users")
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return 1
        
    finally:
        db.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 