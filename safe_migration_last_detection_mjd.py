#!/usr/bin/env python3
"""
SAFE migration script for adding last_detection_mjd column to feature_bank table.
This uses the app's own database configuration to ensure compatibility.
"""

import sys
import os

# Add the current directory to Python path so we can import the app modules
sys.path.insert(0, os.path.dirname(__file__))

# Set up the app path
app_path = os.path.join(os.path.dirname(__file__), 'app')
if app_path not in sys.path:
    sys.path.insert(0, app_path)

def safe_migrate():
    """Safely add last_detection_mjd column using app's database configuration."""
    
    try:
        # Import the app's database configuration  
        from app.database import engine, SessionLocal
        from app.models import FeatureBank
        import sqlalchemy
        from sqlalchemy import text
        
        print("🔍 Connecting to database using app configuration...")
        
        # Create a session
        session = SessionLocal()
        
        try:
            # Check if column already exists by trying to query it
            try:
                result = session.execute(text("SELECT last_detection_mjd FROM feature_bank LIMIT 1"))
                print("✅ Column 'last_detection_mjd' already exists. No migration needed.")
                return True
            except sqlalchemy.exc.OperationalError as e:
                if "no such column" in str(e):
                    print("📝 Column 'last_detection_mjd' not found. Adding it now...")
                else:
                    raise e
            
            # Add the column
            print("🔧 Adding last_detection_mjd column to feature_bank table...")
            session.execute(text("""
                ALTER TABLE feature_bank 
                ADD COLUMN last_detection_mjd FLOAT
            """))
            
            session.commit()
            
            print("✅ Successfully added 'last_detection_mjd' column")
            
            # Verify the column was added
            result = session.execute(text("""
                SELECT COUNT(*) FROM feature_bank 
                WHERE last_detection_mjd IS NULL
            """))
            count = result.scalar()
            print(f"✅ Column added successfully. {count} records currently have NULL values (expected).")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()
            
    except ImportError as e:
        print(f"❌ Could not import app modules: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting SAFE database migration...")
    print("This will add the 'last_detection_mjd' column to the feature_bank table")
    print("without affecting any existing data.")
    print()
    
    response = input("⚠️  IMPORTANT: Have you backed up your database? (y/N): ")
    if response.lower() != 'y':
        print("❌ Please backup your database first:")
        print("   cp app.db app.db.backup.$(date +%Y%m%d_%H%M%S)")
        sys.exit(1)
    
    if safe_migrate():
        print("\n🎉 Migration completed successfully!")
        print("The 'last_detection_mjd' column has been added to the feature_bank table.")
        print("You can now restart your server to use the new features.")
    else:
        print("\n💥 Migration failed. Please check the errors above.")
        print("Your database backup is available for restoration if needed.")
        sys.exit(1) 