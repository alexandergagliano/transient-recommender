#!/usr/bin/env python3
"""
SAFE migration script for adding is_automatic column.
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
    """Safely add is_automatic column using app's database configuration."""
    
    try:
        # Import the app's database configuration  
        from app.database import engine, SessionLocal
        from app.models import FeatureExtractionRun
        import sqlalchemy
        from sqlalchemy import text
        
        print("🔍 Connecting to database using app configuration...")
        
        # Create a session
        session = SessionLocal()
        
        try:
            # Check if column already exists by trying to query it
            try:
                result = session.execute(text("SELECT is_automatic FROM feature_extraction_runs LIMIT 1"))
                print("✅ Column 'is_automatic' already exists. No migration needed.")
                return
            except sqlalchemy.exc.OperationalError as e:
                if "no such column" in str(e):
                    print("📝 Column 'is_automatic' not found. Adding it now...")
                else:
                    raise e
            
            # Add the column
            session.execute(text("""
                ALTER TABLE feature_extraction_runs 
                ADD COLUMN is_automatic BOOLEAN DEFAULT 1
            """))
            
            # Update existing records
            result = session.execute(text("""
                UPDATE feature_extraction_runs 
                SET is_automatic = 1 
                WHERE is_automatic IS NULL
            """))
            
            session.commit()
            
            print("✅ Successfully added 'is_automatic' column")
            
            # Verify
            count_result = session.execute(text("SELECT COUNT(*) FROM feature_extraction_runs WHERE is_automatic = 1"))
            count = count_result.scalar()
            print(f"✅ Updated {count} existing records to mark as automatic")
            
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
    print("This will add the 'is_automatic' column to your existing database")
    print("without affecting any existing data.")
    print()
    
    if safe_migrate():
        print("\n🎉 Migration completed successfully!")
        print("You can now restart your server to use the new features.")
    else:
        print("\n💥 Migration failed. Please check the errors above.")
        sys.exit(1) 