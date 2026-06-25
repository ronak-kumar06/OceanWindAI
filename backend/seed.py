import os
from database.config import engine, SessionLocal, Base
from models.all_models import User
from core.security import get_password_hash

def seed_data():
    # Ensure tables are created
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    demo_email = "demo@oceanwind.ai"
    demo_password = "demo_password"
    
    user = db.query(User).filter(User.email == demo_email).first()
    if not user:
        hashed_password = get_password_hash(demo_password)
        new_user = User(email=demo_email, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        print(f"Created demo user: {demo_email} / {demo_password}")
    else:
        print(f"Demo user already exists: {demo_email} / {demo_password}")
        
    db.close()

if __name__ == "__main__":
    seed_data()
