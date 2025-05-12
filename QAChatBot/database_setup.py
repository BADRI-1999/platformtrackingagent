from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData, Table, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from db_utils import add_qa_pair

# Database URL
DB_URL = "sqlite:///qa_database.db"

# Create engine
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

# Define the QA pairs table
class QAPair(Base):
    __tablename__ = "qa_pairs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    response = Column(Text)
    context = Column(Text)
    source_type = Column(String)  # 'realtime' or 'vector'
    mt = Column(Text)  # For storing additional information in JSON format
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_realtime = Column(Integer, default=0)  # Flag for real-time data
    confidence_score = Column(Integer)  # Confidence score for the answer
    vector_embedding = Column(Text, nullable=True)  # Store vector embedding if needed

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")

    db = SessionLocal()
    try:
        # Add a real-time QA pair
        add_qa_pair(
            db=db,
            question="How many sensors we are using in apt 101?",
            response="total of 5 sensors",
            context="Recent legal update on data privacy requirements",
            source_type="realtime",
            metadata={"confidence": 0.95, "last_updated": "2024-01-20"}
        )
    finally:
        db.close() 