import json

from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

def add_qa_pair(
    db: Session,
    question: str,
    response: str,
    context: str,
    source_type: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Add a new QA pair to the database
    """
    from database_setup import SessionLocal, QAPair
    db_qa_pair = QAPair(
        question=question,
        response=response,
        context=context,
        source_type=source_type,
        metadata=json.dumps(metadata) if metadata else None
    )
    db.add(db_qa_pair)
    db.commit()
    db.refresh(db_qa_pair)
    return db_qa_pair

def get_qa_pair_by_question(db: Session, question: str):
    """
    Retrieve QA pair by exact question match
    """
    return db.query(QAPair).filter(QAPair.question == question).first()

def get_similar_qa_pairs(db: Session, question: str, limit: int = 5):
    """
    Retrieve similar QA pairs using SQL LIKE
    """
    return db.query(QAPair).filter(QAPair.question.like(f"%{question}%")).limit(limit).all()

def update_qa_pair(
    db: Session,
    qa_pair_id: int,
    response: Optional[str] = None,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Update an existing QA pair
    """
    qa_pair = db.query(QAPair).filter(QAPair.id == qa_pair_id).first()
    if qa_pair:
        if response is not None:
            qa_pair.response = response
        if context is not None:
            qa_pair.context = context
        if metadata is not None:
            qa_pair.metadata = json.dumps(metadata)
        db.commit()
        db.refresh(qa_pair)
    return qa_pair

def delete_qa_pair(db: Session, qa_pair_id: int):
    """
    Delete a QA pair from the database
    """
    qa_pair = db.query(QAPair).filter(QAPair.id == qa_pair_id).first()
    if qa_pair:
        db.delete(qa_pair)
        db.commit()
        return True
    return False 