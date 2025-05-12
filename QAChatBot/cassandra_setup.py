from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cassandra configuration
CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', 'localhost')
CASSANDRA_PORT = int(os.getenv('CASSANDRA_PORT', '9042'))
CASSANDRA_KEYSPACE = os.getenv('CASSANDRA_KEYSPACE', 'qa_bot')
CASSANDRA_USER = os.getenv('CASSANDRA_USER', 'cassandra')
CASSANDRA_PASSWORD = os.getenv('CASSANDRA_PASSWORD', 'cassandra')

def get_cluster():
    auth_provider = PlainTextAuthProvider(
        username=CASSANDRA_USER,
        password=CASSANDRA_PASSWORD
    )
    
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        auth_provider=auth_provider
    )
    return cluster

def init_cassandra():
    cluster = get_cluster()
    session = cluster.connect()
    
    # Create keyspace
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS qa_bot
        WITH replication = {
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }
    """)
    
    # Use keyspace
    session.set_keyspace(CASSANDRA_KEYSPACE)
    
    # Create tables
    session.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id uuid PRIMARY KEY,
            question text,
            response text,
            context text,
            source_type text,
            metadata text,
            last_updated timestamp,
            created_at timestamp,
            is_realtime boolean,
            confidence_score float,
            vector_embedding text
        )
    """)
    
    # Create index on question for searching
    session.execute("""
        CREATE INDEX IF NOT EXISTS idx_question ON qa_pairs (question)
    """)
    
    return session

def get_session():
    cluster = get_cluster()
    session = cluster.connect(CASSANDRA_KEYSPACE)
    session.row_factory = dict_factory
    return session

if __name__ == "__main__":
    print("Initializing Cassandra database...")
    session = init_cassandra()
    print("Cassandra database initialized successfully!")
    
    # Add a test entry
    from uuid import uuid4
    test_entry = {
        'id': uuid4(),
        'question': "What are the latest legal requirements for data privacy?",
        'response': "As of 2024, organizations must implement...",
        'context': "Recent legal update on data privacy requirements",
        'source_type': "realtime",
        'metadata': json.dumps({'confidence': 0.95}),
        'last_updated': datetime.now(),
        'created_at': datetime.now(),
        'is_realtime': True,
        'confidence_score': 0.95,
        'vector_embedding': None
    }
    
    session.execute("""
        INSERT INTO qa_pairs (
            id, question, response, context, source_type, metadata,
            last_updated, created_at, is_realtime, confidence_score, vector_embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        test_entry['id'], test_entry['question'], test_entry['response'],
        test_entry['context'], test_entry['source_type'], test_entry['metadata'],
        test_entry['last_updated'], test_entry['created_at'],
        test_entry['is_realtime'], test_entry['confidence_score'],
        test_entry['vector_embedding']
    ))
    
    print("Test entry added successfully!") 