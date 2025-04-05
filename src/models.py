#---------------------------
# models.py
#---------------------------
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class UserEmbedding(Base):
    __tablename__ = "user_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    embedding = Column(Vector(192), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())