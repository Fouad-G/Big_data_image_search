from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    width = Column(Integer)
    height = Column(Integer)


def get_engine(db_path: str = "sqlite:///images.db"):
    return create_engine(db_path, echo=False)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
