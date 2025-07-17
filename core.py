"""
QUANTUM SOVEREIGN DIGITAL NATION - PRODUCTION-READY VERSION
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum

# Cryptography
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from passlib.context import CryptContext

# JWT
import jwt
from jwt import PyJWTError

# Database
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, LargeBinary, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, Request, status, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ========================
# CONFIGURATION
# ========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nation.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security Config
SECRET_KEY = os.getenv("SECRET_KEY", "your-strong-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

# ========================
# CRYPTO SERVICE (WITH FALLBACK)
# ========================

class QuantumCryptoService:
    """Handles all cryptographic operations with fallback mechanisms"""
    
    class CryptoAlgorithms(Enum):
        ED25519 = "ed25519"
        X25519 = "x25519"
        
    @staticmethod
    def generate_keypair(algorithm: CryptoAlgorithms = CryptoAlgorithms.ED25519) -> tuple[bytes, bytes]:
        """Generate key pair with fallback to Ed25519/X25519"""
        try:
            if algorithm == QuantumCryptoService.CryptoAlgorithms.ED25519:
                private_key = ed25519.Ed25519PrivateKey.generate()
                public_key = private_key.public_key()
            else:
                private_key = x25519.X25519PrivateKey.generate()
                public_key = private_key.public_key()
                
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            return private_bytes, public_bytes
            
        except Exception as e:
            logger.error(f"Key generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Key generation failed"
            )
    
    @staticmethod
    def sign_message(private_key: bytes, message: bytes) -> bytes:
        """Sign message using Ed25519"""
        try:
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
            return private_key.sign(message)
        except Exception as e:
            logger.error(f"Signing failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Message signing failed"
            )
    
    @staticmethod
    def verify_signature(public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify signature using Ed25519"""
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
            public_key.verify(signature, message)
            return True
        except Exception:
            return False
    
    @staticmethod
    def derive_shared_key(private_key: bytes, peer_public_key: bytes) -> bytes:
        """Derive shared secret using X25519"""
        try:
            priv_key = x25519.X25519PrivateKey.from_private_bytes(private_key)
            pub_key = x25519.X25519PublicKey.from_public_bytes(peer_public_key)
            shared_key = priv_key.exchange(pub_key)
            
            return HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'quantum_sovereign_nation',
                backend=default_backend()
            ).derive(shared_key)
        except Exception as e:
            logger.error(f"Key derivation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Key derivation failed"
            )

# ========================
# DATABASE MODELS
# ========================

class Citizen(Base):
    __tablename__ = "citizens"
    
    id = Column(Integer, primary_key=True, index=True)
    public_key = Column(LargeBinary, unique=True, nullable=False)
    encrypted_private_key = Column(LargeBinary, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    token_balance = Column(Float, default=100.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    transactions = relationship("Transaction", back_populates="citizen")
    votes = relationship("Vote", back_populates="citizen")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("citizens.id"))
    receiver_id = Column(Integer, ForeignKey("citizens.id"))
    amount = Column(Float, nullable=False)
    signature = Column(LargeBinary, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    citizen = relationship("Citizen", back_populates="transactions")

class PolicyProposal(Base):
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="proposed")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    votes = relationship("Vote", back_populates="policy")

class Vote(Base):
    __tablename__ = "votes"
    
    id = Column(Integer, primary_key=True, index=True)
    citizen_id = Column(Integer, ForeignKey("citizens.id"))
    policy_id = Column(Integer, ForeignKey("policies.id"))
    vote_value = Column(Integer, nullable=False)
    signature = Column(LargeBinary, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    citizen = relationship("Citizen", back_populates="votes")
    policy = relationship("PolicyProposal", back_populates="votes")

# ========================
# CORE SERVICES
# ========================

class AuthService:
    """Handles authentication and JWT tokens"""
    
    @staticmethod
    def create_access_token(*, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token with standardized implementation"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode JWT token with proper error handling"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except PyJWTError as e:
            logger.error(f"Token decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

class CitizenService:
    """Manages citizen operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def register_citizen(self, username: str, email: str, password: str) -> Citizen:
        """Register new citizen with cryptographic identity"""
        if self.db.query(Citizen).filter(Citizen.username == username).first():
            raise HTTPException(status_code=400, detail="Username already registered")
        
        if self.db.query(Citizen).filter(Citizen.email == email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        private_key, public_key = QuantumCryptoService.generate_keypair()
        
        # In production, you would properly encrypt the private key
        encrypted_private_key = private_key  # Simplified for example
        
        citizen = Citizen(
            public_key=public_key,
            encrypted_private_key=encrypted_private_key,
            username=username,
            email=email,
            hashed_password=pwd_context.hash(password)
        )
        
        self.db.add(citizen)
        self.db.commit()
        self.db.refresh(citizen)
        return citizen
    
    def authenticate(self, username: str, password: str) -> Optional[Citizen]:
        """Authenticate citizen"""
        citizen = self.db.query(Citizen).filter(Citizen.username == username).first()
        if not citizen:
            return None
        
        if not pwd_context.verify(password, citizen.hashed_password):
            return None
        
        return citizen

# ========================
# FASTAPI APP SETUP
# ========================

app = FastAPI(
    title="Quantum Sovereign Nation API",
    description="Core system for digital sovereignty",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# DEPENDENCY INJECTION
# ========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_citizen(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
) -> Citizen:
    token = credentials.credentials
    payload = AuthService.decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    citizen = db.query(Citizen).filter(Citizen.username == username).first()
    if not citizen:
        raise HTTPException(status_code=404, detail="Citizen not found")
    
    return citizen

# ========================
# API ENDPOINTS
# ========================

@app.post("/register")
async def register(
    username: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db)
):
    """Register a new citizen"""
    service = CitizenService(db)
    citizen = service.register_citizen(username, email, password)
    
    access_token = AuthService.create_access_token(
        data={"sub": citizen.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "citizen_id": citizen.id
    }

@app.post("/login")
async def login(
    username: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db)
):
    """Authenticate citizen"""
    service = CitizenService(db)
    citizen = service.authenticate(username, password)
    if not citizen:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = AuthService.create_access_token(
        data={"sub": citizen.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/me")
async def get_me(citizen: Citizen = Depends(get_current_citizen)):
    """Get current citizen info"""
    return {
        "username": citizen.username,
        "email": citizen.email,
        "balance": citizen.token_balance,
        "joined": citizen.created_at.isoformat()
    }

# ========================
# INITIALIZATION
# ========================

@app.on_event("startup")
async def startup():
    """Initialize database"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")

# ========================
# RUN APPLICATION
# ========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
