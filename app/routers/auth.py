"""
Authentication Router for AgriSmart
Handles user registration, login, and authentication with Supabase integration
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
import jwt
from supabase import create_client, Client
from dotenv import load_dotenv

# Import location service
from ..services.location_service import location_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    logger.error("Supabase configuration missing. Please check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env file")
    raise ValueError("Supabase configuration is required")

supabase: Client = create_client(supabase_url, supabase_key)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer()

# Pydantic Models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None
    lat: float
    lon: float
    region: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @validator('lat')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('lon')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    phone: Optional[str]
    lat: float
    lon: float
    region: str
    created_at: datetime
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class RegisterUserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr

class RegisterResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class LocationUpdate(BaseModel):
    lat: float
    lon: float

# Utility Functions
def parse_iso_datetime(value: Optional[str]) -> datetime:
    """Robust ISO datetime parser that handles 'Z' suffix and None.
    Falls back to UTC now if parsing fails or value is missing.
    """
    try:
        if not value:
            return datetime.utcnow()
        # Replace trailing 'Z' with '+00:00' for Python fromisoformat compatibility
        normalized = value.replace('Z', '+00:00') if isinstance(value, str) else value
        return datetime.fromisoformat(normalized)
    except Exception:
        return datetime.utcnow()

def hash_password(password: str) -> str:
    """Hash password using passlib bcrypt"""
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash using passlib"""
    try:
        return pwd_context.verify(password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {str(e)}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from Supabase
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return response.data[0]
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Authentication Endpoints
@router.post("/register", response_model=RegisterResponse)
async def register_user(user_data: UserRegistration):
    """
    Registers a new user, hashes their password, stores them in the database,
    and returns a JWT token for immediate authentication.

    - **name**: User's full name.
    - **email**: User's email address (must be unique).
    - **password**: User's password (will be hashed).
    """
    try:
        # Check if user already exists
        existing_user = supabase.table("users").select("email").eq("email", user_data.email).execute()

        if existing_user.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Get location details using reverse geocoding
        try:
            location_data = await location_service.reverse_geocode(user_data.lat, user_data.lon)
            region = location_data.get("region", "Unknown Region")
        except Exception as e:
            logger.warning(f"Failed to get location details: {str(e)}")
            region = user_data.region or "Unknown Region"

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Create user in Supabase
        user_record = {
            "email": user_data.email,
            "password_hash": hashed_password,
            "name": user_data.name,
            "phone": user_data.phone,
            "lat": user_data.lat,
            "lon": user_data.lon,
            "region": region,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        }

        response = supabase.table("users").insert(user_record).execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user account."
            )

        created_user = response.data[0]

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(created_user['id'])}, expires_delta=access_token_expires
        )

        logger.info(f"User registered successfully: {user_data.email}")

        # Prepare user response
        user_response = UserResponse(
            id=created_user['id'],
            email=created_user['email'],
            full_name=created_user['name'],
            phone=created_user['phone'],
            lat=created_user['lat'],
            lon=created_user['lon'],
            region=created_user['region'],
            created_at=parse_iso_datetime(created_user.get("created_at")),
            is_active=created_user['is_active']
        )

        return RegisterResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(user_credentials: UserLogin):
    """
    Authenticate user and return JWT token
    """
    try:
        # Get user from database
        response = supabase.table("users").select("*").eq("email", user_credentials.email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        user = response.data[0]
        
        # Verify password
        if not verify_password(user_credentials.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"]}, 
            expires_delta=access_token_expires
        )

        # Note: Skipping last_login update as column doesn't exist in schema
        # supabase.table("users").update({
        #     "last_login": datetime.utcnow().isoformat()
        # }).eq("id", user["id"]).execute()
        
        # Prepare user response
        user_response = UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["name"],  # Map name to full_name for response
            phone=user["phone"],
            lat=user["lat"],
            lon=user["lon"],
            region=user["region"],
            created_at=parse_iso_datetime(user.get("created_at")),
            is_active=user["is_active"]
        )
        
        logger.info(f"User logged in successfully: {user_credentials.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["name"],  # Map name to full_name for response
        phone=current_user["phone"],
        lat=current_user["lat"],
        lon=current_user["lon"],
        region=current_user["region"],
        created_at=parse_iso_datetime(current_user.get("created_at")),
        is_active=current_user["is_active"]
    )

@router.put("/location")
async def update_user_location(
    location_data: LocationUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user's location coordinates
    """
    try:
        # Get location details using reverse geocoding
        try:
            location_info = await location_service.reverse_geocode(location_data.lat, location_data.lon)
            region = location_info.get("region", "Unknown Region")
        except Exception as e:
            logger.warning(f"Failed to get location details: {str(e)}")
            region = current_user.get("region", "Unknown Region")
        
        # Update user location in database
        response = supabase.table("users").update({
            "lat": location_data.lat,
            "lon": location_data.lon,
            "region": region,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", current_user["id"]).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update location"
            )
        
        logger.info(f"Location updated for user: {current_user['email']}")
        
        return {
            "message": "Location updated successfully",
            "lat": location_data.lat,
            "lon": location_data.lon,
            "region": region
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during location update"
        )

@router.post("/logout")
async def logout_user(current_user: dict = Depends(get_current_user)):
    """
    Logout user (client should remove token)
    """
    logger.info(f"User logged out: {current_user['email']}")
    return {"message": "Logged out successfully"}

@router.delete("/account")
async def delete_account(current_user: dict = Depends(get_current_user)):
    """
    Deactivate user account
    """
    try:
        # Deactivate account instead of deleting
        response = supabase.table("users").update({
            "is_active": False,
            "deactivated_at": datetime.utcnow().isoformat()
        }).eq("id", current_user["id"]).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate account"
            )
        
        logger.info(f"Account deactivated: {current_user['email']}")
        
        return {"message": "Account deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during account deletion"
        )