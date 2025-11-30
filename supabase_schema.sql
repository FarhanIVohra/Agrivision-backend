-- Supabase Database Schema for AgriSmart Application
-- This file contains the SQL commands to create the necessary tables for the authentication and weather logging system

-- Enable UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table for storing user registration and profile information
CREATE TABLE users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    language VARCHAR(10) DEFAULT 'en',
    region VARCHAR(255),
    farm_size DECIMAL(10,2),
    experience_level VARCHAR(50),
    main_crops TEXT[], -- Array of crop types
    lat DECIMAL(10,8), -- Latitude with high precision
    lon DECIMAL(11,8), -- Longitude with high precision
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Weather logs table for storing weather API requests and responses
CREATE TABLE weather_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    request_type VARCHAR(50) NOT NULL, -- 'current', 'forecast', 'soil'
    location VARCHAR(255),
    lat DECIMAL(10,8),
    lon DECIMAL(11,8),
    request_data JSONB, -- Store request parameters
    response_data JSONB, -- Store API response
    status_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Soil health table for storing soil health monitoring data
CREATE TABLE soil_health (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    data JSONB NOT NULL, -- Store soil metrics, health scores, and recommendations
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Yield predictions table for storing ML model predictions
CREATE TABLE yield_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    crop VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL,
    season VARCHAR(50) NOT NULL,
    area DECIMAL(10,2) NOT NULL,
    predicted_yield DECIMAL(10,2) NOT NULL,
    input_parameters JSONB, -- Store all input parameters used for prediction
    model_version VARCHAR(50),
    confidence_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_location ON users(lat, lon);
CREATE INDEX idx_weather_logs_user_id ON weather_logs(user_id);
CREATE INDEX idx_weather_logs_created_at ON weather_logs(created_at);
CREATE INDEX idx_weather_logs_request_type ON weather_logs(request_type);
CREATE INDEX idx_soil_health_user_id ON soil_health(user_id);
CREATE INDEX idx_soil_health_fetched_at ON soil_health(fetched_at);
CREATE INDEX idx_yield_predictions_user_id ON yield_predictions(user_id);
CREATE INDEX idx_yield_predictions_created_at ON yield_predictions(created_at);
CREATE INDEX idx_yield_predictions_crop_state ON yield_predictions(crop, state);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at column
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies for data protection
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE weather_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE soil_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE yield_predictions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own data
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid()::text = id::text);

-- Policy: Users can only access their own weather logs
CREATE POLICY "Users can view own weather logs" ON weather_logs
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own weather logs" ON weather_logs
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

-- Policy: Users can only access their own soil health data
CREATE POLICY "Users can view own soil health data" ON soil_health
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own soil health data" ON soil_health
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update own soil health data" ON soil_health
    FOR UPDATE USING (auth.uid()::text = user_id::text);

-- Policy: Users can only access their own yield predictions
CREATE POLICY "Users can view own yield predictions" ON yield_predictions
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own yield predictions" ON yield_predictions
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update own yield predictions" ON yield_predictions
    FOR UPDATE USING (auth.uid()::text = user_id::text);

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON users TO authenticated;
GRANT ALL ON weather_logs TO authenticated;
GRANT ALL ON soil_health TO authenticated;
GRANT ALL ON yield_predictions TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- Sample data for testing (optional)
-- INSERT INTO users (email, password_hash, name, phone, language, region, farm_size, experience_level, main_crops, lat, lon)
-- VALUES (
--     'test@example.com',
--     '$2b$12$example_hash_here',
--     'Test Farmer',
--     '+91 9876543210',
--     'en',
--     'Gujarat, India',
--     5.5,
--     'intermediate',
--     ARRAY['wheat', 'rice', 'cotton'],
--     22.6708,
--     71.5724
-- );