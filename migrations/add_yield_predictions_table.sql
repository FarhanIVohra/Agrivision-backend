-- Migration: Add yield_predictions table
-- Execute this in your Supabase SQL Editor to add the missing yield_predictions table

-- Create yield_predictions table for storing ML model predictions
CREATE TABLE IF NOT EXISTS yield_predictions (
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
CREATE INDEX IF NOT EXISTS idx_yield_predictions_user_id ON yield_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_yield_predictions_created_at ON yield_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_yield_predictions_crop_state ON yield_predictions(crop, state);

-- Enable Row Level Security
ALTER TABLE yield_predictions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for yield_predictions
CREATE POLICY IF NOT EXISTS "Users can view own yield predictions" ON yield_predictions
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY IF NOT EXISTS "Users can insert own yield predictions" ON yield_predictions
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY IF NOT EXISTS "Users can update own yield predictions" ON yield_predictions
    FOR UPDATE USING (auth.uid()::text = user_id::text);

-- Grant permissions
GRANT ALL ON yield_predictions TO authenticated;

-- Verify table creation
SELECT 'yield_predictions table created successfully' as status;