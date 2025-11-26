# Supabase Setup Guide for AgriSmart

This guide will help you set up Supabase for the AgriSmart application with proper authentication and database configuration.

## Prerequisites

1. A Supabase account (sign up at [supabase.com](https://supabase.com))
2. A new Supabase project created

## Step 1: Create Supabase Project

1. Go to [Supabase Dashboard](https://app.supabase.com)
2. Click "New Project"
3. Choose your organization
4. Enter project details:
   - Name: `agrismart`
   - Database Password: (choose a strong password)
   - Region: (choose closest to your users)
5. Click "Create new project"

## Step 2: Configure Database Schema

1. In your Supabase dashboard, go to the "SQL Editor"
2. Copy and paste the contents of `supabase_schema.sql` into the editor
3. Click "Run" to execute the schema creation

## Step 3: Configure Authentication

1. Go to "Authentication" > "Settings" in your Supabase dashboard
2. Configure the following settings:

### Site URL Configuration
- Site URL: `http://localhost:3000` (for development)
- Additional redirect URLs: Add your production domain when deploying

### Email Templates (Optional)
- Customize the email templates for user confirmation and password reset

## Step 4: Get API Keys

1. Go to "Settings" > "API" in your Supabase dashboard
2. Copy the following values:
   - Project URL
   - `anon` public key
   - `service_role` secret key (keep this secure!)

## Step 5: Update Environment Variables

Update your `.env` file in the backend (`agrismart/app/.env`) with:

```env
# Supabase Configuration
SUPABASE_URL=your_project_url_here
SUPABASE_KEY=your_anon_key_here
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Existing API keys
OPENWEATHERMAP_API_KEY=4d16be1669a3625caa757c4df10d2f60
GEOAPIFY_API_KEY=your_geoapify_key_here
```

## Step 6: Test the Setup

1. Start your backend server:
   ```bash
   cd agrismart/app
   uvicorn main:app --reload --port 8000
   ```

2. Start your frontend:
   ```bash
   cd agrismart
   npm run dev
   ```

3. Test user registration and login functionality

## Database Tables Created

### `users` table
- Stores user registration information
- Includes location data (lat/lon) for weather services
- Contains farm-specific information (size, experience, crops)

### `weather_logs` table
- Logs all weather API requests and responses
- Helps with debugging and usage analytics
- Stores request parameters and response data as JSONB

## Security Features

- **Row Level Security (RLS)** enabled on all tables
- Users can only access their own data
- Proper authentication policies configured
- Password hashing with bcrypt

## API Endpoints Available

After setup, the following endpoints will be available:

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/dashboard/weather` - Get current weather for user's location
- `GET /api/dashboard/weather/forecast` - Get weather forecast
- `GET /api/dashboard/soil` - Get soil recommendations

## Troubleshooting

### Common Issues

1. **Authentication errors**: Check that your Supabase URL and keys are correct
2. **Database connection issues**: Ensure your IP is whitelisted in Supabase
3. **RLS policy errors**: Make sure you're using the correct authentication headers

### Debugging Tips

1. Check the Supabase logs in the dashboard
2. Use the Supabase SQL editor to test queries
3. Verify environment variables are loaded correctly

## Production Deployment

When deploying to production:

1. Update the Site URL in Supabase authentication settings
2. Use environment variables for all sensitive keys
3. Enable additional security features like email confirmation
4. Set up proper CORS policies
5. Consider enabling database backups

## Support

For issues with this setup:
1. Check the Supabase documentation
2. Review the application logs
3. Test API endpoints with tools like Postman
4. Verify database schema matches the expected structure