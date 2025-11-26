# AgriSmart - AI-Powered Crop Yield Prediction Platform

A modern React-based agricultural platform utilizing AI and real-time data for crop yield prediction, soil health monitoring, and smart farming recommendations.

## 🚀 Features

- **React 18** - React version with improved rendering and concurrent features
- **Vite** - Lightning-fast build tool and development server
- **Redux Toolkit** - State management with simplified Redux setup
- **TailwindCSS** - Utility-first CSS framework with extensive customization
- **React Router v6** - Declarative routing for React applications
- **Data Visualization** - Integrated D3.js and Recharts for powerful data visualization
- **Form Management** - React Hook Form for efficient form handling
- **Animation** - Framer Motion for smooth UI animations
- **Testing** - Jest and React Testing Library setup
- **🌱 Soil Health Monitoring** - Real-time soil analysis with health scoring and recommendations
- **📊 Advanced Analytics** - NPK levels tracking with 7-day trend analysis
- **🌤️ Weather Integration** - OpenWeatherMap API integration for environmental data
- **💡 Smart Recommendations** - AI-powered farming recommendations based on soil conditions

## 🌱 Soil Health Monitoring System

### Overview
The Soil Health Monitoring system provides comprehensive analysis of soil conditions with real-time data integration, health scoring, and actionable recommendations for farmers.

### Key Features

#### 📊 Real-time Soil Metrics
- **pH Level Monitoring** - Optimal range detection (6.0-7.5)
- **Moisture Content** - Percentage-based moisture tracking
- **Temperature & Humidity** - Environmental condition monitoring
- **Organic Matter** - Soil composition analysis
- **NPK Levels** - Nitrogen, Phosphorus, and Potassium tracking

#### 🎯 Health Scoring Algorithm
- **Overall Health Score** - Comprehensive soil health rating (0-100)
- **Individual Metric Scores** - Detailed scoring for each soil parameter
- **Color-coded Status** - Visual indicators (Excellent/Good/Needs Attention)
- **Trend Analysis** - 7-day historical data visualization

#### 💡 Smart Recommendations
- **Priority-based Suggestions** - High/Medium/Low priority recommendations
- **Cost Estimates** - Expected investment for improvements
- **Expected Outcomes** - Predicted yield improvements
- **Weather-aware Advice** - Recommendations adjusted for current conditions

### API Integration

#### Backend Endpoint: `/api/dashboard/soil`
```javascript
// Example API Response
{
  "metrics": {
    "ph_level": 6.8,
    "moisture": 65,
    "temperature": 22,
    "humidity": 70,
    "organic_matter": 3.2,
    "nitrogen": 45,
    "phosphorus": 25,
    "potassium": 35
  },
  "health_scores": {
    "overall": 75,
    "nitrogen": 65,
    "phosphorus": 70,
    "potassium": 60,
    "ph": 85,
    "moisture": 80,
    "organic_matter": 75
  },
  "recommendations": [
    {
      "title": "Increase Nitrogen Levels",
      "description": "Apply nitrogen-rich fertilizer to improve crop yield",
      "priority": "Medium",
      "expected_outcome": "15-20% yield increase",
      "cost": "$50-75 per acre"
    }
  ],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

#### Features:
- **5-minute Caching** - Optimized API calls with intelligent caching
- **OpenWeatherMap Integration** - Real-time weather data for soil analysis
- **Supabase Storage** - Persistent data storage with user authentication
- **Fallback System** - LocalStorage and mock data fallbacks for offline use

### Frontend Components

#### SoilHealthCard Component
Located at: `src/pages/dashboard/components/SoilHealthCard.jsx`

**Features:**
- Real-time data fetching with JWT authentication
- Interactive charts with toggle between NPK levels and 7-day trends
- Responsive design with mobile optimization
- Health status indicators with color coding
- Recommendations display with priority badges
- Manual refresh functionality

**Chart Visualization:**
- **NPK Bar Chart** - Current nitrogen, phosphorus, and potassium levels
- **7-Day Trend Lines** - Historical data visualization for pH, moisture, and nutrients
- **Responsive Design** - Optimized for desktop and mobile viewing

### Database Schema

#### Supabase `soil_health` Table
```sql
CREATE TABLE soil_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    data JSONB NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_soil_health_user_id ON soil_health(user_id);
CREATE INDEX idx_soil_health_fetched_at ON soil_health(fetched_at);

-- Row Level Security
ALTER TABLE soil_health ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own soil health data" ON soil_health
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own soil health data" ON soil_health
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own soil health data" ON soil_health
    FOR UPDATE USING (auth.uid() = user_id);
```

### Usage Instructions

#### 1. Environment Setup
Ensure your `.env` file includes:
```env
VITE_OPENWEATHER_API_KEY=your_openweather_api_key
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

#### 2. Database Setup
1. Copy the schema from `supabase_schema.sql`
2. Execute in your Supabase SQL editor
3. Verify tables and policies are created

#### 3. Component Integration
```jsx
import SoilHealthCard from './components/SoilHealthCard';

function Dashboard() {
  return (
    <div className="dashboard">
      <SoilHealthCard />
    </div>
  );
}
```

#### 4. Styling
The component includes comprehensive CSS styling in `SoilHealthCard.css` with:
- Responsive design breakpoints
- Dark mode support
- Animation effects
- Color-coded health indicators

### Troubleshooting

#### Common Issues:
1. **API Authentication Errors** - Ensure JWT token is properly stored in localStorage
2. **Chart Rendering Issues** - Verify Recharts is properly installed
3. **Styling Problems** - Check CSS import and Tailwind configuration
4. **Data Loading Failures** - Component includes fallback to localStorage and mock data

#### Fallback Behavior:
1. **Primary**: API endpoint with real-time data
2. **Secondary**: LocalStorage cached data
3. **Tertiary**: Mock data for demonstration

### Performance Optimizations
- **Caching Strategy** - 5-minute cache for API calls
- **Lazy Loading** - Charts load only when needed
- **Responsive Images** - Optimized for different screen sizes
- **Error Boundaries** - Graceful error handling

## 📋 Prerequisites

- Node.js (v14.x or higher)
- npm or yarn

## 🛠️ Installation

1. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```
   
2. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

## 📁 Project Structure

```
react_app/
├── public/             # Static assets
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Page components
│   ├── styles/         # Global styles and Tailwind configuration
│   ├── App.jsx         # Main application component
│   ├── Routes.jsx      # Application routes
│   └── index.jsx       # Application entry point
├── .env                # Environment variables
├── index.html          # HTML template
├── package.json        # Project dependencies and scripts
├── tailwind.config.js  # Tailwind CSS configuration
└── vite.config.js      # Vite configuration
```

## 🧩 Adding Routes

To add new routes to the application, update the `Routes.jsx` file:

```jsx
import { useRoutes } from "react-router-dom";
import HomePage from "pages/HomePage";
import AboutPage from "pages/AboutPage";

const ProjectRoutes = () => {
  let element = useRoutes([
    { path: "/", element: <HomePage /> },
    { path: "/about", element: <AboutPage /> },
    // Add more routes as needed
  ]);

  return element;
};
```

## 🎨 Styling

This project uses Tailwind CSS for styling. The configuration includes:

- Forms plugin for form styling
- Typography plugin for text styling
- Aspect ratio plugin for responsive elements
- Container queries for component-specific responsive design
- Fluid typography for responsive text
- Animation utilities

## 📱 Responsive Design

The app is built with responsive design using Tailwind CSS breakpoints.


## 📦 Deployment

Build the application for production:

```bash
npm run build
```

## 🙏 Acknowledgments

- Built with [Rocket.new](https://rocket.new)
- Powered by React and Vite
- Styled with Tailwind CSS

Built with ❤️ on Rocket.new
