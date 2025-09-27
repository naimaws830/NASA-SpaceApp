# Climate Indices Dashboard

A modern, real-time climate analysis dashboard with NASA data integration, ARIMA forecasting, and interactive 3D globe visualization.

## 🚀 Features

- **Real NASA MERRA-2 Data**: Authentic climate data from NASA Earthdata
- **ARIMA Climate Forecasting**: Scientific forecasts for recent dates
- **Interactive 3D Globe**: Click countries for instant climate analysis
- **AI-Powered Summaries**: Natural language climate insights
- **Consistent & Deterministic**: Same inputs always produce same results

## 📁 Project Structure

```
climate-dashboard/
├── backend/
│   ├── server.py          # FastAPI server with NASA integration & forecasting
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main dashboard component
│   │   ├── App.css        # Dashboard styling
│   │   ├── index.js       # React entry point
│   │   ├── index.css      # Global styles
│   │   ├── components/
│   │   │   ├── CustomInteractiveGlobe.jsx  # 3D globe component
│   │   │   └── ui/        # Essential UI components (5 files)
│   │   └── lib/
│   │       └── utils.js   # Utility functions
│   ├── public/            # Static assets
│   ├── package.json       # Node.js dependencies
│   ├── craco.config.js    # Build configuration
│   ├── tailwind.config.js # Tailwind CSS config
│   ├── postcss.config.js  # PostCSS config
│   └── .env              # Frontend environment
└── README.md
```

## 🛠️ Technology Stack

**Backend:**
- FastAPI (Python web framework)
- Motor + MongoDB (Database)
- NASA Earthdata MERRA-2 API (Real climate data)
- OpenRouter API (AI summaries)
- ARIMA/SARIMA (Climate forecasting)
- Scipy (Scientific computing)

**Frontend:**
- React 18 (UI framework)
- Tailwind CSS (Styling)
- React-Globe.gl (3D globe visualization)
- Chart.js (Data visualization)
- Lucide React (Icons)

## 🔧 Installation & Setup

See the complete setup guide for local development.

## 🌍 Data Sources

- **Historical Data (>3 months)**: Real NASA MERRA-2 reanalysis data
- **Recent Data (<3 months)**: ARIMA forecasts based on 30 years NASA data  
- **Fallback**: Deterministic simulated data with geographic accuracy

## 📊 Climate Parameters

- Temperature (°C)
- Precipitation (mm/day)  
- Humidity (g/kg)
- Wind Speed (m/s)
- Atmospheric Pressure (hPa)
- Heat Index (°C)

## 🎯 Key Capabilities

1. **Real-time Analysis**: Click any country on the 3D globe for instant climate data
2. **Scientific Forecasting**: ARIMA-based predictions with confidence intervals
3. **Extreme Event Probability**: Statistical analysis of climate conditions
4. **AI Insights**: Natural language summaries of climate conditions
5. **Interactive Visualization**: Professional 3D globe with climate-coded countries

## 📈 System Architecture

```
User Input → 3D Globe Click → Location Detection → Data Source Selection
                                                      ↓
NASA API (>3mo) ← Climate Analysis → ARIMA Forecast (<3mo)
                                                      ↓  
Database Storage ← Climate Dashboard ← AI Summary ← Data Processing
```

## 🌟 Unique Features

- **Deterministic Results**: Same location/date/time always produces identical analysis
- **Geographic Intelligence**: Location-aware climate modeling for accuracy
- **Seamless Data Integration**: Automatic switching between real data and forecasts
- **Professional UI**: Modern, responsive design with scientific visualization
- **No External Dependencies**: Runs completely independently

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

**Built with real NASA climate data and scientific forecasting methods for accurate, reliable climate analysis.**
