import React, { useState, useEffect, useRef } from 'react';
import Globe from 'react-globe.gl';
import { Loader2, Thermometer, CloudRain, Wind, Droplets, Gauge, Globe as GlobeIcon } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CustomInteractiveGlobe = ({ onLocationSelect, selectedDate }) => {
  const globeRef = useRef();
  const [countriesData, setCountriesData] = useState([]);
  const [climateData, setClimateData] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [hoverCountry, setHoverCountry] = useState(null);
  const [weatherDetails, setWeatherDetails] = useState(null);

  // NASA MERRA-2 based climate colors (border lines only)
  const climateColors = {
    'Very Hot': '#ef4444',        // 🔴 Red (T2M > 35°C)
    'Very Cold': '#06b6d4',       // 🔵 Cyan (T2M < 5°C)
    'Very Wet': '#3b82f6',        // 🔵 Blue (Precipitation > 20mm/day)
    'Very Windy': '#eab308',      // 🟡 Yellow (Wind > 10 m/s)
    'Very Uncomfortable': '#8b5cf6', // 🟣 Purple (RH > 70% + T > 30°C)
    'Normal': '#10b981',          // 🟢 Green (mild conditions)
    'No Data': '#6b7280'          // Gray
  };

  // Get temperature-based classification (NASA MERRA-2 logic)
  const analyzeClimateCategory = (data) => {
    if (!data || !data.temperature) return 'No Data';
    
    const temp = parseFloat(data.temperature.value);
    const precip = parseFloat(data.precipitation.value);
    const windSpeed = parseFloat(data.wind_speed.value);
    const humidity = parseFloat(data.humidity.value);
    
    // Priority-based NASA MERRA-2 classification
    if (temp > 35) return 'Very Hot';        // 🔴 Red
    if (temp < 5) return 'Very Cold';        // 🔵 Cyan
    if (precip > 20) return 'Very Wet';      // 🔵 Blue  
    if (windSpeed > 10) return 'Very Windy'; // 🟡 Yellow
    if (humidity > 70 && temp > 30) return 'Very Uncomfortable'; // 🟣 Purple
    
    return 'Normal';                         // 🟢 Green
  };

  // Get temperature category for display
  const getTemperatureCategory = (data) => {
    if (!data || !data.temperature) return 'No Data';
    
    const temp = parseFloat(data.temperature.value);
    
    if (temp > 35) return 'Very Hot';
    if (temp < 0) return 'Very Cold';
    if (temp >= 30) return 'Hot';
    if (temp >= 25) return 'Warm';
    if (temp >= 10) return 'Normal';
    if (temp >= 0) return 'Cold';
    
    return 'Very Cold';
  };

  // Major countries to preload
  const majorCountries = ['USA', 'CHN', 'IND', 'BRA', 'RUS', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'CAN', 'AUS'];

  // Load GeoJSON data for countries
  useEffect(() => {
    const loadCountriesData = async () => {
      try {
        // Using the same GeoJSON source for consistency
        const response = await fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson');
        const data = await response.json();
        
        // Add climate properties to each country feature with NASA MERRA-2 classification
        const countriesWithClimate = data.features.map(feature => ({
          ...feature,
          properties: {
            ...feature.properties,
            climateCategory: 'Normal',
            nasaCategory: 'Normal', // NASA MERRA-2 based category
            climateData: null,
            loading: false
          }
        }));
        
        setCountriesData(countriesWithClimate);
        
        // Preload climate data for major countries
        await loadClimateDataForMajorCountries(countriesWithClimate);
        
      } catch (error) {
        console.error('Error loading countries data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadCountriesData();
  }, [selectedDate]);

  // Load climate data for major countries
  const loadClimateDataForMajorCountries = async (countries) => {
    const majorCountryFeatures = countries.filter(country => 
      majorCountries.includes(country.properties.ISO_A3)
    );

    const promises = majorCountryFeatures.map(async (country) => {
      try {
        const climateInfo = await fetchClimateData(country);
        return { countryCode: country.properties.ISO_A3, climateInfo };
      } catch (error) {
        console.error(`Error loading climate data for ${country.properties.NAME}:`, error);
        return { countryCode: country.properties.ISO_A3, climateInfo: null };
      }
    });

    const results = await Promise.allSettled(promises);
    const newClimateData = {};
    
    results.forEach((result) => {
      if (result.status === 'fulfilled' && result.value.climateInfo) {
        newClimateData[result.value.countryCode] = result.value.climateInfo;
      }
    });

    setClimateData(newClimateData);
    
    // Update countries with NASA MERRA-2 classifications
    setCountriesData(prevCountries => 
      prevCountries.map(country => ({
        ...country,
        properties: {
          ...country.properties,
          climateCategory: newClimateData[country.properties.ISO_A3]?.category || 'Normal',
          nasaCategory: newClimateData[country.properties.ISO_A3] 
            ? analyzeClimateCategory(newClimateData[country.properties.ISO_A3].data)
            : 'Normal',
          climateData: newClimateData[country.properties.ISO_A3] || null
        }
      }))
    );
  };

  // Calculate polygon centroid
  const calculateCentroid = (geometry) => {
    if (geometry.type === 'Polygon') {
      const coordinates = geometry.coordinates[0];
      let lat = 0, lon = 0;
      
      coordinates.forEach(coord => {
        lon += coord[0];
        lat += coord[1];
      });
      
      return {
        lat: lat / coordinates.length,
        lon: lon / coordinates.length
      };
    } else if (geometry.type === 'MultiPolygon') {
      const largestPolygon = geometry.coordinates.reduce((largest, current) => 
        current[0].length > largest[0].length ? current : largest
      );
      
      const coordinates = largestPolygon[0];
      let lat = 0, lon = 0;
      
      coordinates.forEach(coord => {
        lon += coord[0];
        lat += coord[1];
      });
      
      return {
        lat: lat / coordinates.length,
        lon: lon / coordinates.length
      };
    }
    
    return { lat: 0, lon: 0 };
  };

  // Fetch climate data from backend
  const fetchClimateData = async (country) => {
    try {
      const centroid = calculateCentroid(country.geometry);
      const formattedDate = selectedDate.replace(/-/g, '');
      
      const response = await fetch(`${API}/climate-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          location: `${centroid.lat},${centroid.lon}`,
          date: formattedDate,
          time: '12:00'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch climate data');
      }

      const data = await response.json();
      const category = getTemperatureCategory(data);
      
      return {
        category,
        data,
        centroid,
        country: country.properties.name || country.properties.NAME
      };
      
    } catch (error) {
      console.error('Error fetching climate data:', error);
      return {
        category: 'Normal',
        data: null,
        centroid: null,
        country: country.properties.NAME
      };
    }
  };

  // Climate classification logic
  const classifyClimate = (data) => {
    if (!data || !data.temperature) return 'Normal';
    
    const temp = parseFloat(data.temperature.value);
    const precip = parseFloat(data.precipitation.value);
    const windSpeed = parseFloat(data.wind_speed.value);
    const humidity = parseFloat(data.humidity.value);
    
    if (temp > 35) return 'Very Hot';
    if (temp < 0) return 'Very Cold';
    if (precip > 20) return 'Very Wet';
    if (windSpeed > 10) return 'Very Windy';
    if (humidity > 70 && temp > 30) return 'Very Uncomfortable';
    
    return 'Normal';
  };

  // Handle country click - Enhanced with live date/time and comprehensive dashboard updates
  const handleCountryClick = async (polygon, event, coordinates) => {
    if (!polygon || !polygon.properties) {
      console.error('No polygon data received in click handler');
      return;
    }
    
    // Try different possible name properties
    const countryName = polygon.properties.name || 
                       polygon.properties.NAME || 
                       polygon.properties.NAME_EN || 
                       polygon.properties.ADMIN || 
                       'Unknown Country';
    
    console.log('Country clicked:', countryName);
    setSelectedCountry(polygon);
    
    try {
      // Fetch fresh climate data for clicked country with current date/time
      const centroid = calculateCentroid(polygon.geometry);
      const currentDate = new Date().toISOString().split('T')[0];
      // Format time without seconds (HH:MM format)
      const currentTime = new Date().toTimeString().slice(0, 5);
      
      const response = await fetch(`${API}/climate-data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: `${centroid.lat},${centroid.lon}`,
          date: currentDate.replace(/-/g, ''),
          time: '12:00'
        })
      });

      if (!response.ok) throw new Error('Failed to fetch climate data');

      const data = await response.json();
      const tempCategory = getTemperatureCategory(data);
      
      // Show weather details modal
      setWeatherDetails({
        category: tempCategory,
        data,
        centroid,
        country: countryName,
        clickDate: currentDate,
        clickTime: currentTime
      });
      
      // Update dashboard with live data - Enhanced for all dashboard components
      if (onLocationSelect && countryName !== 'Unknown Country') {
        onLocationSelect({
          location: countryName,
          date: currentDate,
          time: currentTime,
          climateData: data
        });
      }
      
    } catch (error) {
      console.error('Error fetching live climate data:', error);
      
      // Fallback to cached data if available
      const isoCode = polygon.properties.ISO_A3 || polygon.properties.iso_a3 || polygon.properties.ISO3 || polygon.id;
      if (climateData[isoCode]) {
        setWeatherDetails({
          ...climateData[isoCode],
          clickDate: new Date().toISOString().split('T')[0],
          clickTime: new Date().toTimeString().slice(0, 5)  // Remove seconds
        });
        
        if (onLocationSelect) {
          onLocationSelect({
            location: countryName,
            date: new Date().toISOString().split('T')[0],
            time: new Date().toTimeString().slice(0, 5),  // Remove seconds
            climateData: climateData[isoCode].data
          });
        }
      }
    }
  };

  // Handle country hover - Correct signature for react-globe.gl: (polygon, prevPolygon)
  const handleCountryHover = (polygon, prevPolygon) => {
    setHoverCountry(polygon);
  };

  // Get polygon color based on NASA MERRA-2 category (priority system)
  const getPolygonColor = (country) => {
    const nasaCategory = country.properties.nasaCategory || country.properties.climateCategory;
    return climateColors[nasaCategory] || climateColors['Normal'];
  };

  // Get polygon highlight based on hover state
  const getPolygonAltitude = (country) => {
    return hoverCountry === country ? 0.02 : 0.01; // Raise on hover
  };

  // Get activity recommendation based on weather
  const getActivityRecommendation = (category, data) => {
    if (!data) return "Weather data unavailable for recommendations.";
    
    switch (category) {
      case 'Very Hot':
        return "Stay indoors during peak hours. Consider early morning or evening outdoor activities with plenty of water.";
      case 'Very Cold':
        return "Dress warmly in layers. Great for winter sports if properly equipped. Limit outdoor exposure time.";
      case 'Very Wet':
        return "Perfect weather for indoor activities. If going out, bring waterproof gear and avoid flood-prone areas.";
      case 'Very Windy':
        return "Good for wind sports like sailing or kite flying. Secure loose objects and be cautious with outdoor activities.";
      case 'Very Uncomfortable':
        return "High heat and humidity make outdoor activities challenging. Stay hydrated and seek air-conditioned spaces.";
      case 'Normal':
        return "Great weather for most outdoor activities! Perfect for hiking, sightseeing, or sports.";
      default:
        return "Check local weather conditions before planning outdoor activities.";
    }
  };

  if (loading) {
    return (
      <div className="h-full w-full bg-black flex items-center justify-center">
        <div className="text-center text-slate-400">
          <Loader2 className="h-8 w-8 mx-auto mb-4 animate-spin" />
          <p className="text-sm">Loading Interactive Earth...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      {/* Interactive Earth Title */}
      <div className="absolute top-2 left-2 z-10">
        <div className="flex items-center gap-2 text-slate-200 text-lg">
          <GlobeIcon className="h-5 w-5 text-cyan-400" />
          Interactive Earth
        </div>
      </div>

      {/* Centered Wireframe Globe with border lines only */}
      <div className="h-full w-full flex items-center justify-center">
        <Globe
          ref={globeRef}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          backgroundColor="rgba(0,0,0,0)"
          
          polygonsData={countriesData}
          polygonCapColor={() => 'rgba(0,0,0,0)'} // Transparent fill
          polygonSideColor={() => 'rgba(0,0,0,0)'} // Transparent fill
          polygonStrokeColor={(country) => {
            const baseColor = getPolygonColor(country);
            return hoverCountry === country ? '#00ffff' : baseColor;
          }}
          polygonAltitude={0.001} // Very thin for border-only effect
          
          onPolygonClick={handleCountryClick}
          onPolygonHover={handleCountryHover}
          
          width={380}  // Centered in card
          height={300} // Centered in card
          
          atmosphereColor="#00aaff" // Holographic blue atmosphere
          atmosphereAltitude={0.08}
          
          enablePointerInteraction={true}
          
          // Enhanced controls for futuristic feel
          controls={{
            enableZoom: true,
            enablePan: false,
            enableRotate: true,
            autoRotate: true,
            autoRotateSpeed: 0.6
          }}
        />
      </div>

      {/* Floating Hover Tooltip (Country Name Only) */}
      {hoverCountry && (
        <div className="absolute top-16 left-4 bg-slate-900/95 backdrop-blur-sm rounded-lg p-2 border border-cyan-400/50 shadow-lg shadow-cyan-400/20">
          <h3 className="text-xs font-medium text-cyan-100">
            {hoverCountry.properties.name || hoverCountry.properties.NAME}
          </h3>
        </div>
      )}

    </div>
  );
};

export default CustomInteractiveGlobe;