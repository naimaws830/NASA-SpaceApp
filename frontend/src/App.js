import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import CustomInteractiveGlobe from './components/CustomInteractiveGlobe';
import { 
  MapPin, 
  Calendar, 
  Thermometer, 
  CloudRain, 
  Wind, 
  Droplets,
  Flame,
  Sun,
  AlertTriangle,
  TrendingUp,
  Globe,
  FileText,
  Gauge
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [location, setLocation] = useState('New York, NY');
  const [selectedDate, setSelectedDate] = useState('2025-08-01');
  const [selectedTime, setSelectedTime] = useState('12:00');
  const [selectedVariable, setSelectedVariable] = useState('temperature');
  const [selectedDistributionVariable, setSelectedDistributionVariable] = useState('temperature');
  const [climateData, setClimateData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [error, setError] = useState(null);

  // Handle globe location selection with enhanced date/time
  const handleGlobeLocationSelect = (locationData) => {
    // Update location and date/time if provided
    setLocation(locationData.location);
    
    if (locationData.date) {
      setSelectedDate(locationData.date);
    }
    
    if (locationData.time) {
      setSelectedTime(locationData.time);
    }
    
    if (locationData.climateData) {
      setClimateData(locationData.climateData);
    }
    
    // Auto-trigger analysis with new data
    if (locationData.climateData) {
      setLoading(false); // Skip additional loading since we have the data
    }
  };

  const fetchClimateData = async () => {
    setLoading(true);
    setError(null);
    setAiSummary(null);
    try {
      // Convert date format from YYYY-MM-DD to YYYYMMDD
      const formattedDate = selectedDate.replace(/-/g, '');
      
      const response = await axios.post(`${API}/climate-data`, {
        location,
        date: formattedDate,
        time: selectedTime
      });
      setClimateData(response.data);
      
      // Fetch 30-year historical data with the selected date
      const historyResponse = await axios.get(`${API}/climate-history/${encodeURIComponent(location)}?date=${formattedDate}`);
      setHistoryData(historyResponse.data);
      
      // Generate AI summary after getting climate and historical data
      await generateAiSummary(response.data, historyResponse.data);
      
    } catch (err) {
      console.error('Error fetching climate data:', err);
      setError('Failed to fetch climate data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const generateAiSummary = async (climateData, historicalData) => {
    setLoadingSummary(true);
    try {
      const summaryResponse = await axios.post(`${API}/generate-climate-summary`, {
        climate_data: climateData,
        historical_data: historicalData
      });
      setAiSummary(summaryResponse.data);
    } catch (err) {
      console.error('Error generating AI summary:', err);
      // Set a fallback summary if AI generation fails
      setAiSummary({
        summary: `Climate analysis for ${climateData.location} shows ${climateData.primary_condition} conditions with ${climateData.primary_percentage}% probability. Current temperature: ${climateData.temperature.value}°C with ${climateData.temperature.chance}% chance of threshold exceedance.`,
        location: climateData.location,
        fallback: true
      });
    } finally {
      setLoadingSummary(false);
    }
  };

  useEffect(() => {
    fetchClimateData();
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'Normal': return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
      case 'Above Average': return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'Anomalous': return 'bg-red-500/20 text-red-300 border-red-500/30';
      default: return 'bg-slate-500/20 text-slate-300 border-slate-500/30';
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
      },
    },
    elements: {
      point: {
        radius: 0,
      },
      line: {
        borderWidth: 2,
      },
    },
  };

  const generateSparklineData = (values, color) => ({
    labels: Array.from({ length: values.length }, (_, i) => i),
    datasets: [
      {
        data: values,
        borderColor: color,
        backgroundColor: `${color}20`,
        fill: true,
        tension: 0.4,
      },
    ],
  });

  // Variable configuration for different climate parameters
  const variableConfig = {
    temperature: {
      label: 'Temperature',
      unit: '°C',
      color: '#06b6d4',
      threshold: 35,
      thresholdLabel: 'Very Hot',
      icon: '🌡️',
      description: 'exceeded 35°C'
    },
    precipitation: {
      label: 'Precipitation',
      unit: 'mm/day',
      color: '#0ea5e9',
      threshold: 20,
      thresholdLabel: 'Very Wet',
      icon: '🌧️',
      description: 'exceeded 20mm/day'
    },
    humidity: {
      label: 'Humidity',
      unit: '%',
      color: '#3b82f6',
      threshold: 70,
      thresholdLabel: 'Very Humid',
      icon: '💧',
      description: 'exceeded 70%'
    },
    wind_speed: {
      label: 'Wind Speed',
      unit: 'm/s',
      color: '#10b981',
      threshold: 10,
      thresholdLabel: 'Very Windy',
      icon: '💨',
      description: 'exceeded 10m/s'
    },
    pressure: {
      label: 'Pressure',
      unit: 'hPa',
      color: '#f59e0b',
      threshold: 50, // Deviation from 1013.25
      thresholdLabel: 'Pressure Anomaly',
      icon: '🔘',
      description: 'deviated >50hPa from standard'
    },
    heat_index: {
      label: 'Heat Index',
      unit: '°C',
      color: '#ef4444',
      threshold: 32,
      thresholdLabel: 'Very Uncomfortable',
      icon: '🔥',
      description: 'exceeded 32°C'
    }
  };

  const getCurrentVariableConfig = () => variableConfig[selectedVariable];

  const getCurrentDistributionVariableConfig = () => variableConfig[selectedDistributionVariable];

  const getVariableData = (historyData, variable) => {
    if (!historyData?.historical_data) return [];
    return historyData.historical_data.map(d => d[variable]);
  };

  const getThresholdExceedanceCount = (historyData, variable) => {
    if (!historyData?.historical_data) return 0;
    
    const config = variableConfig[variable];
    const values = getVariableData(historyData, variable);
    
    if (variable === 'pressure') {
      // For pressure, count deviations from standard (1013.25 hPa)
      return values.filter(value => Math.abs(value - 1013.25) > config.threshold).length;
    } else {
      // For other variables, count values above threshold
      return values.filter(value => value > config.threshold).length;
    }
  };

  const getVariableProbability = (historyData, variable) => {
    const exceedanceCount = getThresholdExceedanceCount(historyData, variable);
    return Math.round((exceedanceCount / 30) * 100);
  };

  const generateHistoryValues = (baseValue, count = 7) => {
    // Create deterministic values based on baseValue to ensure consistency
    const seed = Math.floor(baseValue * 1000) % 1000; // Use baseValue as seed
    const values = [];
    
    for (let i = 0; i < count; i++) {
      // Simple deterministic pseudo-random function
      const pseudoRandom = ((seed + i * 17) % 127) / 127 - 0.5;
      values.push(baseValue + pseudoRandom * baseValue * 0.3);
    }
    
    return values;
  };

  const generateDistributionLabels = (historicalData, variable) => {
    const values = historicalData.map(d => d[variable]).sort((a, b) => a - b);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = (max - min) / 20; // 20 bins
    
    const labels = [];
    for (let i = 0; i <= 20; i++) {
      labels.push((min + i * step).toFixed(1));
    }
    return labels;
  };

  const generateDistributionData = (historicalData, variable) => {
    const values = historicalData.map(d => d[variable]).sort((a, b) => a - b);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = (max - min) / 20;
    
    // Create histogram bins
    const bins = Array(21).fill(0);
    values.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / step), 20);
      bins[binIndex]++;
    });
    
    // Convert to percentages
    return bins.map(count => (count / values.length) * 100);
  };

  const generateThresholdArea = (historicalData, variable, threshold) => {
    const distributionData = generateDistributionData(historicalData, variable);
    const labels = generateDistributionLabels(historicalData, variable);
    
    // Only show data above threshold
    return distributionData.map((value, index) => {
      const labelValue = parseFloat(labels[index]);
      return labelValue > threshold ? value : 0;
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950/50 to-slate-900">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>

      <div className="relative z-10 container mx-auto p-6 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent mb-4">
            Climate Indices Dashboard
          </h1>
        </div>

        {/* Header Card - Location Input */}
        <Card className="mb-8 bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl">
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
              <div className="md:col-span-1">
                <label className="block text-sm font-medium text-slate-300 mb-2">Location</label>
                <Input
                  placeholder="Enter location (e.g., New York, NY)"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="bg-slate-800/50 border-slate-600 text-slate-200 placeholder-slate-400 focus:border-cyan-400 focus:ring-cyan-400/20"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  <Calendar className="h-4 w-4 inline mr-1" />
                  Date
                </label>
                <Input
                  type="date"
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="bg-slate-800/50 border-slate-600 text-slate-200 focus:border-cyan-400 focus:ring-cyan-400/20"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Time</label>
                <Input
                  type="time"
                  value={selectedTime}
                  onChange={(e) => setSelectedTime(e.target.value)}
                  className="bg-slate-800/50 border-slate-600 text-slate-200 focus:border-cyan-400 focus:ring-cyan-400/20"
                />
              </div>
              <div>
                <Button 
                  onClick={fetchClimateData}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0 shadow-lg hover:shadow-cyan-500/25 transition-all duration-300"
                >
                  {loading ? 'Analyzing...' : 'Analyze Climate'}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="mb-8 bg-red-900/20 backdrop-blur-xl border-red-500/30">
            <CardContent className="p-4">
              <p className="text-red-300 flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                {error}
              </p>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Side - Interactive Globe */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl" style={{ height: '384px' }}>
              <CardContent className="p-0 h-full">
                <CustomInteractiveGlobe 
                  onLocationSelect={handleGlobeLocationSelect}
                  selectedDate={selectedDate}
                />
              </CardContent>
            </Card>
          </div>

          {/* Right Side - Climate Analysis Cards */}
          <div className="lg:col-span-3 flex flex-col" style={{ height: '384px' }}> {/* Match globe card height */}
            {/* Climate Analysis Card - Slightly reduced height */}
            {climateData && (
              <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl flex-1 mb-4" style={{ height: '55%' }}>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-slate-200 text-lg">
                    <MapPin className="h-4 w-4 text-cyan-400" />
                    Climate Analysis Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="pb-3 h-full flex flex-col justify-between">
                  <div className="space-y-2">
                    {/* Location and Time Display - Compact */}
                    <div className="grid grid-cols-2 gap-2 p-2 bg-slate-800/30 rounded border border-slate-700/50">
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Location</p>
                        <p className="text-xs font-semibold text-slate-200">{climateData.location}</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Date & Time</p>
                        <p className="text-xs font-semibold text-slate-200">
                          {selectedDate} at {climateData.time}
                        </p>
                      </div>
                    </div>
                    
                    {/* Primary Climate Analysis - Compact Format with Dynamic Colors */}
                    <div className="text-center py-2">
                      <div 
                        className="text-2xl font-bold mb-1"
                        style={{
                          color: climateData.primary_condition === 'Very Hot' ? '#ef4444' :
                                 climateData.primary_condition === 'Very Cold' ? '#06b6d4' :
                                 climateData.primary_condition === 'Very Wet' ? '#3b82f6' :
                                 climateData.primary_condition === 'Very Windy' ? '#10b981' :
                                 climateData.primary_condition === 'Very Uncomfortable' ? '#f97316' :
                                 '#06b6d4' // Default cyan
                        }}
                      >
                        {climateData.primary_condition}
                      </div>
                      <div className="text-lg font-semibold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-2">
                        {climateData.primary_percentage}% chance
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* AI Summary Card - Adjusted height */}
            <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl" style={{ height: '45%' }}>
              <CardHeader className="pb-0">
                <CardTitle className="flex items-center gap-2 text-slate-200 text-lg mb-0">
                  <FileText className="h-5 w-5 text-cyan-400" />
                  AI Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 pb-3 h-full flex flex-col">
                {loadingSummary ? (
                  <div className="text-center text-slate-400">
                    <div className="animate-pulse">
                      <div className="h-3 bg-slate-700 rounded mb-2"></div>
                      <div className="h-3 bg-slate-700 rounded mb-2"></div>
                      <div className="h-3 bg-slate-700 rounded w-3/4 mx-auto"></div>
                    </div>
                    <p className="mt-3 text-sm">Generating AI insights...</p>
                  </div>
                ) : aiSummary ? (
                  <div className="text-slate-200">
                    <p className="text-sm leading-tight">{aiSummary.summary.replace(/"/g, '')}</p>
                    {aiSummary.fallback && (
                      <p className="text-xs text-slate-500 mt-2">
                        ⚠️ Fallback summary (AI service temporarily unavailable)
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-slate-400">
                    <div className="animate-pulse">
                      <div className="h-3 bg-slate-700 rounded mb-2"></div>
                      <div className="h-3 bg-slate-700 rounded mb-2"></div>
                      <div className="h-3 bg-slate-700 rounded w-3/4 mx-auto"></div>
                    </div>
                    <p className="mt-3 text-sm">AI-powered insights will appear here</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Climate Details Grid - New Section with added gap */}
        {climateData && (
          <Card className="mt-8 mb-6 bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-slate-200 text-lg">
                <Globe className="h-5 w-5 text-cyan-400" />
                Climate Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { 
                    data: climateData.temperature, 
                    icon: Thermometer, 
                    color: '#06b6d4',
                    values: generateHistoryValues(climateData.temperature.value)
                  },
                  { 
                    data: climateData.precipitation, 
                    icon: CloudRain, 
                    color: '#0ea5e9',
                    values: generateHistoryValues(climateData.precipitation.value)
                  },
                  { 
                    data: climateData.humidity, 
                    icon: Droplets, 
                    color: '#3b82f6',
                    values: generateHistoryValues(climateData.humidity.value)
                  },
                  { 
                    data: climateData.wind_speed, 
                    icon: Wind, 
                    color: '#10b981',
                    values: generateHistoryValues(climateData.wind_speed.value)
                  },
                  { 
                    data: climateData.pressure, 
                    icon: Gauge, 
                    color: '#f59e0b',
                    values: generateHistoryValues(climateData.pressure.value)
                  },
                  { 
                    data: climateData.heat_index, 
                    icon: Flame, 
                    color: '#ef4444',
                    values: generateHistoryValues(climateData.heat_index.value)
                  },
                ].map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <Card key={index} className="bg-slate-800/50 backdrop-blur-xl border-slate-600/50 shadow-lg">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Icon className="h-5 w-5" style={{ color: item.color }} />
                            <span className="text-sm font-medium text-slate-300">{item.data.name}</span>
                          </div>
                          <Badge className={getStatusColor(item.data.status)}>
                            {item.data.status}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <div className="text-xl font-bold text-slate-200">
                              {item.data.value} {item.data.unit}
                            </div>
                            <div className="text-sm font-semibold" style={{ color: item.color }}>
                              {item.data.chance}% chance
                            </div>
                          </div>
                          <div className="w-16 h-12">
                            <Line
                              data={generateSparklineData(item.values, item.color)}
                              options={chartOptions}
                            />
                          </div>
                        </div>
                        <p className="text-xs text-slate-400">{item.data.description}</p>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Charts Section - Stacked Layout */}
        {climateData && historyData && (
          <div className="space-y-6">
            {/* Historical Trend Line Chart - Full Width */}
            <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-slate-200 text-lg">
                      <TrendingUp className="h-5 w-5 text-cyan-400" />
                      Historical Trend (30 Years)
                    </CardTitle>
                    <p className="text-sm text-slate-400">
                      {historyData.analysis_period} • Climate Zone: {historyData.climate_zone}
                    </p>
                  </div>
                  <div className="w-48">
                    <select
                      value={selectedVariable}
                      onChange={(e) => setSelectedVariable(e.target.value)}
                      className="w-full bg-slate-800/50 border-slate-600 text-slate-200 rounded-md px-3 py-2 text-sm focus:border-cyan-400 focus:ring-cyan-400/20"
                    >
                      {Object.entries(variableConfig).map(([key, config]) => (
                        <option key={key} value={key}>
                          {config.icon} {config.label} ({config.unit})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Line
                    data={{
                      labels: historyData.historical_data.map(h => h.year),
                      datasets: [
                        {
                          label: `${getCurrentVariableConfig().label} (${getCurrentVariableConfig().unit})`,
                          data: getVariableData(historyData, selectedVariable),
                          borderColor: getCurrentVariableConfig().color,
                          backgroundColor: 'transparent',
                          pointBackgroundColor: getVariableData(historyData, selectedVariable).map(value => {
                            const config = getCurrentVariableConfig();
                            if (selectedVariable === 'pressure') {
                              return Math.abs(value - 1013.25) > config.threshold ? '#ef4444' : config.color;
                            }
                            return value > config.threshold ? '#ef4444' : config.color;
                          }),
                          pointRadius: 4,
                          fill: false,
                          tension: 0.1,
                        },
                        {
                          label: `${getCurrentVariableConfig().thresholdLabel} Threshold`,
                          data: Array(historyData.historical_data.length).fill(
                            selectedVariable === 'pressure' 
                              ? 1013.25 + getCurrentVariableConfig().threshold 
                              : getCurrentVariableConfig().threshold
                          ),
                          borderColor: '#ef4444',
                          backgroundColor: 'transparent',
                          borderDash: [5, 5],
                          pointRadius: 0,
                          fill: false,
                        },
                        // Add second threshold line for pressure (negative deviation)
                        ...(selectedVariable === 'pressure' ? [{
                          label: `Low Pressure Threshold`,
                          data: Array(historyData.historical_data.length).fill(1013.25 - getCurrentVariableConfig().threshold),
                          borderColor: '#ef4444',
                          backgroundColor: 'transparent',
                          borderDash: [5, 5],
                          pointRadius: 0,
                          fill: false,
                        }] : [])
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'top',
                          labels: {
                            color: '#cbd5e1',
                            usePointStyle: true,
                          },
                        },
                        tooltip: {
                          callbacks: {
                            afterBody: function(context) {
                              const dataIndex = context[0].dataIndex;
                              const value = getVariableData(historyData, selectedVariable)[dataIndex];
                              const config = getCurrentVariableConfig();
                              
                              if (selectedVariable === 'pressure') {
                                const deviation = Math.abs(value - 1013.25);
                                if (deviation > config.threshold) {
                                  return `Pressure deviation: ${deviation.toFixed(1)}hPa from standard`;
                                }
                              } else if (value > config.threshold) {
                                return `Exceeds threshold by ${(value - config.threshold).toFixed(1)}${config.unit}`;
                              }
                              return '';
                            }
                          }
                        }
                      },
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: 'Years',
                            color: '#cbd5e1'
                          },
                          ticks: { color: '#64748b' },
                          grid: { color: '#334155' },
                        },
                        y: {
                          title: {
                            display: true,
                            text: `${getCurrentVariableConfig().label} (${getCurrentVariableConfig().unit})`,
                            color: '#cbd5e1'
                          },
                          ticks: { color: '#64748b' },
                          grid: { color: '#334155' },
                        },
                      },
                    }}
                  />
                </div>
                <div className="mt-4 text-center">
                  <p className="text-sm text-slate-300">
                    <span className="font-semibold">
                      {getCurrentVariableConfig().label} {getCurrentVariableConfig().description} in{' '}
                      <span className="text-cyan-400">
                        {getThresholdExceedanceCount(historyData, selectedVariable)}
                      </span> of 30 years
                    </span>
                    <span className="mx-2 text-slate-400">→</span>
                    <span className="text-lg font-bold" style={{ color: getCurrentVariableConfig().color }}>
                      {getVariableProbability(historyData, selectedVariable)}% chance
                    </span>
                    <span className="text-slate-400 ml-1">of {getCurrentVariableConfig().thresholdLabel}</span>
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Probability Distribution Chart - Full Width */}
            <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700/50 shadow-2xl">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-slate-200 text-lg">
                      <CloudRain className="h-5 w-5 text-cyan-400" />
                      Probability Distribution
                    </CardTitle>
                    <p className="text-sm text-slate-400">
                      {getCurrentDistributionVariableConfig().label} distribution & threshold analysis
                    </p>
                  </div>
                  <div className="w-48">
                    <select
                      value={selectedDistributionVariable}
                      onChange={(e) => setSelectedDistributionVariable(e.target.value)}
                      className="w-full bg-slate-800/50 border-slate-600 text-slate-200 rounded-md px-3 py-2 text-sm focus:border-cyan-400 focus:ring-cyan-400/20"
                    >
                      {Object.entries(variableConfig).map(([key, config]) => (
                        <option key={key} value={key}>
                          {config.icon} {config.label} ({config.unit})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Line
                    data={{
                      labels: generateDistributionLabels(historyData.historical_data, selectedDistributionVariable),
                      datasets: [
                        {
                          label: `${getCurrentDistributionVariableConfig().label} Distribution`,
                          data: generateDistributionData(historyData.historical_data, selectedDistributionVariable),
                          borderColor: getCurrentDistributionVariableConfig().color,
                          backgroundColor: (context) => {
                            const chart = context.chart;
                            const {ctx, chartArea} = chart;
                            if (!chartArea) return null;
                            
                            const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                            const color = getCurrentDistributionVariableConfig().color;
                            gradient.addColorStop(0, color + '20');
                            gradient.addColorStop(1, color + '50');
                            return gradient;
                          },
                          fill: true,
                          tension: 0.4,
                          pointRadius: 0,
                        },
                        {
                          label: `Threshold Area (${getCurrentDistributionVariableConfig().thresholdLabel})`,
                          data: generateThresholdArea(historyData.historical_data, selectedDistributionVariable, getCurrentDistributionVariableConfig().threshold),
                          borderColor: '#ef4444',
                          backgroundColor: 'rgba(239, 68, 68, 0.3)',
                          fill: true,
                          tension: 0.4,
                          pointRadius: 0,
                        }
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'top',
                          labels: {
                            color: '#cbd5e1',
                          },
                        },
                      },
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: `${getCurrentDistributionVariableConfig().label} (${getCurrentDistributionVariableConfig().unit})`,
                            color: '#cbd5e1'
                          },
                          ticks: { color: '#64748b' },
                          grid: { color: '#334155' },
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Probability Density (%)',
                            color: '#cbd5e1'
                          },
                          ticks: { color: '#64748b' },
                          grid: { color: '#334155' },
                        },
                      },
                    }}
                  />
                </div>
                <div className="mt-4 text-center">
                  <p className="text-sm text-slate-300">
                    <span className="font-semibold">
                      Shaded area above {getCurrentDistributionVariableConfig().threshold}{getCurrentDistributionVariableConfig().unit}{' '}
                      {getCurrentDistributionVariableConfig().label.toLowerCase()} = {' '}
                    </span>
                    <span style={{ color: getCurrentDistributionVariableConfig().color }} className="text-lg font-bold">
                      {getVariableProbability(historyData, selectedDistributionVariable)}% chance
                    </span>
                    <span className="text-slate-400 ml-1">of {getCurrentDistributionVariableConfig().thresholdLabel}</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;