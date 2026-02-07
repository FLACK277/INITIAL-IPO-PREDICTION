import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_ENDPOINTS } from '../../utils/api';

const modelPerformanceData = [
  { model: 'Logistic Reg', auc: 0.7025, accuracy: 0.656 },
  { model: 'Random Forest', auc: 0.6837, accuracy: 0.688 },
  { model: 'SVM', auc: 0.6562, accuracy: 0.641 },
  { model: 'Ensemble', auc: 0.6611, accuracy: 0.672 },
  { model: 'Neural Net', auc: 0.90, accuracy: 0.740 },
];

export default function VisualizationSection() {
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(API_ENDPOINTS.HISTORICAL_DATA);
      // Take a sample of the data for visualization
      const sample = response.data.data.slice(0, 50);
      setHistoricalData(sample);
    } catch (error) {
      console.error('Failed to fetch historical data:', error);
    } finally {
      setLoading(false);
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card p-3">
          <p className="text-white font-medium">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
        className="space-y-8"
      >
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-accent to-green-accent bg-clip-text text-transparent">
            Model Performance & Insights
          </h2>
          <p className="text-gray-400 text-lg">
            Data-driven analysis of IPO prediction accuracy
          </p>
        </div>

        {/* Model Performance Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-2xl font-bold text-white mb-6">Model Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelPerformanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="model" stroke="#fff" />
              <YAxis stroke="#fff" />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="auc" fill="#00d4ff" name="AUC Score" />
              <Bar dataKey="accuracy" fill="#00ff88" name="Accuracy" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Historical IPO Performance */}
        {!loading && historicalData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            viewport={{ once: true }}
            className="glass-card p-6"
          >
            <h3 className="text-2xl font-bold text-white mb-6">Historical IPO Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  type="number"
                  dataKey="Subscription_Total"
                  name="Total Subscription"
                  stroke="#fff"
                  label={{ value: 'Subscription (x)', position: 'insideBottom', offset: -5, fill: '#fff' }}
                />
                <YAxis
                  type="number"
                  dataKey="Listing_Gains_Percent"
                  name="Listing Gains %"
                  stroke="#fff"
                  label={{ value: 'Listing Gains (%)', angle: -90, position: 'insideLeft', fill: '#fff' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Scatter name="IPOs" data={historicalData}>
                  {historicalData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.Listing_Gains_Percent > 0 ? '#00ff88' : '#ff4444'}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-sm text-gray-400 mt-4 text-center">
              Scatter plot showing relationship between subscription rate and listing gains
            </p>
          </motion.div>
        )}

        {/* Key Insights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <div className="glass-card p-6 glass-card-hover">
            <div className="text-cyan-accent text-4xl font-bold mb-2">90%</div>
            <div className="text-gray-300 font-medium">Neural Network AUC</div>
            <div className="text-sm text-gray-400 mt-2">
              Highest performing model for IPO prediction
            </div>
          </div>

          <div className="glass-card p-6 glass-card-hover">
            <div className="text-green-accent text-4xl font-bold mb-2">70%+</div>
            <div className="text-gray-300 font-medium">Logistic Regression AUC</div>
            <div className="text-sm text-gray-400 mt-2">
              Best traditional ML model performance
            </div>
          </div>

          <div className="glass-card p-6 glass-card-hover">
            <div className="text-gold-accent text-4xl font-bold mb-2">326</div>
            <div className="text-gray-300 font-medium">IPOs Analyzed</div>
            <div className="text-sm text-gray-400 mt-2">
              Years of Indian IPO market data
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
