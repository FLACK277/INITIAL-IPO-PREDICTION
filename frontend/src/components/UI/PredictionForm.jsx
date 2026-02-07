import { useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { API_ENDPOINTS } from '../../utils/api';
import LoadingSpinner from './LoadingSpinner';
import PredictionResults from './PredictionResults';

export default function PredictionForm() {
  const [formData, setFormData] = useState({
    ipo_name: '',
    issue_size: '',
    issue_price: '',
    subscription_qib: '',
    subscription_hni: '',
    subscription_rii: '',
    subscription_total: '',
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        ipo_name: formData.ipo_name,
        issue_size: parseFloat(formData.issue_size),
        issue_price: parseFloat(formData.issue_price),
        subscription_qib: parseFloat(formData.subscription_qib),
        subscription_hni: parseFloat(formData.subscription_hni),
        subscription_rii: parseFloat(formData.subscription_rii),
        subscription_total: parseFloat(formData.subscription_total),
      };

      const response = await axios.post(API_ENDPOINTS.PREDICT, payload);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please check your input and try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full px-4 py-3 bg-dark-navy border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-accent transition-colors";

  return (
    <div className="w-full max-w-6xl mx-auto px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
        className="glass-card p-8"
      >
        <h2 className="text-3xl font-bold text-center mb-6 bg-gradient-to-r from-cyan-accent to-green-accent bg-clip-text text-transparent">
          Predict IPO Performance
        </h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* IPO Name */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                IPO Name
              </label>
              <input
                type="text"
                name="ipo_name"
                value={formData.ipo_name}
                onChange={handleChange}
                required
                placeholder="e.g., Acme Corporation"
                className={inputClass}
              />
            </div>

            {/* Issue Size */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Issue Size (₹ Crores)
              </label>
              <input
                type="number"
                name="issue_size"
                value={formData.issue_size}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 1000"
                className={inputClass}
              />
            </div>

            {/* Issue Price */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Issue Price (₹)
              </label>
              <input
                type="number"
                name="issue_price"
                value={formData.issue_price}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 250"
                className={inputClass}
              />
            </div>

            {/* QIB Subscription */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                QIB Subscription (times)
              </label>
              <input
                type="number"
                name="subscription_qib"
                value={formData.subscription_qib}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 5.5"
                className={inputClass}
              />
            </div>

            {/* HNI Subscription */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                HNI Subscription (times)
              </label>
              <input
                type="number"
                name="subscription_hni"
                value={formData.subscription_hni}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 3.2"
                className={inputClass}
              />
            </div>

            {/* RII Subscription */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Retail/RII Subscription (times)
              </label>
              <input
                type="number"
                name="subscription_rii"
                value={formData.subscription_rii}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 2.8"
                className={inputClass}
              />
            </div>

            {/* Total Subscription */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Total Subscription (times)
              </label>
              <input
                type="number"
                name="subscription_total"
                value={formData.subscription_total}
                onChange={handleChange}
                required
                step="0.01"
                min="0"
                placeholder="e.g., 4.1"
                className={inputClass}
              />
            </div>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            disabled={loading}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full py-4 bg-gradient-to-r from-cyan-accent to-green-accent text-dark-bg font-bold text-lg rounded-lg hover:shadow-lg hover:shadow-cyan-accent/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Predict IPO Performance'}
          </motion.button>
        </form>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-6 p-4 bg-red-accent/10 border border-red-accent rounded-lg text-red-accent"
          >
            <p className="font-medium">Error</p>
            <p className="text-sm mt-1">{error}</p>
          </motion.div>
        )}

        {/* Loading Display */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-8 flex justify-center"
          >
            <LoadingSpinner />
          </motion.div>
        )}
      </motion.div>

      {/* Results Display */}
      {result && !loading && <PredictionResults result={result} />}
    </div>
  );
}
