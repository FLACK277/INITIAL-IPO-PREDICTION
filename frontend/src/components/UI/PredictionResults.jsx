import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

function ConfidenceGauge({ confidence, prediction }) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(confidence), 100);
    return () => clearTimeout(timer);
  }, [confidence]);

  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (animatedValue / 100) * circumference;
  
  const color = prediction === 'PROFITABLE' ? '#00ff88' : '#ff4444';

  return (
    <div className="relative w-32 h-32">
      <svg className="transform -rotate-90 w-32 h-32">
        <circle
          cx="64"
          cy="64"
          r="45"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
          fill="none"
        />
        <motion.circle
          cx="64"
          cy="64"
          r="45"
          stroke={color}
          strokeWidth="8"
          fill="none"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center flex-col">
        <motion.span
          className="text-2xl font-bold"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          style={{ color }}
        >
          {Math.round(animatedValue)}%
        </motion.span>
        <span className="text-xs text-gray-400">Confidence</span>
      </div>
    </div>
  );
}

function RiskBadge({ riskLevel }) {
  const colors = {
    'Low': 'bg-green-accent/20 text-green-accent border-green-accent',
    'Medium': 'bg-yellow-500/20 text-yellow-500 border-yellow-500',
    'Medium-High': 'bg-orange-500/20 text-orange-500 border-orange-500',
    'High': 'bg-red-accent/20 text-red-accent border-red-accent',
  };

  return (
    <span className={`px-4 py-2 rounded-full border ${colors[riskLevel] || colors['Medium']} font-semibold text-sm`}>
      {riskLevel} Risk
    </span>
  );
}

function RecommendationBadge({ recommendation }) {
  const isBuy = recommendation.includes('BUY');
  const isAvoid = recommendation.includes('AVOID');
  
  let colorClass = 'bg-yellow-500/20 text-yellow-500 border-yellow-500';
  if (isBuy) {
    colorClass = 'bg-green-accent/20 text-green-accent border-green-accent';
  } else if (isAvoid) {
    colorClass = 'bg-red-accent/20 text-red-accent border-red-accent';
  }

  return (
    <span className={`px-4 py-2 rounded-full border ${colorClass} font-bold text-lg`}>
      {recommendation}
    </span>
  );
}

export default function PredictionResults({ result }) {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  const isProfitable = result.prediction === 'PROFITABLE';
  const gainColor = result.predicted_listing_gain_percent > 0 ? 'text-green-accent' : 'text-red-accent';

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="mt-8 space-y-6"
    >
      {/* Warning if using mock predictions */}
      {result.warning && (
        <motion.div
          variants={itemVariants}
          className="glass-card p-4 border-yellow-500/50"
        >
          <p className="text-yellow-500 text-sm">{result.warning}</p>
        </motion.div>
      )}

      {/* Main Result Card */}
      <motion.div
        variants={itemVariants}
        className="glass-card p-8"
      >
        <div className="text-center mb-6">
          <h3 className="text-2xl font-bold text-gray-300 mb-2">{result.ipo_name}</h3>
          <div className="flex items-center justify-center gap-4">
            <span className={`text-4xl font-bold ${isProfitable ? 'text-green-accent' : 'text-red-accent'}`}>
              {result.prediction}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap justify-center items-center gap-8 mb-8">
          <ConfidenceGauge confidence={result.confidence} prediction={result.prediction} />
          
          <div className="flex flex-col gap-3">
            <RiskBadge riskLevel={result.risk_level} />
            <RecommendationBadge recommendation={result.recommendation} />
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <motion.div
            variants={itemVariants}
            className="glass-card p-6 glass-card-hover"
          >
            <div className="text-sm text-gray-400 mb-2">Expected Listing Gain</div>
            <div className={`text-3xl font-bold ${gainColor}`}>
              {result.predicted_listing_gain_percent > 0 ? '+' : ''}
              {result.predicted_listing_gain_percent}%
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="glass-card p-6 glass-card-hover"
          >
            <div className="text-sm text-gray-400 mb-2">Predicted Opening Price</div>
            <div className="text-3xl font-bold text-cyan-accent">
              ₹{result.predicted_opening_price}
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="glass-card p-6 glass-card-hover"
          >
            <div className="text-sm text-gray-400 mb-2">Prediction Probability</div>
            <div className="text-3xl font-bold text-gold-accent">
              {(result.probability * 100).toFixed(1)}%
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="glass-card p-6 glass-card-hover"
          >
            <div className="text-sm text-gray-400 mb-2">Risk Assessment</div>
            <div className="text-2xl font-bold text-white">
              {result.risk_level}
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Model Results */}
      <motion.div
        variants={itemVariants}
        className="glass-card p-6"
      >
        <h4 className="text-xl font-bold text-gray-300 mb-4">Model Predictions</h4>
        <div className="space-y-3">
          {result.model_results.map((modelResult, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className="flex items-center justify-between p-4 bg-dark-navy/50 rounded-lg"
            >
              <span className="text-gray-300 font-medium">{modelResult.model}</span>
              <div className="flex items-center gap-4">
                <span className="text-gray-400">
                  {(modelResult.probability * 100).toFixed(1)}%
                </span>
                <span className={`font-semibold ${
                  modelResult.prediction === 'PROFITABLE' ? 'text-green-accent' : 'text-red-accent'
                }`}>
                  {modelResult.prediction}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Investment Recommendation */}
      <motion.div
        variants={itemVariants}
        className={`glass-card p-6 border-2 ${
          isProfitable ? 'border-green-accent/50' : 'border-red-accent/50'
        }`}
      >
        <div className="text-center">
          <div className="text-lg text-gray-300 mb-2">Investment Recommendation</div>
          <div className={`text-3xl font-bold ${
            isProfitable ? 'text-green-accent' : 'text-red-accent'
          }`}>
            {result.recommendation}
          </div>
          {isProfitable ? (
            <p className="text-gray-400 mt-3">
              Based on our analysis, this IPO shows potential for positive returns.
              {result.confidence > 70 && ' High confidence in positive listing gains.'}
            </p>
          ) : (
            <p className="text-gray-400 mt-3">
              Based on our analysis, this IPO may not be suitable for investment.
              {result.confidence > 70 && ' High confidence in negative listing performance.'}
            </p>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}
