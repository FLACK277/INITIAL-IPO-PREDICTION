import { motion } from 'framer-motion';

export default function AboutSection() {
  const features = [
    {
      title: 'Ensemble Models',
      description: 'Combines Random Forest, Gradient Boosting, Logistic Regression, and SVM for robust predictions',
      icon: '🤖',
    },
    {
      title: 'Neural Networks',
      description: 'Deep learning model with 85% training accuracy and advanced regularization techniques',
      icon: '🧠',
    },
    {
      title: 'Feature Engineering',
      description: 'Advanced feature creation including interaction terms, log transformations, and statistical metrics',
      icon: '⚙️',
    },
    {
      title: 'Real-time Analysis',
      description: 'Instant predictions with confidence scores and investment recommendations',
      icon: '⚡',
    },
  ];

  const models = [
    { name: 'Logistic Regression', auc: '0.7025', accuracy: '65.6%' },
    { name: 'Random Forest', auc: '0.6837', accuracy: '68.8%' },
    { name: 'SVM', auc: '0.6562', accuracy: '64.1%' },
    { name: 'Voting Ensemble', auc: '0.6611', accuracy: '67.2%' },
    { name: 'Neural Network', auc: '~0.90', accuracy: '74.0%' },
  ];

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
            About Our AI Models
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            Our IPO prediction system leverages cutting-edge machine learning techniques
            trained on years of Indian IPO market data
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="glass-card p-6 glass-card-hover"
            >
              <div className="text-4xl mb-3">{feature.icon}</div>
              <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Model Performance Table */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-2xl font-bold text-white mb-6">Model Performance Metrics</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-300">Model</th>
                  <th className="text-right py-3 px-4 text-gray-300">AUC Score</th>
                  <th className="text-right py-3 px-4 text-gray-300">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    viewport={{ once: true }}
                    className="border-b border-gray-800 hover:bg-dark-navy/50 transition-colors"
                  >
                    <td className="py-3 px-4 text-white font-medium">{model.name}</td>
                    <td className="py-3 px-4 text-right text-cyan-accent font-mono">
                      {model.auc}
                    </td>
                    <td className="py-3 px-4 text-right text-green-accent font-mono">
                      {model.accuracy}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* How It Works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="glass-card p-8"
        >
          <h3 className="text-2xl font-bold text-white mb-6">How It Works</h3>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 bg-cyan-accent/20 rounded-full flex items-center justify-center text-cyan-accent font-bold">
                1
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-1">Data Input</h4>
                <p className="text-gray-400">
                  Enter IPO details including issue size, price, and subscription data from different investor categories
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 bg-green-accent/20 rounded-full flex items-center justify-center text-green-accent font-bold">
                2
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-1">Feature Engineering</h4>
                <p className="text-gray-400">
                  Advanced algorithms create 20+ engineered features including interaction terms, logarithmic transformations, and statistical metrics
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 bg-gold-accent/20 rounded-full flex items-center justify-center text-gold-accent font-bold">
                3
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-1">Multi-Model Prediction</h4>
                <p className="text-gray-400">
                  Multiple ML models analyze the data simultaneously, each contributing their unique perspective
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 bg-cyan-accent/20 rounded-full flex items-center justify-center text-cyan-accent font-bold">
                4
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-1">Consensus & Recommendation</h4>
                <p className="text-gray-400">
                  Results are aggregated to provide a consensus prediction with confidence scores and actionable investment recommendations
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
