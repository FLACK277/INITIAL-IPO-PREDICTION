import { motion } from 'framer-motion';
import Background3D from './components/3D/Background3D';
import PredictionForm from './components/UI/PredictionForm';
import VisualizationSection from './components/UI/VisualizationSection';
import AboutSection from './components/UI/AboutSection';
import './App.css';

function App() {
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="relative min-h-screen">
      {/* 3D Background */}
      <Background3D />

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold bg-gradient-to-r from-cyan-accent to-green-accent bg-clip-text text-transparent"
          >
            IPO Predict AI
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex gap-6"
          >
            <button
              onClick={() => scrollToSection('home')}
              className="text-gray-300 hover:text-cyan-accent transition-colors"
            >
              Home
            </button>
            <button
              onClick={() => scrollToSection('predict')}
              className="text-gray-300 hover:text-cyan-accent transition-colors"
            >
              Predict
            </button>
            <button
              onClick={() => scrollToSection('visualizations')}
              className="text-gray-300 hover:text-cyan-accent transition-colors"
            >
              Analytics
            </button>
            <button
              onClick={() => scrollToSection('about')}
              className="text-gray-300 hover:text-cyan-accent transition-colors"
            >
              About
            </button>
          </motion.div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10 pt-20">
        {/* Hero Section */}
        <section id="home" className="min-h-screen flex items-center justify-center px-4">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-6xl md:text-8xl font-bold mb-6">
                <span className="bg-gradient-to-r from-cyan-accent via-green-accent to-gold-accent bg-clip-text text-transparent">
                  IPO Predict AI
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
                Harness the power of machine learning to predict IPO performance
                with confidence
              </p>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => scrollToSection('predict')}
                className="px-8 py-4 bg-gradient-to-r from-cyan-accent to-green-accent text-dark-bg font-bold text-lg rounded-full hover:shadow-lg hover:shadow-cyan-accent/50 transition-all"
              >
                Start Predicting
              </motion.button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 1 }}
              className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto"
            >
              <div className="glass-card p-6 glass-card-hover">
                <div className="text-4xl mb-3">🎯</div>
                <div className="text-2xl font-bold text-cyan-accent mb-2">90% AUC</div>
                <div className="text-gray-400">Neural Network Accuracy</div>
              </div>
              <div className="glass-card p-6 glass-card-hover">
                <div className="text-4xl mb-3">🚀</div>
                <div className="text-2xl font-bold text-green-accent mb-2">5 Models</div>
                <div className="text-gray-400">Ensemble Prediction</div>
              </div>
              <div className="glass-card p-6 glass-card-hover">
                <div className="text-4xl mb-3">📊</div>
                <div className="text-2xl font-bold text-gold-accent mb-2">326 IPOs</div>
                <div className="text-gray-400">Training Dataset</div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Prediction Section */}
        <section id="predict" className="min-h-screen flex items-center justify-center py-20 px-4">
          <PredictionForm />
        </section>

        {/* Visualizations Section */}
        <section id="visualizations" className="min-h-screen flex items-center justify-center py-20 px-4">
          <VisualizationSection />
        </section>

        {/* About Section */}
        <section id="about" className="min-h-screen flex items-center justify-center py-20 px-4">
          <AboutSection />
        </section>

        {/* Footer */}
        <footer className="relative z-10 py-8 px-4 border-t border-white/10">
          <div className="max-w-6xl mx-auto text-center">
            <p className="text-gray-400">
              © 2026 IPO Predict AI. Powered by Machine Learning.
            </p>
            <p className="text-gray-500 text-sm mt-2">
              Disclaimer: Predictions are for informational purposes only. Always do your own research before investing.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
