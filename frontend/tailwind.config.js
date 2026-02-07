/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0a0a1a',
        'dark-navy': '#0f1729',
        'cyan-accent': '#00d4ff',
        'green-accent': '#00ff88',
        'red-accent': '#ff4444',
        'gold-accent': '#ffd700',
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
