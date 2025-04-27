/** @type {import('tailwindcss').Config} */
const config = {
    content: [
      './src/**/*.{js,ts,jsx,tsx}',
    ],
    theme: {
      extend: {
        colors: {
          primary: '#ffffff', // for white text
          background: '#000000', // for black background
        },
        fontFamily: {
          sans: ['Montserrat', 'sans-serif'],
        },
      },
    },
    plugins: [],
  }
  
  export default config;
  