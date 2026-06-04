/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#6366f1",
          hover: "#4f46e5",
        },
      },
    },
  },
  plugins: [],
};
