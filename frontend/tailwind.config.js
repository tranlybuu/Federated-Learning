/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ["./src/**/*.{html,js,vue}"],
  theme: {
    extend: {
      screens: {
				sm: "576px",
				md: "768px",
				lg: "992px",
				xl: "1280px",
			},
      colors: {
        main: "white",
        submain: "red",
        opmain: "black",
      }
    },
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: true,
  },
}

