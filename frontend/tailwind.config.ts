import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ocean: {
          950: "#0a1628",
          900: "#0f2744",
          800: "#1a3a5c",
        },
      },
    },
  },
  plugins: [],
};

export default config;
