/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      boxShadow: {
        abyss: "0 24px 80px rgba(1, 11, 18, 0.35)",
      },
      backgroundImage: {
        "scan-lines":
          "linear-gradient(rgba(144, 232, 221, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(144, 232, 221, 0.08) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};
