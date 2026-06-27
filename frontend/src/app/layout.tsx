import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanWind AI",
  description: "SAR-based coastal wind field estimation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
