"use client";

import { useEffect } from "react";
import L from "leaflet";
import { MapContainer, TileLayer, Rectangle, useMapEvents } from "react-leaflet";
import type { LatLngBounds } from "leaflet";
import "leaflet/dist/leaflet.css";

interface WindMapProps {
  bounds: LatLngBounds | null;
  onBoundsChange: (bounds: LatLngBounds) => void;
}

function BoundsSelector({ onBoundsChange }: { onBoundsChange: (b: LatLngBounds) => void }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      const delta = 1.5;
      onBoundsChange(
        L.latLngBounds([lat - delta, lng - delta], [lat + delta, lng + delta]),
      );
    },
  });
  return null;
}

export default function WindMap({ bounds, onBoundsChange }: WindMapProps) {
  useEffect(() => {
    // Fix Leaflet default icon paths in Next.js
    import("leaflet").then((L) => {
      delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });
    });
  }, []);

  return (
    <MapContainer
      center={[12.5, 78.0]}
      zoom={5}
      className="h-full w-full rounded-lg"
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <BoundsSelector onBoundsChange={onBoundsChange} />
      {bounds && (
        <Rectangle
          bounds={bounds}
          pathOptions={{ color: "#38bdf8", weight: 2, fillOpacity: 0.15 }}
        />
      )}
    </MapContainer>
  );
}
