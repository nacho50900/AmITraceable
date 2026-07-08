import React from 'react';
import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { ImageLocationPoint } from '../types';

interface LocationMapProps {
  points: ImageLocationPoint[];
}

// Centro aproximado de España peninsular, usado como fallback si no hay
// puntos con coordenadas (el mapa igual se muestra, solo que centrado).
const SPAIN_CENTER: [number, number] = [40.0, -3.7];

function confidenceColor(confidence: number): string {
  if (confidence >= 0.7) return '#d3403a'; // alta confianza -> rojo
  if (confidence >= 0.4) return '#d6a51c'; // media -> ámbar
  return '#3aa657'; // baja -> verde (menos preocupante)
}

const LocationMap: React.FC<LocationMapProps> = ({ points }) => {
  const validPoints = points.filter((p) => p.lat !== null && p.lon !== null);

  if (validPoints.length === 0) {
    return (
      <p className="note">
        No se ha podido estimar la ubicación de ninguna de tus fotos (el índice de
        geolocalización no está disponible, o ninguna imagen tuvo suficiente similitud
        con el índice de referencia).
      </p>
    );
  }

  const center: [number, number] = [
    validPoints.reduce((sum, p) => sum + (p.lat ?? 0), 0) / validPoints.length,
    validPoints.reduce((sum, p) => sum + (p.lon ?? 0), 0) / validPoints.length,
  ];

  return (
    <>
      <MapContainer
        center={validPoints.length > 0 ? center : SPAIN_CENTER}
        zoom={6}
        style={{ height: '360px', width: '100%', borderRadius: '12px' }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {validPoints.map((point, i) => (
          <CircleMarker
            key={i}
            center={[point.lat as number, point.lon as number]}
            radius={8 + point.confidence * 10}
            pathOptions={{
              color: confidenceColor(point.confidence),
              fillColor: confidenceColor(point.confidence),
              fillOpacity: 0.5,
            }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              <strong>{point.province}</strong>
              <br />
              Confianza: {Math.round(point.confidence * 100)}%
              <br />
              {(point.lat as number).toFixed(4)}, {(point.lon as number).toFixed(4)}
            </Tooltip>
            <Popup>
              <strong>{point.province}</strong>
              <br />
              Confianza: {Math.round(point.confidence * 100)}%
              <br />
              Coordenadas: {(point.lat as number).toFixed(4)}, {(point.lon as number).toFixed(4)}
              <br />
              <a href={point.permalink} target="_blank" rel="noreferrer">
                Ver publicación
              </a>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      <p className="note">
        Cada punto es una estimación aproximada (no una ubicación exacta) basada en
        similitud visual contra un índice de imágenes de referencia de España. El tamaño y
        color del punto reflejan la confianza de la estimación, no la precisión exacta del
        lugar.
      </p>
    </>
  );
};

export default LocationMap;
