import React from 'react';
import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { ImageLocationPoint, Platform } from '../types';

interface LocationMapProps {
  points: ImageLocationPoint[];
  platform: Platform;
  /** report.geolocation_available -- false si el índice FAISS no está
   * construido en este servidor (o faltan sus dependencias opcionales).
   * Distinto de "se analizaron fotos pero ninguna dio resultado fiable":
   * nunca se muestran ambos mensajes a la vez. */
  available: boolean;
}

// Centro aproximado de España peninsular, usado como fallback si no hay
// puntos con coordenadas (el mapa igual se muestra, solo que centrado).
const SPAIN_CENTER: [number, number] = [40.0, -3.7];

function confidenceColor(confidence: number): string {
  if (confidence >= 0.7) return '#d3403a'; // alta confianza -> rojo
  if (confidence >= 0.4) return '#d6a51c'; // media -> ámbar
  return '#3aa657'; // baja -> verde (menos preocupante)
}

function formatDate(isoDate: string | null): string {
  if (!isoDate) return 'fecha desconocida';
  return new Date(isoDate).toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' });
}

const LocationMap: React.FC<LocationMapProps> = ({ points, platform, available }) => {
  if (platform !== 'instagram') {
    return (
      <p className="note">
        Reddit no tiene fotos que analizar aquí -- esta sección solo aplica a Instagram.
      </p>
    );
  }

  if (!available) {
    return (
      <p className="note">
        El índice de geolocalización por imagen no está construido en este servidor, así que
        esta función no está disponible ahora mismo.
      </p>
    );
  }

  // Las fotos no representativas (vecinos más parecidos repartidos por una
  // zona demasiado amplia, ver ImageLocationEstimate.representative en
  // app/vision/geolocation.py) NO se pintan en el mapa ni en la lista
  // principal -- aparecen aparte, más abajo, solo con el enlace a la
  // publicación, para no sugerir una ubicación fiable que en realidad no
  // lo es.
  const representativePoints = points.filter((p) => p.representative);
  const nonRepresentativePoints = points.filter((p) => !p.representative);

  if (points.length === 0) {
    return (
      <p className="note">
        No se ha podido analizar ninguna de tus fotos (puede que no tuvieras fotos públicas,
        o que no se pudieran descargar).
      </p>
    );
  }

  const mappablePoints = representativePoints.filter((p) => p.lat !== null && p.lon !== null);
  const center: [number, number] =
    mappablePoints.length > 0
      ? [
          mappablePoints.reduce((sum, p) => sum + (p.lat ?? 0), 0) / mappablePoints.length,
          mappablePoints.reduce((sum, p) => sum + (p.lon ?? 0), 0) / mappablePoints.length,
        ]
      : SPAIN_CENTER;

  return (
    <>
      {representativePoints.length === 0 && (
        <p className="note">
          Ninguna de tus fotos analizadas fue lo bastante representativa de un lugar concreto como
          para estimar una ubicación fiable.
        </p>
      )}

      {mappablePoints.length > 0 && (
        <MapContainer
          center={center}
          zoom={6}
          style={{ height: '360px', width: '100%', borderRadius: '12px' }}
          scrollWheelZoom={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {mappablePoints.map((point) => (
            <CircleMarker
              key={point.permalink}
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
      )}

      {representativePoints.length > 0 && (
        <>
          <p className="note">
            Cada foto se ha comparado contra un índice de imágenes de referencia de España; el
            resultado es una similitud visual aproximada, no una ubicación exacta. Se muestran{' '}
            <strong>todas</strong> las fotos con una estimación representativa de un lugar
            concreto, incluidas las de confianza baja.
          </p>

          <ul className="image-location-list">
            {representativePoints.map((point) => (
              <li key={point.permalink}>
                <details className="photo-details">
                  <summary>
                    <a
                      href={point.permalink}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Ver publicación
                    </a>
                    <span>
                      {formatDate(point.created_utc)} — {point.province} — confianza{' '}
                      {Math.round(point.confidence * 100)}%
                      {point.lat === null && ' (sin coordenadas para el mapa)'}
                    </span>
                  </summary>
                  <p className="photo-visual-description">
                    {point.visual_description
                      ? point.visual_description
                      : 'Sin descripción visual disponible para esta foto.'}
                  </p>
                </details>
              </li>
            ))}
          </ul>
        </>
      )}

      {nonRepresentativePoints.length > 0 && (
        <details className="image-location-details">
          <summary>Imágenes no representativas ({nonRepresentativePoints.length})</summary>
          <p className="note">
            Estas fotos se analizaron, pero sus vecinos más parecidos en el índice están
            repartidos por una zona demasiado amplia como para asignarles una ubicación fiable --
            no se muestran en el mapa ni cuentan para estimar tu residencia.
          </p>
          <ul className="image-location-list image-location-list-scroll">
            {nonRepresentativePoints.map((point) => (
              <li key={point.permalink}>
                <details className="photo-details">
                  <summary>
                    <a
                      href={point.permalink}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Ver publicación
                    </a>
                    <span>{formatDate(point.created_utc)}</span>
                  </summary>
                  <p className="photo-visual-description">
                    {point.visual_description
                      ? point.visual_description
                      : 'Sin descripción visual disponible para esta foto.'}
                  </p>
                </details>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
};

export default LocationMap;
