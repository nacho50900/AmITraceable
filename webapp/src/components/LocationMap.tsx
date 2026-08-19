import React from 'react';
import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useTranslation } from 'react-i18next';
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

// La foto de perfil no tiene página de publicación a la que enlazar (no es
// un post) -- `permalink`, en ese caso, es la URL directa de la propia
// imagen (ver app/vision/geolocation.py::estimate_locations_for_posts). Si
// por lo que sea no hay ninguna URL utilizable, se muestra el texto suelto
// "Foto de perfil" sin enlace, en vez de un enlace roto.
function PhotoLink({
  point,
  t,
  onClick,
}: {
  point: ImageLocationPoint;
  t: (key: string, opts?: Record<string, unknown>) => string;
  onClick?: (e: React.MouseEvent) => void;
}) {
  const label = point.is_profile_picture
    ? t('components.locationMap.profilePictureLabel')
    : point.visual_description_general || t('components.locationMap.viewPost');

  if (point.is_profile_picture && !point.permalink) {
    return <span className="photo-link-label">{label}</span>;
  }

  return (
    <a href={point.permalink} target="_blank" rel="noreferrer" onClick={onClick}>
      {label}
    </a>
  );
}

const LocationMap: React.FC<LocationMapProps> = ({ points, platform, available }) => {
  const { t, i18n } = useTranslation();
  const dateLocale = i18n.language?.split('-')[0] === 'en' ? 'en-US' : 'es-ES';

  function formatDate(isoDate: string | null): string | null {
    if (!isoDate) return null;
    return new Date(isoDate).toLocaleDateString(dateLocale, { day: 'numeric', month: 'short', year: 'numeric' });
  }

  if (platform !== 'instagram') {
    return <p className="note">{t('components.locationMap.redditNoPhotos')}</p>;
  }

  if (!available) {
    return <p className="note">{t('components.locationMap.indexUnavailable')}</p>;
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
    return <p className="note">{t('components.locationMap.noPhotos')}</p>;
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
      {representativePoints.length === 0 && <p className="note">{t('components.locationMap.noRepresentative')}</p>}

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
          {mappablePoints.map((point, idx) => (
            <CircleMarker
              key={point.permalink ?? `mappable-${idx}`}
              center={[point.lat as number, point.lon as number]}
              radius={8 + point.confidence * 10}
              pathOptions={{
                color: confidenceColor(point.confidence),
                fillColor: confidenceColor(point.confidence),
                fillOpacity: 0.5,
              }}
            >
              {point.created_utc ? (
               <>
                 <Tooltip direction="top" offset={[0, -8]} opacity={1}>
                   {/* point.province viene del backend (nombre de provincia en
                       español, INE); no se traduce en esta fase. */}
                   <strong>{point.province}</strong>
                   <br />
                   {t('components.locationMap.confidence', { value: Math.round(point.confidence * 100) })}
                   <br />
                   {(point.lat as number).toFixed(4)}, {(point.lon as number).toFixed(4)}
                 </Tooltip>
                 <Popup>
                   {t('components.locationMap.confidence', { value: Math.round(point.confidence * 100) })}
                   <br />
                   {t('components.locationMap.coordinates', {
                     lat: (point.lat as number).toFixed(4),
                     lon: (point.lon as number).toFixed(4),
                   })}
                   <br />
                   <PhotoLink point={point} t={t} />
                 </Popup>
               </>
              ) : null}
            </CircleMarker>
          ))}
        </MapContainer>
      )}

      {representativePoints.length > 0 && (
        <>
          <p className="note">
            {t('components.locationMap.comparisonNote')} <strong>{t('components.locationMap.comparisonNoteAll')}</strong>{' '}
            {t('components.locationMap.comparisonNoteEnd')}
          </p>

          <ul className="image-location-list">
            {representativePoints.map((point, idx) => (
                          <li key={point.permalink ?? `rep-${idx}`}>
                <details className="photo-details">
                  <summary>
                    <PhotoLink point={point} t={t} onClick={(e) => e.stopPropagation()} />
                    <span>
                      {point.province && <>{point.province}{' — '}</>}
                      {point.created_utc && <>{formatDate(point.created_utc)} — </>}
                      {t('components.locationMap.confidenceInline', { value: Math.round(point.confidence * 100) })}
                      {point.lat === null && t('components.locationMap.noCoordinates')}
                    </span>
                  </summary>
                  <p className="photo-visual-description">
                    {point.visual_description
                      ? point.visual_description
                      : t('components.locationMap.noVisualDescription')}
                  </p>
                </details>
              </li>
            ))}
          </ul>
        </>
      )}

      {nonRepresentativePoints.length > 0 && (
        <details className="image-location-details">
          <summary>
            {t('components.locationMap.nonRepresentativeSummary', { count: nonRepresentativePoints.length })}
          </summary>
          <p className="note">{t('components.locationMap.nonRepresentativeNote')}</p>
          <ul className="image-location-list image-location-list-scroll">
            {nonRepresentativePoints.map((point, idx) => (
                          <li key={point.permalink ?? `nonrep-${idx}`}>
                <details className="photo-details">
                  <summary>
                    <PhotoLink point={point} t={t} onClick={(e) => e.stopPropagation()} />
                    <span>{formatDate(point.created_utc)}</span>
                  </summary>
                  <p className="photo-visual-description">
                    {point.visual_description
                      ? point.visual_description
                      : t('components.locationMap.noVisualDescription')}
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
