import React from 'react';
import { useTranslation } from 'react-i18next';
import type { InferredAttribute } from '../types';

interface InferredAttributesListProps {
  attributes: InferredAttribute[];
}

// Categorías conocidas con etiqueta traducida -- las que llegan de la IA
// (ver app/nlp/ai_attribute_extraction.py::_parse_soft_inferences) son
// texto LIBRE, sin lista cerrada, así que siempre hace falta un fallback
// (ver `categoryLabel` más abajo) para cualquier categoría no listada aquí.
const KNOWN_CATEGORY_KEYS: Record<string, string> = {
  ubicacion: 'ubicacion',
  ocupacion: 'ocupacion',
  rutina: 'rutina',
  aficion: 'aficion',
  texto_visible: 'texto_visible',
  matricula: 'matricula',
};

// Fallback para categorías libres devueltas por la IA (p. ej.
// "mascota_visible"): "mascota_visible" -> "Mascota visible". Nunca se
// intenta adivinar significado, solo formatear la clave tal cual llega.
function fallbackCategoryLabel(category: string): string {
  const spaced = category.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function categoryLabel(
  category: string,
  t: (key: string) => string,
  i18nExists: (key: string) => boolean,
): string {
  const key = KNOWN_CATEGORY_KEYS[category];
  const i18nKey = key ? `dashboard.inferredCategory.${key}` : null;
  if (i18nKey && i18nExists(i18nKey)) return t(i18nKey);
  return fallbackCategoryLabel(category);
}

const InferredAttributesList: React.FC<InferredAttributesListProps> = ({ attributes }) => {
  const { t, i18n } = useTranslation();

  if (attributes.length === 0) return null;

  return (
    <div className="inferred-attributes-section">
      <h3>{t('components.inferredAttributes.title')}</h3>
      <p className="note">{t('components.inferredAttributes.subtitle')}</p>

      <div className="inferred-attributes-list">
        {attributes.map((attribute, index) => (
          // eslint-disable-next-line react/no-array-index-key -- no hay id estable, y la categoría se repite entre filas.
          <div className="inferred-attribute-item" key={`${attribute.category}-${index}`}>
            <div className="inferred-attribute-header">
              <span className="inferred-attribute-category">
                {categoryLabel(attribute.category, t, i18n.exists.bind(i18n))}
              </span>
              {attribute.confidence !== null && attribute.confidence !== undefined && (
                <span
                  className="confidence-badge"
                  title={t('components.populationNarrowing.confidenceBadgeTitle')}
                >
                  ~{Math.round(attribute.confidence * 100)}%
                </span>
              )}
            </div>
            <p className="inferred-attribute-value">{attribute.value}</p>
            {attribute.evidence.length > 0 && (
              <p className="note-inline">
                {t('components.inferredAttributes.evidenceCount', { count: attribute.evidence.length })}
              </p>
            )}

            {attribute.category === 'matricula' && (
              <div className="dgt-info-box">
                <p className="dgt-info-box-title">{t('components.inferredAttributes.dgt.title')}</p>

                <p>{t('components.inferredAttributes.dgt.freeIntro')}</p>
                <ul>
                  <li>{t('components.inferredAttributes.dgt.freeItem1')}</li>
                  <li>{t('components.inferredAttributes.dgt.freeItem2')}</li>
                  <li>{t('components.inferredAttributes.dgt.freeItem3')}</li>
                  <li>{t('components.inferredAttributes.dgt.freeItem4')}</li>
                  <li>{t('components.inferredAttributes.dgt.freeItem5')}</li>
                </ul>

                <p>{t('components.inferredAttributes.dgt.paidIntro')}</p>
                <ul>
                  <li>{t('components.inferredAttributes.dgt.paidItem1')}</li>
                  <li>{t('components.inferredAttributes.dgt.paidItem2')}</li>
                  <li>{t('components.inferredAttributes.dgt.paidItem3')}</li>
                  <li>{t('components.inferredAttributes.dgt.paidItem4')}</li>
                  <li>{t('components.inferredAttributes.dgt.paidItem5')}</li>
                  <li>{t('components.inferredAttributes.dgt.paidItem6')}</li>
                </ul>

                <p className="note-inline">{t('components.inferredAttributes.dgt.closingNote')}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default InferredAttributesList;
