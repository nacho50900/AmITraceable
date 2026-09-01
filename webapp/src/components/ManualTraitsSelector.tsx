import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FiUserCheck, FiX } from 'react-icons/fi';
import type { ManualAttribute } from '../types';

interface Props {
  onApplyTraits: (traits: ManualAttribute[]) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

// Categorías y valores respaldados por datos reales y citados (ver
// ADR-34 y ADR-35, y app/data/ine_reference.py::EYE_COLOR_DISTRIBUTION /
// HAIR_COLOR_DISTRIBUTION / SKIN_TONE_DISTRIBUTION en el backend --
// Navarro-López et al., Genes 2024, 15(10):1330, DOI 10.3390/genes15101330).
// El backend es la única fuente de verdad de las proporciones; aquí solo
// se listan las claves de categoría/valor para construir el formulario y
// sus etiquetas i18n (dashboard.manualTraits.fields.* /
// dashboard.attributeValue.*), nunca un porcentaje.
const MANUAL_CATEGORIES: Record<string, { fieldKey: string; options: string[] }> = {
  color_ojos: {
    fieldKey: 'eyeColor',
    options: ['marron', 'intermedio', 'azul'],
  },
  color_pelo: {
    fieldKey: 'hairColor',
    options: ['castano', 'negro', 'rubio', 'pelirrojo'],
  },
  color_piel: {
    fieldKey: 'skinColor',
    options: ['claro', 'medio', 'oscuro'],
  },
};

export const ManualTraitsSelector: React.FC<Props> = ({ onApplyTraits, isOpen, setIsOpen }) => {
  const { t } = useTranslation();
  const [selectedTraits, setSelectedTraits] = useState<Record<string, string>>({});

  const handleSelect = (category: string, value: string) => {
    setSelectedTraits((prev) => ({ ...prev, [category]: value }));
  };

  const handleApply = () => {
    const traits: ManualAttribute[] = Object.entries(selectedTraits)
      .filter(([, value]) => value !== '')
      .map(([category, value]) => ({ category, value }));
    onApplyTraits(traits);
    setIsOpen(false);
  };

  const handleClear = () => {
    setSelectedTraits({});
    onApplyTraits([]);
    setIsOpen(false);
  };

  if (!isOpen) {
    return (
      <button type="button" onClick={() => setIsOpen(true)} className="manual-traits-open-button">
        <FiUserCheck aria-hidden="true" />
        {t('dashboard.manualTraits.openButton')}
      </button>
    );
  }

  return (
    <div className="manual-traits-panel">
      <div className="manual-traits-header">
        <h3>
          <FiUserCheck aria-hidden="true" />
          {t('dashboard.manualTraits.panelTitle')}
        </h3>
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="manual-traits-close-button"
          aria-label={t('dashboard.manualTraits.closeAriaLabel')}
        >
          <FiX aria-hidden="true" />
        </button>
      </div>

      <p className="manual-traits-description">{t('dashboard.manualTraits.description')}</p>

      <div className="manual-traits-grid">
        {Object.entries(MANUAL_CATEGORIES).map(([category, data]) => (
          <div key={category} className="manual-traits-field">
            <label htmlFor={category}>{t(`dashboard.manualTraits.fields.${data.fieldKey}`)}</label>
            <select
              id={category}
              className="manual-traits-select"
              value={selectedTraits[category] || ''}
              onChange={(e) => handleSelect(category, e.target.value)}
            >
              <option value="">{t('dashboard.manualTraits.placeholder')}</option>
              {data.options.map((opt) => (
                <option key={opt} value={opt}>
                  {t(`dashboard.attributeValue.${category}.${opt}`)}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <div className="manual-traits-actions">
        <button type="button" onClick={handleClear} className="btn-secondary">
          {t('dashboard.manualTraits.clearButton')}
        </button>
        <button type="button" onClick={handleApply} className="btn-primary">
          {t('dashboard.manualTraits.applyButton')}
        </button>
      </div>
    </div>
  );
};
