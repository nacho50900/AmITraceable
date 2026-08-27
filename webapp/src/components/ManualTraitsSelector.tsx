import React, { useState } from 'react';
import type { ManualAttribute } from '../types';

interface Props {
  onApplyTraits: (traits: ManualAttribute[]) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const MANUAL_CATEGORIES = {
  color_ojos: {
    label: 'Color de Ojos',
    options: ['marron', 'verde', 'azul', 'miel', 'negro'],
  },
  color_pelo: {
    label: 'Color de Pelo',
    options: ['moreno', 'castano', 'rubio', 'pelirrojo', 'canoso'],
  },
  color_piel: {
    label: 'Tono de Piel',
    options: ['claro', 'medio', 'oscuro'],
  }
};

export const ManualTraitsSelector: React.FC<Props> = ({ onApplyTraits, isOpen, setIsOpen }) => {
  const [selectedTraits, setSelectedTraits] = useState<Record<string, string>>({});

  const handleSelect = (category: string, value: string) => {
    setSelectedTraits(prev => ({ ...prev, [category]: value }));
  };

  const handleApply = () => {
    const traits: ManualAttribute[] = Object.entries(selectedTraits).map(([category, value]) => ({
      category,
      value
    }));
    onApplyTraits(traits);
    setIsOpen(false);
  };

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        className="w-full mb-6 p-4 bg-purple-900/40 hover:bg-purple-800/50 border border-purple-500/30 rounded-xl text-purple-200 transition-all flex items-center justify-center gap-2 font-medium"
      >
        <span className="material-icons">add_reaction</span>
        Añadir rasgos físicos evidentes (Ojos, Pelo, Piel...)
      </button>
    );
  }

  return (
    <div className="mb-6 p-6 bg-slate-800/80 border border-purple-500/50 rounded-xl shadow-xl backdrop-blur-sm animate-fade-in-up">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="material-icons text-purple-400">face</span>
          Añadir Rasgos Físicos Evidentes
        </h3>
        <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white">
          <span className="material-icons">close</span>
        </button>
      </div>
      <p className="text-slate-300 text-sm mb-6">
        Estos datos no pueden inferirse automáticamente por restricciones legales (RGPD), pero si son evidentes en tus fotos o perfil, reducirán significativamente cuántas personas comparten tu exposición.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {Object.entries(MANUAL_CATEGORIES).map(([category, data]) => (
          <div key={category} className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-slate-300 uppercase tracking-wider">{data.label}</label>
            <select
              className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-100 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none transition-all"
              value={selectedTraits[category] || ""}
              onChange={(e) => handleSelect(category, e.target.value)}
            >
              <option value="">Selecciona una opción...</option>
              {data.options.map(opt => (
                <option key={opt} value={opt}>
                  {opt.charAt(0).toUpperCase() + opt.slice(1).replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <div className="flex justify-end gap-3">
        <button
          onClick={() => {
            setSelectedTraits({});
            onApplyTraits([]);
            setIsOpen(false);
          }}
          className="px-4 py-2 rounded-lg text-slate-300 hover:bg-slate-700 transition-colors"
        >
          Limpiar y Cerrar
        </button>
        <button
          onClick={handleApply}
          className="px-6 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors shadow-lg shadow-purple-900/50"
        >
          Aplicar y Recalcular
        </button>
      </div>
    </div>
  );
};
