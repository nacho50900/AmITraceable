import React from 'react';
import { useTranslation } from 'react-i18next';

interface ScoreBarProps {
  label: string;
  value: number;
}

function riskLabel(value: number, t: (key: string) => string): { text: string; color: string } {
  if (value < 25) return { text: t('components.scoreBar.low'), color: '#3aa657' };
  if (value < 55) return { text: t('components.scoreBar.medium'), color: '#d6a51c' };
  if (value < 80) return { text: t('components.scoreBar.high'), color: '#e0792f' };
  return { text: t('components.scoreBar.veryHigh'), color: '#d3403a' };
}

const ScoreBar: React.FC<ScoreBarProps> = ({ label, value }) => {
  const { t } = useTranslation();
  const risk = riskLabel(value, t);

  return (
    <div className="score-row">
      <div className="score-row-header">
        <span>{label}</span>
        <span style={{ color: risk.color, fontWeight: 600 }}>
          {risk.text} ({value.toFixed(1)})
        </span>
      </div>
      <div className="score-track">
        <div
          className="score-fill"
          style={{ width: `${Math.min(value, 100)}%`, background: risk.color }}
        />
      </div>
    </div>
  );
};

export default ScoreBar;
