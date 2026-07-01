import React from 'react';

interface ScoreBarProps {
  label: string;
  value: number;
}

function riskLabel(value: number): { text: string; color: string } {
  if (value < 25) return { text: 'Bajo', color: '#3aa657' };
  if (value < 55) return { text: 'Medio', color: '#d6a51c' };
  if (value < 80) return { text: 'Alto', color: '#e0792f' };
  return { text: 'Muy alto', color: '#d3403a' };
}

const ScoreBar: React.FC<ScoreBarProps> = ({ label, value }) => {
  const risk = riskLabel(value);

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
