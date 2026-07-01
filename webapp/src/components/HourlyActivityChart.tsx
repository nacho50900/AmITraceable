import React from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface HourlyActivityChartProps {
  hourlyData: Record<string, number>;
}

interface ChartPoint {
  hour: string;
  actividad: number;
}

const HourlyActivityChart: React.FC<HourlyActivityChartProps> = ({ hourlyData }) => {
  const data: ChartPoint[] = Object.entries(hourlyData).map(([hour, ratio]) => ({
    hour: `${hour}h`,
    actividad: Math.round(ratio * 1000) / 10,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data}>
        <XAxis dataKey="hour" interval={1} fontSize={11} />
        <YAxis unit="%" fontSize={11} />
        <Tooltip formatter={(value: number) => [`${value}%`, 'Actividad']} />
        <Bar dataKey="actividad" fill="#5b6cf0" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default HourlyActivityChart;
