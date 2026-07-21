import type { ExposureReport } from '../types';

/** Dispara la descarga del informe completo en JSON (sin petición al
 * servidor: el informe ya está en memoria tras el análisis). Formato
 * válido para portabilidad de datos (RGPD Art. 20: estructurado, de uso
 * común, lectura mecánica). */
export function downloadReportAsJson(report: ExposureReport): void {
  const json = JSON.stringify(report, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const dateStr = report.generated_at.slice(0, 10);
  const filename = `informe_exposicion_${report.platform}_${report.username}_${dateStr}.json`;

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
