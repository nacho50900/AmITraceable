import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';
import DownloadReportButton from '../components/DownloadReportButton';
import { downloadReportAsJson } from '../utils/reportToJson';
import { makeExposureReport } from './fixtures';

vi.mock('../utils/reportToJson', () => ({
  downloadReportAsJson: vi.fn(),
}));

describe('DownloadReportButton', () => {
  test('muestra el texto del botón de descarga', () => {
    render(<DownloadReportButton report={makeExposureReport()} />);
    expect(screen.getByText('Descargar informe completo (JSON)')).toBeInTheDocument();
  });

  test('al hacer clic, descarga exactamente el informe recibido por props', () => {
    const report = makeExposureReport({ username: 'otro_usuario' });
    render(<DownloadReportButton report={report} />);

    fireEvent.click(screen.getByText('Descargar informe completo (JSON)'));

    expect(downloadReportAsJson).toHaveBeenCalledTimes(1);
    expect(downloadReportAsJson).toHaveBeenCalledWith(report);
  });
});
