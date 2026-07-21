import React from 'react';
import type { ExposureReport } from '../types';
import { downloadReportAsJson } from '../utils/reportToJson';

interface DownloadReportButtonProps {
  report: ExposureReport;
}

const DownloadReportButton: React.FC<DownloadReportButtonProps> = ({ report }) => {
  return (
    <button type="button" className="download-report-button" onClick={() => downloadReportAsJson(report)}>
      Descargar informe completo (JSON)
    </button>
  );
};

export default DownloadReportButton;
