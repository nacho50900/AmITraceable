import React from 'react';
import { useTranslation } from 'react-i18next';
import type { ExposureReport } from '../types';
import { downloadReportAsJson } from '../utils/reportToJson';

interface DownloadReportButtonProps {
  report: ExposureReport;
}

const DownloadReportButton: React.FC<DownloadReportButtonProps> = ({ report }) => {
  const { t } = useTranslation();
  return (
    <button type="button" className="download-report-button" onClick={() => downloadReportAsJson(report)}>
      {t('components.downloadButton.label')}
    </button>
  );
};

export default DownloadReportButton;
