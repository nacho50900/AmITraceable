import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { downloadReportAsJson } from '../utils/reportToJson';
import { makeExposureReport } from './fixtures';

describe('downloadReportAsJson', () => {
  let createObjectURLSpy: ReturnType<typeof vi.fn>;
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    createObjectURLSpy = vi.fn(() => 'blob:mock-url');
    revokeObjectURLSpy = vi.fn();
    // jsdom no implementa URL.createObjectURL/revokeObjectURL de forma nativa.
    URL.createObjectURL = createObjectURLSpy as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = revokeObjectURLSpy;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('genera un Blob de tipo application/json con el informe completo, sin pérdida de datos', () => {
    const report = makeExposureReport();
    let capturedParts: BlobPart[] | undefined;
    let capturedOptions: BlobPropertyBag | undefined;
    const OriginalBlob = globalThis.Blob;

    // OJO: el mock debe ser una `function` normal, no una arrow function --
    // las arrow functions no son invocables con `new` (fallaría con
    // "is not a constructor" en cuanto reportToJson.ts hiciera `new Blob(...)`).
    vi.stubGlobal(
      'Blob',
      vi.fn(function (parts: BlobPart[], options?: BlobPropertyBag) {
        capturedParts = parts;
        capturedOptions = options;
        return new OriginalBlob(parts, options);
      }),
    );
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    try {
      downloadReportAsJson(report);
      expect(capturedOptions?.type).toBe('application/json');
      expect(capturedParts).toHaveLength(1);
      expect(JSON.parse(capturedParts![0] as string)).toEqual(report);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test('nombra el fichero descargado con plataforma, usuario y fecha (YYYY-MM-DD)', () => {
    const report = makeExposureReport({
      platform: 'reddit',
      username: 'pepito',
      generated_at: '2026-03-15T08:30:00+00:00',
    });
    let capturedDownloadName = '';
    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = realCreateElement(tag);
      if (tag === 'a') {
        vi.spyOn(el as HTMLAnchorElement, 'click').mockImplementation(() => {
          capturedDownloadName = (el as HTMLAnchorElement).download;
        });
      }
      return el;
    });

    downloadReportAsJson(report);

    expect(capturedDownloadName).toBe('informe_exposicion_reddit_pepito_2026-03-15.json');
  });

  test('revoca la URL del objeto tras disparar la descarga (no deja fugas de memoria)', () => {
    const report = makeExposureReport();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    downloadReportAsJson(report);

    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url');
  });

  test('no deja el elemento <a> temporal colgado en el DOM tras la descarga', () => {
    const report = makeExposureReport();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    downloadReportAsJson(report);

    expect(document.querySelectorAll('a[download]').length).toBe(0);
  });
});
