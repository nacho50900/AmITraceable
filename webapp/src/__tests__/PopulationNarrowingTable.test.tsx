import '@testing-library/jest-dom';
import { act, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, test } from 'vitest';
import i18n from '../i18n';
import PopulationNarrowingTable from '../components/PopulationNarrowingTable';
import type { PopulationEstimate } from '../types';

function makeStep(overrides: Partial<PopulationEstimate> = {}): PopulationEstimate {
  return {
    attribute_label: 'Sexo: mujer',
    category: 'sexo',
    remaining_population: 24_957_175,
    risk_level: 'bajo',
    evidence: ['https://x/1'],
    source: 'texto',
    note: null,
    proportion: 24_957_175 / 49_128_297,
    reduction_percent: 49.2,
    ...overrides,
  };
}

/** Envoltorio con las dos props del resumen combinado a null por defecto,
 * para no tener que repetirlas en cada test que no las necesita. */
function renderTable(
  steps: PopulationEstimate[],
  combined: { remainingPopulationAllTraits?: number | null; remainingPopulationAllTraitsProportion?: number | null } = {},
) {
  return render(
    <PopulationNarrowingTable
      steps={steps}
      remainingPopulationAllTraits={combined.remainingPopulationAllTraits ?? null}
      remainingPopulationAllTraitsProportion={combined.remainingPopulationAllTraitsProportion ?? null}
    />,
  );
}

describe('PopulationNarrowingTable', () => {
  test('lista vacía: muestra el mensaje explicativo, sin bloques de atributo', () => {
    const { container } = renderTable([]);

    expect(screen.getByText(/No se han detectado declaraciones explícitas/)).toBeInTheDocument();
    expect(container.querySelectorAll('.population-step')).toHaveLength(0);
  });

  test('renderiza un bloque por cada paso, con su etiqueta', () => {
    const steps = [makeStep({ attribute_label: 'Sexo: mujer' }), makeStep({ attribute_label: 'Edad: 24 años' })];
    const { container } = renderTable(steps);

    expect(container.querySelectorAll('.population-step')).toHaveLength(2);
    expect(screen.getByText('Sexo: mujer')).toBeInTheDocument();
    expect(screen.getByText('Edad: 24 años')).toBeInTheDocument();
  });

  test('cada fila muestra el número de población formateado, SIN pictograma propio (solo hay uno, el combinado)', () => {
    const { container } = renderTable([makeStep({ remaining_population: 24957175, proportion: 0.5 })]);

    expect(screen.getByText('24.957.175')).toBeInTheDocument();
    // Ni la clase de pictograma grande ni la de compacto deben aparecer en una fila individual.
    expect(container.querySelector('.population-step .pictogram')).not.toBeInTheDocument();
  });

  test('población no estimable (sin población) se muestra como guion largo', () => {
    renderTable([makeStep({ remaining_population: null, proportion: null })]);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  test.each([
    ['bajo', 'Bajo'],
    ['medio', 'Medio'],
    ['alto', 'Alto'],
    ['critico', 'Crítico'],
    ['no_estimable', 'No estimable'],
  ])('nivel de riesgo %s se etiqueta como "%s"', (risk_level, label) => {
    renderTable([makeStep({ risk_level: risk_level as PopulationEstimate['risk_level'] })]);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  test('muestra la nota inline cuando el paso tiene una', () => {
    renderTable([makeStep({ note: 'Nota explicativa de ejemplo' })]);
    expect(screen.getByText('Nota explicativa de ejemplo')).toBeInTheDocument();
  });

  test('no muestra nota inline cuando el paso no la tiene', () => {
    const { container } = renderTable([makeStep({ note: null })]);
    expect(container.querySelector('.note-inline')).not.toBeInTheDocument();
  });

  test('fuente "texto" muestra el icono y etiqueta correctos', () => {
    renderTable([makeStep({ source: 'texto' })]);
    expect(screen.getByText(/✍️ Texto/)).toBeInTheDocument();
  });

  test('fuente "imagen" muestra el icono y etiqueta correctos', () => {
    renderTable([makeStep({ source: 'imagen' })]);
    expect(screen.getByText(/📷 Imagen/)).toBeInTheDocument();
  });

  test('fuente "ia" muestra el icono y etiqueta correctos', () => {
    renderTable([makeStep({ source: 'ia' })]);
    expect(screen.getByText(/🤖 IA \(texto\)/)).toBeInTheDocument();
  });

  test('fuente "ia_nombre" muestra el icono y etiqueta correctos, y añade la advertencia de fiabilidad', () => {
    renderTable([makeStep({ source: 'ia_nombre' })]);
    expect(screen.getByText(/🤖 IA \(nombre\)/)).toBeInTheDocument();
    expect(screen.getByText(/estimación por el nombre público/)).toBeInTheDocument();
  });

  test('fuente "ia_simbolica" muestra el icono y etiqueta correctos', () => {
    renderTable([makeStep({ source: 'ia_simbolica', category: 'estado_civil', attribute_label: 'Casado/a' })]);
    expect(screen.getByText(/🤖 IA \(simbólico\)/)).toBeInTheDocument();
  });

  test('fuente "manual" (rasgo físico autodeclarado, ver ADR-34) muestra el icono, etiqueta y clase distintiva', () => {
    const { container } = renderTable([
      makeStep({ source: 'manual', category: 'color_ojos', value_raw: 'azul', attribute_label: 'Color de ojos: Azul' }),
    ]);
    expect(screen.getByText(/👤 Manual/)).toBeInTheDocument();
    expect(container.querySelector('.source-badge--manual')).toBeInTheDocument();
  });

  test('fuente "manual": el title del badge explica que es autodeclaración, no inferencia automática', () => {
    renderTable([makeStep({ source: 'manual', category: 'color_ojos', value_raw: 'azul' })]);
    expect(screen.getByTitle(/autodeclaración/)).toBeInTheDocument();
  });

  test('sin filas de fuente "imagen": la nota final NO menciona las fotos', () => {
    renderTable([makeStep({ source: 'texto' })]);
    expect(screen.queryByText(/análisis visual de tus fotos/)).not.toBeInTheDocument();
  });

  test('con al menos una fila de fuente "imagen": añade la advertencia de fiabilidad', () => {
    renderTable([makeStep({ source: 'imagen' })]);
    expect(screen.getByText(/análisis visual de tus fotos/)).toBeInTheDocument();
  });

  test('siempre muestra la nota general de estimación aproximada del INE', () => {
    renderTable([makeStep()]);
    expect(screen.getByText(/distribuciones agregadas del INE/)).toBeInTheDocument();
  });

  test('reduction_percent presente: muestra el badge con el porcentaje de reducción', () => {
    renderTable([makeStep({ reduction_percent: 49.2 })]);
    expect(screen.getByText('-49.2%')).toBeInTheDocument();
  });

  test('reduction_percent null (paso no estimable): no muestra el badge de reducción', () => {
    renderTable([makeStep({ reduction_percent: null })]);
    expect(screen.queryByText(/^-.*%$/)).not.toBeInTheDocument();
  });

  test('confidence presente (tramo de edad estimado por IA): muestra el badge de confianza aproximada', () => {
    renderTable([makeStep({ category: 'edad', source: 'ia_estimada', confidence: 0.6 })]);
    expect(screen.getByText('~60%')).toBeInTheDocument();
  });

  test('confidence ausente: no muestra el badge de confianza', () => {
    renderTable([makeStep()]);
    expect(screen.queryByText(/^~\d+%$/)).not.toBeInTheDocument();
  });

  describe('resumen combinado (el único pictograma grande de la sección)', () => {
    test('remainingPopulationAllTraits null: no se muestra ningún resumen combinado', () => {
      const { container } = renderTable([makeStep()], { remainingPopulationAllTraits: null });
      expect(container.querySelector('.population-combined-summary')).not.toBeInTheDocument();
      expect(screen.queryByText(/En España hay/)).not.toBeInTheDocument();
    });

    test('remainingPopulationAllTraits presente: muestra el texto y el pictograma grande', () => {
      const { container } = renderTable([makeStep()], {
        remainingPopulationAllTraits: 1234567,
        remainingPopulationAllTraitsProportion: 0.025,
      });

      expect(screen.getByText(/En España hay/)).toBeInTheDocument();
      expect(screen.getByText('1.234.567')).toBeInTheDocument();
      expect(container.querySelector('.population-combined-summary .pictogram-lg')).toBeInTheDocument();
    });

    test('es el ÚNICO pictograma grande de toda la sección, aunque haya varias filas', () => {
      const steps = [makeStep({ attribute_label: 'Sexo: mujer' }), makeStep({ attribute_label: 'Edad: 24 años' })];
      const { container } = renderTable(steps, {
        remainingPopulationAllTraits: 1234567,
        remainingPopulationAllTraitsProportion: 0.025,
      });

      expect(container.querySelectorAll('.pictogram-lg')).toHaveLength(1);
    });
  });

  describe('traducción de attribute_label/note vía category+value_raw+note_code', () => {
    afterEach(async () => {
      await act(async () => {
        await i18n.changeLanguage('es');
      });
    });

    test('con value_raw: construye el label a partir de la plantilla + valor traducido (conjunto cerrado)', () => {
      renderTable([
        makeStep({
          attribute_label: 'Sexo: mujer',
          category: 'sexo',
          value_raw: 'mujer',
        }),
      ]);

      expect(screen.getByText('Sexo: mujer')).toBeInTheDocument();
    });

    test('ubicación: usa location_level para elegir la plantilla correcta entre las tres', () => {
      renderTable([
        makeStep({
          attribute_label: 'texto viejo que no debería verse',
          category: 'ubicacion',
          location_level: 'comunidad_autonoma',
          value_raw: 'Canarias',
        }),
      ]);

      expect(screen.getByText('Vive en comunidad autónoma: Canarias')).toBeInTheDocument();
    });

    test('nombre propio (universidad): el valor se interpola tal cual, sin buscar traducción', () => {
      renderTable([
        makeStep({
          attribute_label: 'texto viejo',
          category: 'universidad',
          value_raw: 'Universidad De Oviedo',
        }),
      ]);

      expect(screen.getByText('Universidad: Universidad De Oviedo')).toBeInTheDocument();
    });

    test('sin value_raw (backend antiguo): usa attribute_label tal cual, sin romper', () => {
      renderTable([makeStep({ attribute_label: 'Sexo: mujer', category: 'sexo', value_raw: null })]);

      expect(screen.getByText('Sexo: mujer')).toBeInTheDocument();
    });

    test('note_code: traduce la nota sin depender del texto en español de note', () => {
      renderTable([
        makeStep({
          category: 'sexo',
          value_raw: 'hombre',
          note: 'texto viejo en español que no debería verse',
          note_code: 'sexo_estimado_por_nombre',
        }),
      ]);

      expect(
        screen.getByText(/Estimado por convención cultural del nombre público de la cuenta/),
      ).toBeInTheDocument();
    });

    test('note_code desconocido: cae de vuelta a note tal cual', () => {
      renderTable([
        makeStep({
          category: 'sexo',
          value_raw: 'hombre',
          note: 'nota original',
          note_code: 'codigo_que_no_existe',
        }),
      ]);

      expect(screen.getByText('nota original')).toBeInTheDocument();
    });

    test('cambiar el idioma a inglés traduce plantilla y valor de conjunto cerrado', async () => {
      renderTable([
        makeStep({
          attribute_label: 'Sexo: mujer',
          category: 'sexo',
          value_raw: 'mujer',
        }),
      ]);

      await act(async () => {
        await i18n.changeLanguage('en');
      });

      expect(screen.getByText('Sex: female')).toBeInTheDocument();
    });
  });
});
