// Mock auto.ru-style data — two parser dumps of the same marketplace
// with realistic formatting divergence. Some rows are true duplicates,
// some are false near-matches (same brand/model different car).

window.__DATA__ = (function () {
  // -----------------------------------------------------------------
  // TABLE A — "scrape_2023_q4.xlsx" — clean text formatting
  // -----------------------------------------------------------------
  const tableA = {
    id: 'A',
    name: 'auto_ru_2023_q4.xlsx',
    rows: 14,
    cols: ['sell_id', 'mark', 'model', 'year', 'mileage', 'color', 'bodyType', 'engine', 'transmission', 'price'],
    data: [
      ['77-A1042', 'BMW',       'X5',          2018, 85000,  'белый',        'внедорожник', 3.0, 'АКПП', 3520000],
      ['77-A1043', 'Toyota',    'Camry',       2020, 45000,  'чёрный',       'седан',       2.5, 'АКПП', 2780000],
      ['77-A1044', 'Mercedes',  'E-Class',     2019, 60500,  'серебристый',  'седан',       2.0, 'АКПП', 3210000],
      ['77-A1045', 'Audi',      'A6',          2021, 28000,  'серый',        'седан',       2.0, 'АКПП', 3650000],
      ['77-A1046', 'Volkswagen','Tiguan',      2017, 112000, 'красный',      'внедорожник', 1.4, 'МКПП', 1490000],
      ['77-A1047', 'Kia',       'Rio',         2019, 67000,  'синий',        'седан',       1.6, 'МКПП', 890000],
      ['77-A1048', 'Hyundai',   'Solaris',     2018, 89000,  'белый',        'седан',       1.6, 'АКПП', 870000],
      ['77-A1049', 'Lada',      'Vesta',       2020, 42000,  'тёмно-синий',  'седан',       1.6, 'МКПП', 760000],
      ['77-A1050', 'Skoda',     'Octavia',     2019, 71000,  'белый',        'лифтбек',     1.4, 'АКПП', 1340000],
      ['77-A1051', 'Renault',   'Logan',       2017, 134000, 'серебристый',  'седан',       1.6, 'МКПП', 540000],
      ['77-A1052', 'BMW',       'X5',          2020, 41000,  'чёрный',       'внедорожник', 3.0, 'АКПП', 4890000],
      ['77-A1053', 'Toyota',    'RAV4',        2019, 56000,  'белый',        'внедорожник', 2.0, 'АКПП', 2440000],
      ['77-A1054', 'Mazda',     'CX-5',        2018, 78000,  'красный',      'внедорожник', 2.0, 'АКПП', 1980000],
      ['77-A1055', 'Mitsubishi','Outlander',   2017, 98000,  'серый',        'внедорожник', 2.4, 'АКПП', 1620000],
    ],
  };

  // -----------------------------------------------------------------
  // TABLE B — "scrape_2024_q1.xlsx" — different parser:
  //   * different column names (brand, model_name, probeg, hex colors)
  //   * UPPERCASE bodyType
  //   * decimal price in thousands
  //   * different sell_id format
  // -----------------------------------------------------------------
  const tableB = {
    id: 'B',
    name: 'auto_ru_2024_q1.xlsx',
    rows: 13,
    cols: ['offer_id', 'brand', 'model_name', 'year', 'probeg_km', 'color_hex', 'body_type', 'engine_cc', 'gearbox', 'price_k'],
    data: [
      // — duplicates of A —
      ['o-7745021', 'BMW',        'X5',         2018, 85120,  '#FFFFFF', 'ВНЕДОРОЖНИК', 2998, 'AT',  3500],
      ['o-7745022', 'TOYOTA',     'CAMRY',      2020, 45000,  '#1A1A1A', 'СЕДАН',       2487, 'AT',  2800],
      ['o-7745023', 'Mercedes-Benz','E 200',    2019, 60800,  '#C0C0C0', 'СЕДАН',       1991, 'AT',  3250],
      ['o-7745024', 'AUDI',       'A6',         2021, 27500,  '#808080', 'СЕДАН',       1984, 'AT',  3700],
      ['o-7745025', 'Hyundai',    'Solaris',    2018, 89400,  '#F8F8F8', 'СЕДАН',       1591, 'AT',  860],
      ['o-7745026', 'Skoda',      'Octavia',    2019, 71200,  '#FAFAFA', 'ЛИФТБЕК',     1395, 'AT',  1320],
      ['o-7745027', 'TOYOTA',     'RAV-4',      2019, 56400,  '#FFFFFF', 'ВНЕДОРОЖНИК', 1987, 'AT',  2480],
      ['o-7745028', 'Mazda',      'CX-5',       2018, 78000,  '#B71C1C', 'ВНЕДОРОЖНИК', 1998, 'AT',  1950],
      // — new offers not in A —
      ['o-7745029', 'Ford',       'Focus',      2016, 145000, '#1565C0', 'СЕДАН',       1596, 'MT',  680],
      ['o-7745030', 'Nissan',     'Qashqai',    2019, 64000,  '#FFFFFF', 'ВНЕДОРОЖНИК', 1997, 'AT',  1780],
      ['o-7745031', 'BMW',        'X3',         2019, 58000,  '#1A1A1A', 'ВНЕДОРОЖНИК', 1998, 'AT',  3120],
      // — tricky near-match: same brand/model as A but different car (different year & mileage)
      ['o-7745032', 'BMW',        'X5',         2022, 22000,  '#0D47A1', 'ВНЕДОРОЖНИК', 2998, 'AT',  6450],
      // — Lada Vesta from A
      ['o-7745033', 'LADA',       'Vesta',      2020, 42300,  '#0D47A1', 'СЕДАН',       1596, 'MT',  755],
    ],
  };

  // -----------------------------------------------------------------
  // GROUND-TRUTH clusters (computed for the prototype)
  //   index format: [tableId, rowIdx]
  // -----------------------------------------------------------------
  const clusters = [
    { id: 'C-001', label: 'BMW X5 2018 (белый, 85k km)', members: [['A', 0], ['B', 0]], sim: 0.94 },
    { id: 'C-002', label: 'Toyota Camry 2020 (чёрный)',  members: [['A', 1], ['B', 1]], sim: 0.92 },
    { id: 'C-003', label: 'Mercedes E-Class 2019',       members: [['A', 2], ['B', 2]], sim: 0.88 },
    { id: 'C-004', label: 'Audi A6 2021 (серый)',        members: [['A', 3], ['B', 3]], sim: 0.91 },
    { id: 'C-005', label: 'Hyundai Solaris 2018',        members: [['A', 6], ['B', 4]], sim: 0.89 },
    { id: 'C-006', label: 'Skoda Octavia 2019 (белая)',  members: [['A', 8], ['B', 5]], sim: 0.93 },
    { id: 'C-007', label: 'Toyota RAV4 2019 (белый)',    members: [['A', 11], ['B', 6]], sim: 0.90 },
    { id: 'C-008', label: 'Mazda CX-5 2018 (красная)',   members: [['A', 12], ['B', 7]], sim: 0.91 },
    { id: 'C-009', label: 'Lada Vesta 2020',             members: [['A', 7], ['B', 12]], sim: 0.86, needsReview: true },
    // singletons (unique offers)
    { id: 'C-010', label: 'VW Tiguan 2017',              members: [['A', 4]], sim: 1.0 },
    { id: 'C-011', label: 'Kia Rio 2019',                members: [['A', 5]], sim: 1.0 },
    { id: 'C-012', label: 'Renault Logan 2017',          members: [['A', 9]], sim: 1.0 },
    { id: 'C-013', label: 'BMW X5 2020 (чёрный)',        members: [['A', 10]], sim: 1.0 },
    { id: 'C-014', label: 'Mitsubishi Outlander 2017',   members: [['A', 13]], sim: 1.0 },
    { id: 'C-015', label: 'Ford Focus 2016',             members: [['B', 8]], sim: 1.0 },
    { id: 'C-016', label: 'Nissan Qashqai 2019',         members: [['B', 9]], sim: 1.0 },
    { id: 'C-017', label: 'BMW X3 2019',                 members: [['B', 10]], sim: 1.0 },
    { id: 'C-018', label: 'BMW X5 2022 (синий)',         members: [['B', 11]], sim: 1.0 },
  ];

  // Candidate pairs surfaced by GNN — includes some uncertain ones
  const candidates = [
    { a: ['A', 0],  b: ['B', 0],  sim: 0.94, verdict: 'auto', cluster: 'C-001' },
    { a: ['A', 1],  b: ['B', 1],  sim: 0.92, verdict: 'auto', cluster: 'C-002' },
    { a: ['A', 8],  b: ['B', 5],  sim: 0.93, verdict: 'auto', cluster: 'C-006' },
    { a: ['A', 11], b: ['B', 6],  sim: 0.90, verdict: 'auto', cluster: 'C-007' },
    { a: ['A', 12], b: ['B', 7],  sim: 0.91, verdict: 'auto', cluster: 'C-008' },
    { a: ['A', 3],  b: ['B', 3],  sim: 0.91, verdict: 'auto', cluster: 'C-004' },
    { a: ['A', 6],  b: ['B', 4],  sim: 0.89, verdict: 'auto', cluster: 'C-005' },
    { a: ['A', 2],  b: ['B', 2],  sim: 0.88, verdict: 'review', cluster: 'C-003' },
    { a: ['A', 7],  b: ['B', 12], sim: 0.86, verdict: 'review', cluster: 'C-009' },
    // tricky pair — same model BMW X5 but different car
    { a: ['A', 0],  b: ['B', 11], sim: 0.74, verdict: 'review', cluster: null },
    { a: ['A', 10], b: ['B', 0],  sim: 0.71, verdict: 'review', cluster: null },
    { a: ['A', 10], b: ['B', 11], sim: 0.68, verdict: 'reject', cluster: null },
  ];

  return { tableA, tableB, clusters, candidates };
})();
