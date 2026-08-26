// Prototype survey-area views derived from the supplied orthophoto crops.
// Replace these local demo assets with verified SVAMITVA drone orthophotos before production deployment.
export const VILLAGE_OPTIONS = [
  {
    id: 'area_01',
    name: 'Survey Area 01 — Standard',
    subtitle: 'Typical settlement and mixed land use',
    thumbnail: '/survey/area-01.jpg',
    image: '/survey/area-01.jpg',
    bbox: { x0: 0, x1: 500, y0: 0, y1: 500 },
    quality: 'GOOD',
    qualityScore: 94,
    featureAvailability: { buildings: true, roads: true, water: false },
    description: 'Standard survey scene with a mixed built-up pattern and connected road network.',
  },
  {
    id: 'area_02',
    name: 'Survey Area 02 — Dense',
    subtitle: 'Compact built-up pattern',
    thumbnail: '/survey/area-02.jpg',
    image: '/survey/area-02.jpg',
    bbox: { x0: 500, x1: 1000, y0: 0, y1: 500 },
    quality: 'GOOD',
    qualityScore: 91,
    featureAvailability: { buildings: true, roads: true, water: false },
    description: 'More compact structures and a complex road pattern for review prioritization.',
  },
  {
    id: 'area_03',
    name: 'Survey Area 03 — Mixed',
    subtitle: 'Open land and settlement edge',
    thumbnail: '/survey/area-03.jpg',
    image: '/survey/area-03.jpg',
    bbox: { x0: 0, x1: 500, y0: 500, y1: 1000 },
    quality: 'GOOD',
    qualityScore: 89,
    featureAvailability: { buildings: true, roads: true, water: false },
    description: 'Open terrain and vegetation around a dispersed built-up area; no water feature is asserted in this demo view.',
  },
  {
    id: 'area_04',
    name: 'Survey Area 04 — Challenging',
    subtitle: 'Irregular and high-density area',
    thumbnail: '/survey/area-04.jpg',
    image: '/survey/area-04.jpg',
    bbox: { x0: 500, x1: 1000, y0: 500, y1: 1000 },
    quality: 'WARNING',
    qualityScore: 76,
    featureAvailability: { buildings: true, roads: true, water: false },
    description: 'Complex structures and shadows intended to exercise uncertainty-aware review.',
  },
];

export function getVillageOption(id) {
  return VILLAGE_OPTIONS.find(v => v.id === id) || VILLAGE_OPTIONS[0];
}
