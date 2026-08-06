import type { Polygon } from "geojson";

export interface OverlayInfo {
  url: string;
  bounds: { west: number; south: number; east: number; north: number };
  width_px: number;
}

export interface ClassArea {
  class_id: number;
  name: string;
  color: string;
  description: string;
  pixel_count: number;
  area_sqm: number;
  area_ha: number;
  area_acres: number;
  percentage: number;
}

export interface ClassifyResult {
  status: string;
  model: { id: string; name: string; type: string; description: string; internal_accuracy: number };
  tile_urls: { sentinel_rgb: string; terrain_classified: string };
  terrain_overlay: OverlayInfo;
  summary: { total_area_ha: number; processing_time_sec: number };
  individual_class_areas: ClassArea[];
  export: { geotiff_url: string };
}

export interface ModelMeta {
  name: string;
  type: string;
  description: string;
  internal_accuracy: number;
}

export interface AppConfig {
  training_schema_version: string;
  classes: Record<string, { name: string; color: string; description: string }>;
  bands: string[];
  seasons: Record<string, { start: string; end: string }>;
  default_season: string;
}

const json = async <T>(res: Response): Promise<T> => {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.json();
};

export const getModels = () =>
  fetch("/api/models").then(json<{ models: Record<string, ModelMeta> }>);

export const getConfig = () => fetch("/api/config").then(json<AppConfig>);

export interface ClassifyParams {
  geometry: Polygon;
  model_type: string;
  start_date: string;
  end_date: string;
  cloud_threshold: number;
  smoothing: boolean;
}

export const classify = (params: ClassifyParams) =>
  fetch("/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then(json<ClassifyResult>);
