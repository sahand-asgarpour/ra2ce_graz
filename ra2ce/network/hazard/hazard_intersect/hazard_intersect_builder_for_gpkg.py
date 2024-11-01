"""
                    GNU GENERAL PUBLIC LICENSE
                      Version 3, 29 June 2007

    Risk Assessment and Adaptation for Critical Infrastructure (RA2CE).
    Copyright (C) 2023 Stichting Deltares

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List
from pathlib import Path

import numpy as np
from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree
from numpy import nanmean
from geopandas import GeoDataFrame, read_file
from networkx import Graph
from ra2ce.network.hazard.hazard_intersect.hazard_intersect_builder_base import (
    HazardIntersectBuilderBase,
)
from tqdm import tqdm


@dataclass
class HazardIntersectBuilderForGpkg(HazardIntersectBuilderBase):
    hazard_field_name: str = ""
    hazard_aggregate_wl: str = ""
    ra2ce_names: List[str] = field(default_factory=list)
    hazard_gpkg_files: List[Path] = field(default_factory=list)

    def _from_networkx(self, hazard_overlay: Graph) -> Graph:
        """Overlays the hazard `gpkg` file over the road segments NetworkX graph.

        Args:
            hazard_overlay (NetworkX graph): The graph that should be overlaid with the hazard `gpkg` file(s).

        Returns:
            hazard_overlay (NetworkX graph): The graph with hazard shapefile(s) data joined
        """

        def networkx_overlay(
            _hazard_overlay: Graph, hazard_shp_file: Path, ra2ce_name: str
        ) -> Graph:
            def validate_and_fix_geometry(geometry):
                """Validate and fix invalid geometry using buffer(0). Return None if invalid."""
                if not geometry.is_valid:
                    try:
                        geometry = geometry.buffer(0)
                    except TopologicalError:
                        return None
                return geometry if geometry.is_valid else None

            def process_edge(_u, _v, _k, _edata):
                """Process an edge to calculate the intersection fraction and hazard value."""
                if "geometry" not in _edata:
                    return 0, 0

                # Validate the edge geometry
                geom = validate_and_fix_geometry(_edata["geometry"])
                if geom is None or geom.length == 0:
                    return 0, 0

                total_length = geom.length

                # Query the spatial index tree for intersecting geometries
                possible_matches_indices = tree.query(geom)

                intersection_length = 0
                intersected_values = []

                for idx in possible_matches_indices:
                    poly = hazard_geoms[idx]

                    # Validate and fix the hazard geometry if needed
                    poly = validate_and_fix_geometry(poly)
                    if poly is None:
                        continue

                    # Convert MultiPolygon to list of polygons for uniform processing
                    polygons_to_check = (
                        poly.geoms if isinstance(poly, MultiPolygon) else [poly]
                    )

                    # Calculate intersection length and hazard values
                    for poly_part in polygons_to_check:
                        if geom.intersects(poly_part):
                            intersection_length += geom.intersection(poly_part).length

                            hazard_value = gdf_hazard_exploded.iloc[idx][
                                self.hazard_field_name
                            ]
                            if hazard_value != 0:
                                intersected_values.append(hazard_value)

                if intersection_length == 0:
                    return 0, 0

                _intersection_fraction = intersection_length / total_length

                # Determine hazard value using the specified aggregation method
                if intersected_values:
                    _hazard_value = {"max": max, "min": min, "mean": np.nanmean}.get(
                        self.hazard_aggregate_wl, np.nanmean
                    )(intersected_values)
                else:
                    _hazard_value = 0

                return _intersection_fraction, _hazard_value

            # Load and transform hazard data if necessary
            gdf_hazard = read_file(str(hazard_shp_file))
            if _hazard_overlay.graph["crs"] != gdf_hazard.crs:
                gdf_hazard = gdf_hazard.to_crs(_hazard_overlay.graph["crs"])

            # Explode MultiPolygon geometries for individual processing
            logging.info("Converting MultiPolygon to Polygon")
            gdf_hazard_exploded = self._explode_multigeometries(gdf_hazard)
            hazard_geoms = gdf_hazard_exploded.geometry.values
            tree = STRtree(hazard_geoms)

            logging.info("Processing edges for hazard overlay")
            total_edges = _hazard_overlay.number_of_edges()

            # Processing edges with in-place updates
            for u, v, k, edata in tqdm(
                _hazard_overlay.edges(data=True, keys=True),
                total=total_edges,
                desc="Processing Edges",
                unit="edge",
            ):
                if "geometry" not in edata:
                    continue  # Skip edges without geometry attribute

                # Calculate intersection fraction and hazard value using the process_edge function
                intersection_fraction, hazard_value = process_edge(u, v, k, edata)

                # Update edge data directly
                edata[f"{ra2ce_name}_{self.hazard_aggregate_wl[:2]}"] = hazard_value
                edata[f"{ra2ce_name}_fr"] = intersection_fraction

            return _hazard_overlay
            # gdf_hazard = read_file(str(hazard_shp_file))
            #
            # if _hazard_overlay.graph["crs"] != gdf_hazard.crs:
            #     gdf_hazard = gdf_hazard.to_crs(_hazard_overlay.graph["crs"])
            #
            # # Convert MultiPolygon to Polygon
            # logging.info("Converting MultiPolygon to Polygon")
            # gdf_hazard_exploded = self._explode_multigeometries(gdf_hazard)
            #
            # tree = STRtree(gdf_hazard_exploded["geometry"].tolist())
            # hazard_geoms = gdf_hazard_exploded.geometry.values
            #
            # logging.info("Processing edges for hazard overlay")
            #
            # # Get the total number of edges for progress tracking
            # total_edges = _hazard_overlay.number_of_edges()
            #
            # # Process edges in parallel and track progress
            # with ThreadPoolExecutor() as executor:
            #     # Wrap the edges iterator in tqdm for progress tracking
            #     edges = list(_hazard_overlay.edges(data=True, keys=True))
            #
            #     # Process edges in parallel while showing the tqdm progress bar
            #     results = list(
            #         tqdm(
            #             executor.map(lambda edge: process_edge(*edge), edges),
            #             total=total_edges,
            #             desc="Processing Edges",
            #             unit="edge"
            #         )
            #     )
            #
            # for (u, v, k, edata), (intersection_fraction, hazard_value) in zip(
            #     _hazard_overlay.edges(data=True, keys=True), results
            # ):
            #     edata[ra2ce_name + "_" + self.hazard_aggregate_wl[:2]] = hazard_value
            #     edata[ra2ce_name + "_" + "fr"] = intersection_fraction
            #
            # return _hazard_overlay

        hazard_overlay = self._overlay_hazard_files(
            overlay_func=networkx_overlay, hazard_overlay=hazard_overlay
        )

        return hazard_overlay

    def _from_geodataframe(self, hazard_overlay: GeoDataFrame) -> GeoDataFrame:
        """Overlays the hazard shapefile over the road segments GeoDataFrame."""

        gdf_crs_original = hazard_overlay.crs

        def geodataframe_overlay(
            _hazard_overlay: GeoDataFrame, hazard_shp_file: Path, ra2ce_name: str
        ) -> GeoDataFrame:
            def calculate_intersection_fraction(line):
                total_length = line.length
                if total_length == 0:
                    return 0, []

                possible_matches_indices = list(tree.query(line))
                intersection_length = sum(
                    line.intersection(hazard_geoms[idx]).length
                    for idx in possible_matches_indices
                    if line.intersects(hazard_geoms[idx])
                )
                intersection_fraction = intersection_length / total_length

                _intersected_values = [
                    gdf_hazard_exploded.iloc[idx][self.hazard_field_name]
                    for idx in possible_matches_indices
                    if line.intersects(hazard_geoms[idx])
                    and gdf_hazard_exploded.iloc[idx][self.hazard_field_name] != 0
                ]

                return intersection_fraction, _intersected_values

            # Load hazard data
            gdf_hazard = read_file(str(hazard_shp_file))

            # Align coordinate reference systems if different
            if _hazard_overlay.crs != gdf_hazard.crs:
                _hazard_overlay = _hazard_overlay.to_crs(gdf_hazard.crs)

            # Explode MultiPolygons to individual Polygons for efficient processing
            logging.info("Converting MultiPolygon to Polygon")
            gdf_hazard_exploded = self._explode_multigeometries(gdf_hazard)
            hazard_geoms = gdf_hazard_exploded["geometry"].values
            tree = STRtree(hazard_geoms)

            # Prepare columns for results in _hazard_overlay
            _hazard_overlay[f"{ra2ce_name}_fr"] = 0  # Intersection fraction
            _hazard_overlay[
                f"{ra2ce_name}_{self.hazard_aggregate_wl[:2]}"
            ] = 0  # Hazard value

            logging.info("Processing geometries for hazard overlay")

            # Iterate over each geometry in _hazard_overlay and calculate intersections
            for idx, geom in tqdm(
                enumerate(_hazard_overlay["geometry"]),
                total=len(_hazard_overlay),
                desc="Processing Geometries",
                unit="geom",
            ):
                if geom is None or geom.is_empty:
                    continue  # Skip if geometry is invalid or empty

                # Query STRtree for intersecting hazard geometries
                intersecting_indices = tree.query(geom)

                # Initialize tracking of intersection length and hazard values
                total_length = geom.length
                intersection_length = 0
                intersected_values = []

                for hazard_idx in intersecting_indices:
                    hazard_geom = hazard_geoms[hazard_idx]
                    if not geom.intersects(hazard_geom):
                        continue  # Skip non-intersecting geometries

                    # Calculate intersection length
                    intersection = geom.intersection(hazard_geom)
                    intersection_length += (
                        intersection.length
                        if intersection.is_empty
                        else intersection.length
                    )

                    # Store hazard value for current intersecting geometry
                    hazard_value = gdf_hazard_exploded.iloc[hazard_idx][
                        self.hazard_field_name
                    ]
                    if hazard_value != 0:
                        intersected_values.append(hazard_value)

                # Calculate intersection fraction
                intersection_fraction = (
                    intersection_length / total_length if total_length > 0 else 0
                )
                _hazard_overlay.at[idx, f"{ra2ce_name}_fr"] = intersection_fraction

                # Set hazard value based on aggregation method
                if intersected_values:
                    _hazard_overlay.at[
                        idx, f"{ra2ce_name}_{self.hazard_aggregate_wl[:2]}"
                    ] = {"max": max, "min": min, "mean": np.nanmean}.get(
                        self.hazard_aggregate_wl, np.nanmean
                    )(
                        intersected_values
                    )

            return _hazard_overlay

            # gdf_hazard = read_file(str(hazard_shp_file))
            #
            # if _hazard_overlay.crs != gdf_hazard.crs:
            #     _hazard_overlay = _hazard_overlay.to_crs(gdf_hazard.crs)
            #
            # # Convert MultiPolygon to Polygon
            # logging.info("Converting MultiPolygon to Polygon")
            # gdf_hazard_exploded = self._explode_multigeometries(gdf_hazard)
            #
            # logging.info("Creating STRee")
            # tree = STRtree(gdf_hazard_exploded["geometry"].tolist())
            #
            # intersected_fractions = []
            # hazard_values_list = []
            # hazard_geoms = gdf_hazard_exploded["geometry"].values
            #
            #
            #
            # # Using ThreadPoolExecutor for parallel processing
            # from concurrent.futures import ThreadPoolExecutor
            #
            # logging.info("Calculating intersection fractions for hazard overlay")
            #
            # # Wrap the geometries with tqdm for progress tracking
            # geometries = _hazard_overlay["geometry"]
            #
            # with ThreadPoolExecutor() as executor:
            #     # Use tqdm to track the progress
            #     results = list(
            #         tqdm(
            #             executor.map(calculate_intersection_fraction, geometries),
            #             total=len(geometries),  # Total number of geometries to process
            #             desc="Processing geometries",  # Description for the progress bar
            #         )
            #     )
            #
            # logging.info("Calculating hazard values")
            # for fraction, intersected_values in results:
            #     intersected_fractions.append(fraction)
            #
            #     if intersected_values:
            #         if self.hazard_aggregate_wl == "max":
            #             hazard_value = max(intersected_values)
            #         elif self.hazard_aggregate_wl == "min":
            #             hazard_value = min(intersected_values)
            #         elif self.hazard_aggregate_wl == "mean":
            #             hazard_value = nanmean(intersected_values)
            #     else:
            #         hazard_value = 0
            #
            #     hazard_values_list.append(hazard_value)
            #
            # _hazard_overlay[ra2ce_name + "_" + "fr"] = intersected_fractions
            # _hazard_overlay[
            #     ra2ce_name + "_" + self.hazard_aggregate_wl[:2]
            # ] = hazard_values_list
            #
            # return _hazard_overlay

        hazard_overlay_performed = self._overlay_hazard_files(
            hazard_overlay=hazard_overlay, overlay_func=geodataframe_overlay
        )

        if hazard_overlay_performed.crs != gdf_crs_original:
            hazard_overlay_performed = hazard_overlay_performed.to_crs(gdf_crs_original)

        return hazard_overlay_performed

    def _overlay_hazard_files(
        self,
        overlay_func: Callable[[GeoDataFrame | Graph, Path, str], GeoDataFrame | Graph],
        hazard_overlay: GeoDataFrame | Graph,
    ) -> GeoDataFrame:
        for i, ra2ce_name in enumerate(self.ra2ce_names):
            hazard_overlay = overlay_func(
                _hazard_overlay=hazard_overlay,
                hazard_shp_file=self.hazard_gpkg_files[i],
                ra2ce_name=ra2ce_name,
            )
        return hazard_overlay

    def _explode_multigeometries(self, gdf: GeoDataFrame) -> GeoDataFrame:
        """Convert MultiPolygon geometries in a GeoDataFrame to individual Polygon geometries.

        Args:
            gdf (GeoDataFrame): The input GeoDataFrame with MultiPolygon geometries.

        Returns:
            GeoDataFrame: A new GeoDataFrame with individual Polygon geometries.
        """

        def explode_geometry(_row):
            if isinstance(_row.geometry, MultiPolygon):
                return [Polygon(poly) for poly in _row.geometry.geoms]
            else:
                return [_row.geometry]

        # Explode MultiPolygon geometries
        exploded_geometries = []
        exploded_attributes = []
        for _, row in gdf.iterrows():
            exploded_geom = explode_geometry(row)
            for geom in exploded_geom:
                exploded_geometries.append(geom)
                exploded_attributes.append(row.drop("geometry"))

        exploded_gdf = GeoDataFrame(
            exploded_attributes, geometry=exploded_geometries, crs=gdf.crs
        )
        exploded_gdf["polygon_id"] = range(1, len(exploded_gdf) + 1)

        return exploded_gdf
