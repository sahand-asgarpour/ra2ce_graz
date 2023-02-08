# -*- coding: utf-8 -*-
"""
Created on 21-9-2022

@author: F.C. de Groen, Deltares
"""

from pathlib import Path

import geopandas as gpd


def filter_shp(shp_path, output_dir, keep_osm_classes):
    gdf = gpd.read_file(shp_path)
    gdf = gdf.loc[gdf["fclass"].isin(keep_osm_classes)]

    feather_path = Path(output_dir) / (Path(shp_path).stem + ".shp")
    gdf.to_file(feather_path, index=False)
    print(f"Saved {Path(shp_path).stem} in {output_dir}.")


if __name__ == "__main__":
    _output_dir = r"p:\pakistan-flood\0_Workflows\1_Workflow_test\static\network"
    _shp_path = r"d:\ra2ceMaster\gis_osm_roads_free_1.shp"
    _keep_osm_classes = [
        "tertiary",
        "secondary",
        "trunk",
        "primary",
        "motorway",
        "motorway_link",
        "secondary_link",
        "primary_link",
        "tertiary_link",
    ]

    filter_shp(_shp_path, _output_dir, _keep_osm_classes)