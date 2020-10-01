# create graph from OSM
import os as os
import geopandas as gpd
from shapely.wkb import loads
import ogr
import numpy as np
import networkx as nx
import osmnx
import fiona
from shapely.geometry import shape
import geopandas as gpd
from pathlib import Path
import os
import sys
from numpy import object as np_object
import time
import filecmp

### Overrule the OSMNX default settings to get the additional metadata such as street lighting (lit)
osmnx.config(log_console=True, use_cache=True, useful_tags_path = osmnx.settings.useful_tags_path + ['lit'])
sys.setrecursionlimit(10**5)

def fetch_roads(region, osm_pbf_path, **kwargs):
    """
    Function to extract all roads from OpenStreetMap for the specified region.

    Arguments:
        *osm_data* (string) -- string of data path where the OSM extracts (.osm.pbf) are located.
        *region* (string) -- NUTS3 code of region to consider.

        *log_file* (string) OPTIONAL -- string of data path where the log details should be written to

    Returns:
        *Geodataframe* -- Geopandas dataframe with all roads in the specified **region**.

    """

    ## LOAD FILE

    driver = ogr.GetDriverByName('OSM')
    data = driver.Open(osm_pbf_path)

    ## PERFORM SQL QUERY
    sql_lyr = data.ExecuteSQL("SELECT osm_id,highway,other_tags FROM lines WHERE highway IS NOT NULL")

    log_file = kwargs.get('log_file', None)  # if no log_file is provided when calling the function, no log will be made

    if log_file is not None:  # write to log file
        file = open(log_file, mode="a")
        file.write("\n\nRunning fetch_roads for region: {} at time: {}\n".format(region, time.strftime(
            "%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
        file.close()

    ## EXTRACT ROADS
    roads = []
    for feature in sql_lyr:  # Loop over all highway features
        if feature.GetField('highway') is not None:
            osm_id = feature.GetField('osm_id')
            shapely_geo = loads(feature.geometry().ExportToWkb())  # changed on 14/10/2019
            if shapely_geo is None:
                continue
            highway = feature.GetField('highway')
            try:
                other_tags = feature.GetField('other_tags')
                dct = OSM_dict_from_other_tags(other_tags)  # convert the other_tags string to a dict

                if 'lanes' in dct:  # other metadata can be drawn similarly
                    try:
                        # lanes = int(dct['lanes'])
                        lanes = int(round(float(dct['lanes']), 0))
                        # Cannot directly convert a float that is saved as a string to an integer;
                        # therefore: first integer to float; then road float, then float to integer
                    except:
                        if log_file is not None:  # write to log file
                            file = open(log_file, mode="a")
                            file.write(
                                "\nConverting # lanes to integer did not work for region: {} OSM ID: {} with other tags: {}".format(
                                    region, osm_id, other_tags))
                            file.close()
                        lanes = np.NaN  # added on 20/11/2019 to fix problem with UKH35
                else:
                    lanes = np.NaN

                if 'bridge' in dct:  # other metadata can be drawn similarly
                    bridge = dct['bridge']
                else:
                    bridge = np.NaN

                if 'lit' in dct:
                    lit = dct['lit']
                else:
                    lit = np.NaN

            except Exception as e:
                if log_file is not None:  # write to log file
                    file = open(log_file, mode="a")
                    file.write(
                        "\nException occured when reading metadata from 'other_tags', region: {}  OSM ID: {}, Exception = {}\n".format(
                            region, osm_id, e))
                    file.close()
                lanes = np.NaN
                bridge = np.NaN
                lit = np.NaN

            # roads.append([osm_id,highway,shapely_geo,lanes,bridge,other_tags]) #include other_tags to see all available metata
            roads.append([osm_id, highway, shapely_geo, lanes, bridge,
                          lit])  # ... but better just don't: it could give extra errors...
    ## SAVE TO GEODATAFRAME
    if len(roads) > 0:
        return gpd.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry','lanes','bridge','lit'],crs={'init': 'epsg:4326'})
    else:
        print('No roads in {}'.format(region))
        if log_file is not None:
            file = open(log_file, mode="a")
            file.write('No roads in {}'.format(region))
            file.close()



def graph_from_osm(osm_files, multidirectional=False):
    """
    Takes in a list of osmnx compatible files as strings, creates individual graph from each file then combines all
    graphs using the compose_all function from networkx. Most suited for cases where each file represents part of the
    same greater network.

    Arguments:
        list_of_osm_files [list or str]: list of osm xml filenames as strings, see osmnx documentation for compatible file
        formats
        multidirectional [bool]: if True, function returns a directional graph, if false, function returns an
        undirected graph

    Returns:
        G []: A networkx ... or ... instance

    From Kees van Ginkel
    """
    sys.setrecursionlimit(10**5)

    graph_list = []

    if isinstance(osm_files, str):
        G = osmnx.graph_from_file(osm_files, simplify=True)
    else:
        for osm_file in osm_files:
            graph_list.append(osmnx.graph_from_file(osm_file, simplify=True))

        G = nx.compose_all(graph_list)

    if not multidirectional:
        G = G.to_undirected()

    return G


def convert_osm(osm_convert_path, pbf, o5m):
    """
    Convers an osm PBF file to o5m
    Args:
        osm_convert_path:
        pbf:
        o5m:

    Returns:

    """

    command = '""{}"  "{}" --complete-ways -o="{}""'.format(osm_convert_path, pbf, o5m)
    os.system(command)

def filter_osm(osm_filter_path, o5m, filtered_o5m):
    """Filters an o5m OSM file to only motorways, trunks, primary and secondary roads
    """
    command = '""{}"  "{}" --keep="highway=motorway =motorway_link =primary =primary_link =secondary =secondary_link =trunk =trunk_link" > "{}""'.format(osm_filter_path, o5m, filtered_o5m)
    os.system(command)

def graph_to_gdf(G):
    """Takes in a networkx graph object and outputs shapefiles at the paths indicated by edge_shp and node_shp
    Arguments:
        G []: networkx graph object to be converted
        edge_shp [str]: output path including extension for edges shapefile
        node_shp [str]: output path including extension for nodes shapefile
    Returns:
        None
    """
    # now only multidigraphs and graphs are used
    if type(G) == nx.classes.graph.Graph:
        G = nx.MultiGraph(G)

    nodes, edges = osmnx.graph_to_gdfs(G)

    dfs = [edges, nodes]
    for df in dfs:
        for col in df.columns:
            if df[col].dtype == np_object and col != df.geometry.name:
                df[col] = df[col].astype(str)

    return edges, nodes

def create_network_from_osm_dump(o5m, o5m_filtered, osm_filter_exe, **kwargs):
    """
    Filters and generates a graph from an osm.pbf file
    Args:
        pbf: path object for .pbf file
        o5m: path for o5m file function object for filtering infrastructure from osm pbf file
        **kwargs: all kwargs to osmnx.graph_from_file method. Use simplify=False and retain_all=True to preserve max
        complexity when generating graph

    Returns:
        G: graph object
        nodes: geodataframe representing graph nodes
        edges: geodataframe representing graph edges
    """

    filter_osm(osm_filter_exe, o5m, o5m_filtered)
    G = osmnx.graph_from_file(o5m_filtered, **kwargs)
    G = G.to_undirected()
    nodes, edges = graph_to_gdf(G)

    return G, nodes, edges

def compare_files(ref_files, test_files):
    for ref_file, test_file in zip(ref_files, test_files):
        if str(test_file).endswith('nodes.geojson'):
            pass
        else:
            assert filecmp.cmp(ref_file, test_file), '{} and {} do are not the same'.format(str(ref_file), str(test_file))
        os.remove(test_file)

if __name__=='__main__':

    # run function
    root = Path(__file__).parents[2]
    test_output_dir = root / 'tests/sample_output'
    test_input_dir = root / 'tests/sample_data'

    osm_filter_exe = root / 'osmfilter.exe'
    osm_convert_exe = root / 'osmconvert64.exe'
    pbf = test_input_dir / 'sample_data_osm_dumps' / r"NL332.osm.pbf"
    o5m = test_output_dir / 'sample_output_osm_dumps' / r"NL332.o5m"
    o5m_filtered = test_output_dir / 'sample_output_osm_dumps/NL332_filtered.o5m'

    convert_osm(osm_convert_exe, pbf, o5m)
    filter_osm(osm_filter_exe, o5m,  o5m_filtered)
    G, nodes, edges = create_network_from_osm_dump(o5m, o5m_filtered, osm_filter_exe, simplify=True, retain_all=True)

    nodes.to_file(test_output_dir / 'sample_output_network_shp' / 'NL332_nodes.geojson', driver='GeoJSON')
    edges.to_file(test_output_dir / 'sample_output_network_shp' / 'NL332_edges.geojson', driver='GeoJSON')

    # testing
    ref_files = list((test_output_dir / 'sample_output_network_shp/reference').glob('*.geojson'))
    test_files = list((test_output_dir / 'sample_output_network_shp').glob('*.geojson'))

    compare_files(ref_files, test_files)

    ref_files = list((test_output_dir / 'sample_output_osm_dumps/reference').glob('*.o5m'))
    test_files = list((test_output_dir / 'sample_output_osm_dumps').glob('*.o5m'))

    compare_files(ref_files, test_files)
    print('done')
