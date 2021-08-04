# -*- coding: utf-8 -*-
"""
Created on 30-7-2021

@author: F.C. de Groen, Deltares
"""

from geopy import distance
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
import geopandas as gpd
import rtree
import logging
from tqdm import tqdm, trange
from .networks_utils import pairs, line_length


def read_OD_files(origin_paths, origin_names, destination_paths, destination_names, od_id, crs_):
    origin = gpd.GeoDataFrame(columns=[od_id, 'o_id', 'geometry'], crs=crs_)
    destination = gpd.GeoDataFrame(columns=[od_id, 'd_id', 'geometry'], crs=crs_)

    if isinstance(origin_paths, str):
        origin_paths = [origin_paths]
    if isinstance(destination_paths, str):
        destination_paths = [destination_paths]
    if isinstance(origin_names, str):
        origin_names = [origin_names]
    if isinstance(destination_names, str):
        destination_names = [destination_names]

    for op, on in zip(origin_paths, origin_names):
        origin_new = gpd.read_file(op, crs=crs_)
        origin_new = origin_new[[od_id, 'geometry']]
        origin_new['o_id'] = on + "_" + origin_new[od_id].astype(str)
        origin = origin.append(origin_new, ignore_index=True, sort=False)

    for dp, dn in zip(destination_paths, destination_names):
        destination_new = gpd.read_file(dp, crs=crs_)
        destination_new = destination_new[[od_id, 'geometry']]
        destination_new['d_id'] = dn + "_" + destination_new[od_id].astype(str)
        destination = destination.append(destination_new, ignore_index=True, sort=False)

    od = pd.concat([origin, destination], sort=False)
    od['FID'] = list(range(len(od.index)))

    return od


def create_OD_pairs(od, graph, id_name):
    """Get centroids of the selected NUTS-3 regions and gets closest vertice on the road of a graph.
    Args:
        origins [string]: file path of shapefile of the NUTS-3 regions in Europe
        country_codes [list of string(s)]: list of NUTS country codes
        graph [networkX graph]: graph of the roads of a or multiple European countries
        name [string]: name of the analysis
        file_output [string]: path to folder where the geodataframe and/or pickle should be stored
        save_shp [bool]: True/False to save the OD pairs to a shapefile in the folder 'file_output'
        save_pickle [bool]: True/False to save the OD pairs to a pickle in the folder 'file_output'
    Returns:
        centroids: dataframe of vertices closest to the centroids of the selected NUTS-3 regions
    """

    # find closest vertice of road network to centroid
    # create dictionary of the roads geometries
    edge_list = [e for e in graph.edges.data() if 'geometry' in e[-1]]
    vertices_dict = {}
    for line in edge_list:
        vertices_dict[(line[0], line[1])] = [Point(p) for p in set(list(line[-1]['geometry'].coords))]

    # create list of all points to search in
    all_vertices = [p for sublist in list(vertices_dict.values()) for p in sublist]

    # create an empty spatial index object to search in
    idx = rtree.index.Index()

    # populate the spatial index
    for i, pnt in enumerate(all_vertices):
        idx.insert(i, pnt.bounds)

    od = find_closest_vertice(od, idx, all_vertices, vertices_dict, edge_list, id_name)

    return od


def find_closest_vertice(origins_destinations, spatial_idx, search_vertices, vertices_dict, edge_list, id_name):
    ODs = []
    match_ids = []
    for i, c in enumerate(tqdm(origins_destinations['geometry'], desc='Finding vertices closest to Origins and Destinations', leave=True)):
        # find the closest vertice and line the vertice lays on
        target = list(spatial_idx.nearest(c.bounds))

        # do nothing if the target point is further away from the endpoint than max_dist
        # or if they are at the same location
        if not target:
            continue

        points_list = [search_vertices[ip] for ip in target]

        # check on which road this point lays
        road_i = getKeysByValue(vertices_dict, points_list[0])
        match_ids.append([i[-1][id_name] for i in edge_list if (road_i[0] == i[0]) and (road_i[1] == i[1])][0])

        # save in list
        ODs.append(points_list[0])

    # save in dataframe
    origins_destinations['OD'] = ODs
    origins_destinations['match_ids'] = match_ids

    # save the road vertices closest to the centroids as geometry, delete the centroid geometry
    origins_destinations = gpd.GeoDataFrame(origins_destinations, geometry='OD')
    origins_destinations = origins_destinations.drop(columns=['geometry'])

    return origins_destinations


def find_new_nearest_vertice(edge_list,graph, od, id_name, match_OD,i):
    vertices_dict = {}
    for line in edge_list:
        vertices_dict[(line[0], line[1])] = [Point(p) for p in set(list(line[-1]['geometry'].coords))]
    all_vertices = [p for sublist in list(vertices_dict.values()) for p in sublist]
    # create an empty spatial index object to search in
    idx = rtree.index.Index()
    # populate the spatial index
    for j, pnt in enumerate(all_vertices):
        idx.insert(j, pnt.bounds)
    # find the closest vertice and line the vertice lays on
    target = list(idx.nearest(match_OD.coords[0]))
    match_OD = [all_vertices[ip] for ip in target]
    match_OD = match_OD[0]
    all_matches = [am for am in edge_list if match_OD.coords[0] in [p for p in set(list(am[-1]['geometry'].coords))]]
    return all_matches, match_OD


def split_line_with_points(line, points):
    """Splits a line string in several segments considering a list of points.
    """
    segments = []
    current_line = line

    # make a list of points and its distance to the start to sort them from small to large distance
    list_dist = [current_line.project(pnt) for pnt in points]
    list_dist.sort()

    for d in list_dist:
        # cut the line at a distance d
        seg, current_line = cut(current_line, d)
        if seg:
            segments.append(seg)
    segments.append(current_line)
    return segments


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # This is taken from shapely manual
    if (distance <= 0.0) | (distance >= line.length):
        return [None, LineString(line)]

    if isinstance(line, LineString):
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i + 1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                # check if the LineString contains an Z-value, if so, remove
                # only use XY because otherwise the snapping functionality doesn't work
                return [LineString([xy[0:2] for xy in coords[:i]] + [(cp.x, cp.y)]),
                        LineString([(cp.x, cp.y)] + [xy[0:2] for xy in coords[i:]])]
    elif isinstance(line, MultiLineString):
        for ln in line:
            coords = list(ln.coords)
            for i, p in enumerate(coords):
                pd = ln.project(Point(p))
                if pd == distance:
                    return [
                        LineString(coords[:i + 1]),
                        LineString(coords[i:])]
                if pd > distance:
                    cp = ln.interpolate(distance)
                    # check if the LineString contains an Z-value, if so, remove
                    # only use XY because otherwise the snapping functionality doesn't work
                    return [LineString([xy[0:2] for xy in coords[:i]] + [(cp.x, cp.y)]),
                            LineString([(cp.x, cp.y)] + [xy[0:2] for xy in coords[i:]])]





def getKeysByValue(dictOfElements, value):
    """
    https://thispointer.com/python-how-to-find-keys-by-value-in-dictionary/
    """
    theKey = 0
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if value in item[1]:
            theKey = item[0]
    return theKey


def add_od_nodes(graph, od, id_name):
    """From a geodataframe of vertices on a graph, adds nodes on that graph.
    Args:
        graph [networkX graph]: graph of the roads of a or multiple European countries
        od [geodataframe]: geodataframe from the function 'create_OD_pairs'
        id_name [string]: string
    Returns:
        graph: networkX graph with the nodes closes to the centroids of the NUTS-3 regions added.
        The ID's of the added nodes are adding in number from the highest ID of the nodes in the original graph.
    """
    # To make it easier to match the match_ids potential lists are turned into strings (or strings to strings)
    od['match_ids'] = od['match_ids'].astype(str)

    # Check the highest node id, to add on that
    max_node_id = max([n for n in graph.nodes()])
    logging.info('Adding Origin-Destination nodes to graph...')

    for i in trange(len(od.index), desc='Adding Origin-Destination nodes to graph', leave=True):
        # the vertice on the edge that is closest to the origin/destination point
        match_OD = od.iloc[i]['OD']
        # Check which roads belong to the centroids closest vertices
        all_matches = [e for e in graph.edges(data=True, keys=True) if str(e[-1][id_name]) == od.iloc[i]['match_ids']]
        if len(all_matches) > 1:
            # all_matches = [am for am in all_matches if match_OD in [(Point(p) for p in set(list(am[-1]['geometry'].coords)))]]
            all_matches = [am for am in all_matches if match_OD.coords[0] in [p for p in set(list(am[-1]['geometry'].coords))]]
            if len(all_matches) == 0:  # created to find nearest vertice when a new edge has already been created
                #todo build this in in the other def find nearest vertice
                edge_list = [e for e in graph.edges(data=True, keys=True) if
                             str(e[-1][id_name]) == od.iloc[i]['match_ids']]
                all_matches,match_OD = find_new_nearest_vertice(edge_list,graph, od, id_name, match_OD, i)
        if len(all_matches) == 1:
            if [am for am in all_matches if match_OD.coords[0] in [p for p in set(list(am[-1]['geometry'].coords))]] == []:
                edge_list = [e for e in graph.edges(data=True, keys=True) if
                             str(e[-1][id_name]) == od.iloc[i]['match_ids']]
                all_matches,match_OD = find_new_nearest_vertice(edge_list,graph, od, id_name, match_OD, i)
        if len(all_matches) == 0:  # when the edge does not exist anymore in the adjusted graph. look over the full graph and find the nearest vertice
            edge_list = [e for e in graph.edges.data() if 'geometry' in e[-1]]
            all_matches, match_OD = find_new_nearest_vertice(edge_list, graph, od, id_name, match_OD, i)
        m = all_matches[0]

        if 'geometry' in m[-1]:
            match_geom = m[-1]['geometry']
            if len(m)==3:
                match_edge = m[0],m[1],0
            else:
                match_edge = m[:3]
            match_name = od.iloc[i]['o_id']
            if match_name == 'nan':
                match_name = np.nan  # convert string nans to np.nans to be able to differentiate between origins and destinations in the next step.
            if not match_name == match_name:
                # match_name is nan, the point is not an origin but a destination
                match_name = od.iloc[i]['d_id']
            new_lines = split_line_with_points(match_geom, [match_OD])
            if len(new_lines) == 2:
                line1, line2 = new_lines
            else:
                # if the vertice is at the end of the road; you don't have to add a new node
                # but do add a new attribute to the node
                if (graph.nodes[match_edge[0]]['geometry'].coords[0][1] == match_OD.coords[0][1]) & (graph.nodes[match_edge[0]]['geometry'].coords[0][0] == match_OD.coords[0][0]):
                    if 'od_id' in graph.nodes[match_edge[0]]:
                        # the node already has a origin/destination attribute
                        graph.nodes[match_edge[0]]['od_id'] = graph.nodes[match_edge[0]]['od_id'] + ',' + match_name
                    else:
                        graph.nodes[match_edge[0]]['od_id'] = match_name
                elif (graph.nodes[match_edge[1]]['geometry'].coords[0][1] == match_OD.coords[0][1]) & (
                    graph.nodes[match_edge[1]]['geometry'].coords[0][0] == match_OD.coords[0][0]):
                    if 'od_id' in graph.nodes[match_edge[1]]:
                        graph.nodes[match_edge[1]]['od_id'] = graph.nodes[match_edge[1]]['od_id'] + ',' + match_name
                    else:
                        graph.nodes[match_edge[1]]['od_id'] = match_name
                elif (((graph.nodes[match_edge[0]]['geometry'].coords[0][1] == match_OD.coords[0][1]) & (graph.nodes[match_edge[0]]['geometry'].coords[0][0] == match_OD.coords[0][0])) == False) & (((graph.nodes[match_edge[1]]['geometry'].coords[0][1] == match_OD.coords[0][1]) & (graph.nodes[match_edge[1]]['geometry'].coords[0][0] == match_OD.coords[0][0])) == False):
                     print(i)
                     print('continue')
                     continue
                continue
            new_node_id = max_node_id + 1
            max_node_id = new_node_id

            graph.add_node(new_node_id, y=match_OD.coords[0][1], x=match_OD.coords[0][0], geometry=match_OD,
                           od_id=match_name)

            edge_data = graph.edges[match_edge]

            # Check which line is connected to which node. There can be 8 different combinations
            if (graph.nodes[match_edge[0]]['geometry'].coords[0][1] == line2.coords[-1][1]) & (
                graph.nodes[match_edge[0]]['geometry'].coords[0][0] == line2.coords[-1][0]):
                edge_data.update(length=line_length(line2), geometry=line2)
                graph.add_edge(match_edge[0], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[1]]['geometry'].coords[0][1] == line2.coords[0][1]) & (
                graph.nodes[match_edge[1]]['geometry'].coords[0][0] == line2.coords[0][0]):
                edge_data.update(length=line_length(line2), geometry=line2)
                graph.add_edge(match_edge[1], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[0]]['geometry'].coords[0][1] == line2.coords[0][1]) & (
                graph.nodes[match_edge[0]]['geometry'].coords[0][0] == line2.coords[0][0]):
                edge_data.update(length=line_length(line2), geometry=line2)
                graph.add_edge(match_edge[0], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[1]]['geometry'].coords[0][1] == line2.coords[-1][1]) & (
                graph.nodes[match_edge[1]]['geometry'].coords[0][0] == line2.coords[-1][0]):
                edge_data.update(length=line_length(line2), geometry=line2)
                graph.add_edge(match_edge[1], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[1]]['geometry'].coords[0][1] == line1.coords[0][1]) & (
                graph.nodes[match_edge[1]]['geometry'].coords[0][0] == line1.coords[0][0]):
                edge_data.update(length=line_length(line1), geometry=line1)
                graph.add_edge(match_edge[1], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[0]]['geometry'].coords[0][1] == line1.coords[-1][1]) & (
                graph.nodes[match_edge[0]]['geometry'].coords[0][0] == line1.coords[-1][0]):
                edge_data.update(length=line_length(line1), geometry=line1)
                graph.add_edge(match_edge[0], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[1]]['geometry'].coords[0][1] == line1.coords[-1][1]) & (
                graph.nodes[match_edge[1]]['geometry'].coords[0][0] == line1.coords[-1][0]):
                edge_data.update(length=line_length(line1), geometry=line1)
                graph.add_edge(match_edge[1], new_node_id, match_edge[-1], **edge_data)

            if (graph.nodes[match_edge[0]]['geometry'].coords[0][1] == line1.coords[0][1]) & (
                graph.nodes[match_edge[0]]['geometry'].coords[0][0] == line1.coords[0][0]):
                edge_data.update(length=line_length(line1), geometry=line1)
                graph.add_edge(match_edge[0], new_node_id, match_edge[-1], **edge_data)

            # remove the edge that is split in two
            u, v, k = match_edge
            graph.remove_edge(u, v, k)

    return graph

