#!/usr/bin/env python
import sys
import numpy as np
import json

Tolerance = 1e-5


# Function checks if two vertices are close enough to each other
def same_vertex(v1, v2):
    if (abs(v1[0] - v2[0]) < Tolerance) and abs(v1[1] - v2[1]) < Tolerance:
        return True

    return False


# Function checks the two vertices of both edges are the same (edges could be inverted and result would be the same)
def same_edges(edge1, edge2):
    if not edge1:
        if not edge2:
            return True
        else:
            return False
    else:
        if not edge2:
            return False

    if same_vertex(edge1[0], edge2[0]) and same_vertex(edge1[1], edge2[1]):
        return True

    if same_vertex(edge1[0], edge2[1]) and same_vertex(edge1[1], edge2[0]):
        return True

    return False


# Function verifies if vertex v is in vertices list
def in_vertices_list(v, vertices):
    for u in vertices:
        if same_vertex(u, v):
            return True

    return False


# Function verifies if edge e is in edges list
def in_edges_list(e, edges):
    for s in edges:
        if same_edges(e, s):
            return True

    return False


# Based on vertices, computes min corner of bounding box
def compute_min_corner(vertices):
    min_corner = np.array([float("inf"), float("inf")])

    for u in vertices:
        if u[0] <= min_corner[0]:
            min_corner[0] = u[0]
        if u[1] <= min_corner[1]:
            min_corner[1] = u[1]

    return min_corner


# Based on vertices, computes max corner of bounding box
def compute_max_corner(vertices):
    max_corner = np.array([float("-inf"), float("-inf")])

    for u in vertices:
        if u[0] >= max_corner[0]:
            max_corner[0] = u[0]
        if u[1] >= max_corner[1]:
            max_corner[1] = u[1]

    return max_corner


# find index of vertex v in vertices list
def find_vertex(v, vertices):
    i = 0
    for u in vertices:
        if same_vertex(u, v):
            return i
        i += 1

    return -1


# find all edges connected to v
def all_edges_connected_to(v_index, edges_indices, edges):
    connected_edges = []
    used = [False] * len(edges_indices)

    initial_v = v_index
    while True:
        found_next = False

        for i in range(0, len(edges_indices)):
            if used[i]:
                continue

            v1, v2 = edges_indices[i]

            if v1 == v_index:
                connected_edges.append(edges[i])
                used[i] = True
                v_index = v2

                # restart
                found_next = True
                break
            elif v2 == v_index:
                connected_edges.append(edges[i])
                used[i] = True
                v_index = v1

                # restart
                found_next = True
                break

        if not found_next:
            break

        if initial_v == v_index:
            break

    return connected_edges


# given input polygons corresponding to an .off file (outputs of ), computes the corresponding
# .poly file, that can be used with triangle to mesh structure
def execute(in_path, out_path):
    # read input data
    input_file = open(in_path)
    input_data = input_file.readlines()

    vertices = []
    edges = []

    # parse information
    index = 0
    num_vertices = 0
    for line in input_data:
        if index == 0:
            if line.strip() != "OFF":
                sys.exit("First line of input file should contain OFF and contains " + line + " instead")

        elif index == 1:
            num_vertices, num_edges, num_faces = line.split(" ")
            num_vertices = int(num_vertices)
            num_edges = int(num_edges)
            num_faces = int(num_faces)
            print("# Vertices: " + str(num_vertices))
            print("# Edges: " + str(num_edges))
            print("# Faces: " + str(num_faces))

        elif index < (2 + num_vertices):
            x, y, z = line.split(" ")
            float_coordinates = np.array([float(x), float(y)])
            vertices.append(float_coordinates)

        elif index < (2 + num_vertices + num_edges):
            n, v1, v2 = line.split(" ")
            new_edge = [int(v1), int(v2)]
            edges.append(new_edge)

        index += 1

    # find polygons
    polygons = []
    used_vertices = 0
    part_of_polygon = [False] * len(vertices)
    while used_vertices < len(vertices):
        polygon = []

        # Find non-used vertex
        initial_vertex = 0
        for i in range(0, len(vertices)):
            if not part_of_polygon[i]:
                initial_vertex = i
                break

        # add initial vertex
        polygon.append(vertices[initial_vertex])
        part_of_polygon[initial_vertex] = True
        used_vertices += 1

        # search for whole polygon
        current_vertex = initial_vertex
        # print "Current vertex: " + str(current_vertex)
        while True:
            found_next = False

            # loop through edges to find next vertex
            for edge in edges:
                v1, v2 = edge

                if v1 == current_vertex and not part_of_polygon[v2]:
                    polygon.append(vertices[v2])
                    part_of_polygon[v2] = True
                    used_vertices += 1
                    current_vertex = v2
                    found_next = True
                    break
                elif v2 == current_vertex and not part_of_polygon[v1]:
                    polygon.append(vertices[v1])
                    part_of_polygon[v1] = True
                    current_vertex = v1
                    used_vertices += 1
                    found_next = True
                    break

            if not found_next:
                break

            if initial_vertex == current_vertex:
                print "the end"
                part_of_polygon[initial_vertex] = True
                used_vertices += 1
                break

        polygons.append(polygon)

    # Create edges
    edges = []
    polygons_edges = []
    for polygon in polygons:
        polygon_edges = []
        if same_vertex(polygon[0], polygon[-1]):
            max_index = len(polygon) - 1
        else:
            max_index = len(polygon)
        for i in range(max_index):
            polygon_edges.append([polygon[i], polygon[(i + 1) % len(polygon)]])
        edges += polygon_edges
        polygons_edges.append(polygon_edges)

    # Find number of different edges
    number_different_edges = 0
    different_edges = []
    for edge in edges:
        if not in_edges_list(edge, different_edges):
            different_edges.append(edge)
            number_different_edges += 1

    # Discover all contact edges
    contact_edges = []
    for p in range(0, len(polygons)):
        for q in range(p + 1, len(polygons)):
            for ep in polygons_edges[p]:
                for eq in polygons_edges[q]:
                    if same_edges(ep, eq):
                        contact_edges.append(ep)

    # Print information about edges
    print("Total edges: " + str(len(edges)))
    print("Different edges: " + str(number_different_edges))
    print("Contact edges: " + str(len(contact_edges)))

    # Find all (different) vertices
    vertices = []
    for polygon in polygons:
        for point in polygon:
            is_new = True
            for vertex in vertices:
                if same_vertex(point, vertex):
                    is_new = False
                    break

            if is_new:
                vertices.append(point)

    print("Total vertices: " + str(len(vertices)))

    # With list of vertices, find out how to translate objects to origin and scale then correctly so they always
    # fall into [-1,1]^2
    min_corner = np.array([-1, -1])
    max_corner = np.array([1, 1])
    dimensions = max_corner - min_corner
    max_dimension = max(dimensions[0], dimensions[1])
    center = min_corner + dimensions / 2

    # Write collected information to output file
    out_file = open(out_path, "w")
    out_file.write("# description of polygon, generated by svgToPoly.py on " + in_path + "\n")

    # Now, print vertices
    index = 0
    out_file.write("# vertices list\n")
    out_file.write('{} {} {} {}\n'.format(len(vertices), 2, 0, 0))
    final_vertices = []
    for vertex in vertices:
        # First, transform point to normalized location
        point = 2 * (vertex - center) / max_dimension
        final_vertices.append(point)
        out_file.write('{} {} {}\n'.format(index, point[0], point[1]))
        index += 1

    # Now, print edges of each polygon, with the (index+1) of the polygon as attribute
    index = 0
    polygon_index = 1
    out_file.write("\n# edges list\n")
    out_file.write('{} {}\n'.format(number_different_edges, 1))
    for polygon_edges in polygons_edges:
        for edge in polygon_edges:
            if not in_edges_list(edge, contact_edges):
                v1_index = find_vertex(edge[0], vertices)
                v2_index = find_vertex(edge[1], vertices)

                out_file.write('{} {} {} {}\n'.format(index, v1_index, v2_index, polygon_index))
                index += 1

        polygon_index += 1
    out_file.write("\n")

    # Print contact edges
    out_file.write("\n# contact edges list\n")
    for edge in contact_edges:
        v1_index = find_vertex(edge[0], vertices)
        v2_index = find_vertex(edge[1], vertices)

        out_file.write('{} {} {} {}\n'.format(index, v1_index, v2_index, -1))
        index += 1

    # Print holes
    index = 0
    hole_points = []
    out_file.write("\n# holes list\n")
    out_file.write(str(len(hole_points)) + "\n")
    for hole in hole_points:
        hole = 2 * (hole - center) / max_dimension
        out_file.write('{} {} {}\n'.format(index, hole[0], hole[1]))
        index += 1

    out_file.close()

    # This script also searches for boundary conditions and generates a json with the information collected
    boundary_conditions_json = {"no_rigid_motion": False}
    regions = []

    # Write information about vertices connections, to help building contact boundary conditions
    vertices_connections = [[] for i in range(len(vertices))]
    number_of_contact_vertices = 0
    for edge in contact_edges:
        v1_index = find_vertex(edge[0], vertices)
        v2_index = find_vertex(edge[1], vertices)

        vertices_connections[v1_index].append(v2_index)
        vertices_connections[v2_index].append(v1_index)

        if len(vertices_connections[v1_index]) == 1:
            number_of_contact_vertices += 1
        if len(vertices_connections[v2_index]) == 1:
            number_of_contact_vertices += 1

    # Build contact regions
    added_vertices = 0
    visited = [False] * len(vertices)
    while added_vertices < number_of_contact_vertices:
        # Find end node of a path (only one incident edge)
        for i in range(0, len(vertices)):
            if len(vertices_connections[i]) == 1 and not visited[i]:
                endnode = i
                break

        # Starting from endnode, find entire path
        current_vertex = endnode
        visited[endnode] = True
        contact_path = [final_vertices[current_vertex].tolist()]
        added_vertices += 1
        while True:
            neighbors = vertices_connections[current_vertex]
            next = None
            for neighbor in neighbors:
                if not visited[neighbor]:
                    next = neighbor
                    break

            if next is None:
                break

            current_vertex = next
            visited[current_vertex] = True
            contact_path.append(final_vertices[current_vertex].tolist())
            added_vertices += 1

        # Add information to json structure and append new region
        region = {"type": "fracture", "value": [0, 0, 0], "path": contact_path}
        regions.append(region)

    if len(regions):
        boundary_conditions_json['regions'] = regions
        print(json.dumps(boundary_conditions_json, indent=4, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ./offToPoly.py <input .off> <output .poly>")
        print("example: ./offToPoly.py contact.off contact.poly")
        sys.exit(-1)

    execute(sys.argv[1], sys.argv[2])
