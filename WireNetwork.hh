////////////////////////////////////////////////////////////////////////////////
// WireNetwork.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        This class implements implicit surface computation of a wire network.
*/ 
//  Author:  Qingnan Zhou (qnzhou), qnzhou@gmail.com
//  Company:  New York University
//  Created:  05/07/2014 11:08:00
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <limits>

#include "LevelSet.hh"

template<typename _Vector>
class WireNetwork : public LevelSet<_Vector> {
public:
    typedef LevelSet<_Vector> super;
    using typename super::_BBox;
    using typename super::Vector;
    using typename super::Real;

    typedef Vector Vertex;
    typedef std::vector<Vertex> Vertices;
    typedef std::pair<size_t, size_t> Edge;
    typedef std::vector<Edge> Edges;

public:
    WireNetwork(const _BBox &domain, const std::string& wire_file, Real thickness)
        : super(domain), m_thickness(thickness) {
            parse_wire_file(wire_file);
            fit_wire_to_bbox();
            std::cout << "#v: " << m_vertices.size() << std::endl;
            std::cout << "#e: " << m_edges.size() << std::endl;
    }

    Real signedDistance(const Vector &p) const {
        Real minDist = std::numeric_limits<Real>::max();
        for (Edges::const_iterator itr = m_edges.begin(); itr != m_edges.end();
                itr++) {
            Real dist = compute_distance_to_edge(p, itr->first, itr->second);
            minDist = std::min(dist, minDist);
        }
        return minDist - m_thickness;
    }

private:
    void parse_wire_file(const std::string& wire_file) {
        const size_t LINE_WIDTH = 256;

        std::ifstream fin(wire_file.c_str());
        while (!fin.eof()) {
            char head = fin.peek();
            switch (head) {
                case 'v':
                    parse_vertex(fin);
                    fin.ignore(LINE_WIDTH, '\n');
                    break;
                case 'l':
                    parse_edge(fin);
                    fin.ignore(LINE_WIDTH, '\n');
                    break;
                default:
                    fin.ignore(LINE_WIDTH, '\n');
            }
        }
    }

    void fit_wire_to_bbox() {
        const size_t num_vertices = m_vertices.size();
        compute_wire_bbox();
        const _BBox &domain = this->m_domain;
        Vector bbox_center = (domain.minCorner + domain.maxCorner) * 0.5;
        Vector bbox_size = domain.maxCorner - domain.minCorner;

        Vector wire_center = (m_wire_bbox_max + m_wire_bbox_min) * 0.5;
        Vector offset = bbox_center - wire_center;

        Vector scale = bbox_size.array() / m_wire_bbox_size.array();

        for (size_t i=0; i<num_vertices; i++) {
            m_vertices[i] = (m_vertices[i] - wire_center).array() * scale.array();
            m_vertices[i] += bbox_center;
        }
    }

    void compute_wire_bbox() {
        const size_t num_vertices = m_vertices.size();
        m_wire_bbox_min = m_vertices[0];
        m_wire_bbox_max = m_vertices[0];
        for (size_t i=0; i<num_vertices; i++) {
            const Vector& v = m_vertices[i];
            m_wire_bbox_min[0] = std::min(m_wire_bbox_min[0], v[0]);
            m_wire_bbox_max[0] = std::max(m_wire_bbox_max[0], v[0]);

            m_wire_bbox_min[1] = std::min(m_wire_bbox_min[1], v[1]);
            m_wire_bbox_max[1] = std::max(m_wire_bbox_max[1], v[1]);
#if DIM!=2
            m_wire_bbox_min[2] = std::min(m_wire_bbox_min[2], v[2]);
            m_wire_bbox_max[2] = std::max(m_wire_bbox_max[2], v[2]);
#endif
        }

        m_wire_bbox_size = m_wire_bbox_max - m_wire_bbox_min;
    }

    void parse_vertex(std::ifstream& fin) {
        Vertex v;
        char head;
        fin >> head;
        fin >> v[0] >> v[1];
#if DIM!=2
        fin >> v[2];
#endif
        m_vertices.push_back(v);
    }

    void parse_edge(std::ifstream& fin) {
        char head;
        size_t v1, v2;
        fin >> head;
        fin >> v1 >> v2;
        m_edges.push_back(Edge(v1-1, v2-1));
    }

    Real compute_distance_to_edge(const Vector& p, size_t v1, size_t v2) const {
        const Vector& p1 = m_vertices[v1];;
        const Vector& p2 = m_vertices[v2];;
        Vector v =  p - p1;
        Vector e = p2 - p1;
        Real proj_fraction = v.dot(e) / e.squaredNorm();
        if (proj_fraction < 0.0) {
            return v.norm();
        } else if (proj_fraction > 1.0) {
            return (p - p2).norm();
        } else {
            return (v - e * proj_fraction).norm();
        }
    }

private:
    Vertices m_vertices;
    Edges m_edges;
    Real m_thickness;
    Vector m_wire_bbox_min;
    Vector m_wire_bbox_max;
    Vector m_wire_bbox_size;
};
