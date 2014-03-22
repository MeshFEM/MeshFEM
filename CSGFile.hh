////////////////////////////////////////////////////////////////////////////////
// CSGFile.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements writing/reading .csg files. These are JSON files describing
//      a CSG tree.
//      Uses Boost property trees to parse but not to write! Boost's property
//      trees are untyped (all values are converted to strings) and have a hacky
//      implementation of arrays (anonymous nodes). This is okay for reading,
//      but it would produce garbage output.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/03/2013 23:17:00
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGFILE_HH
#define CSGFILE_HH

#include <iosfwd>

template<typename Vector>
class CSGTree;

template<typename Vector>
void parseCSGFile(const char *path, CSGTree<Vector> &csgTree);

template<typename Vector>
void parseCSGFile(std::istream &is, CSGTree<Vector> &csgTree);

template<typename Vector>
void writeCSGFile(const char *path, const CSGTree<Vector> &csgTree);

template<typename Vector>
void writeCSGFile(std::ofstream &os, const CSGTree<Vector> &csgTree);

#endif // CSGFILE_HH
