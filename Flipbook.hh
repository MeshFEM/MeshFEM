////////////////////////////////////////////////////////////////////////////////
// Flipbook.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Assists in the generation of "flipbooks," renderings of a list of
//      results.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/05/2013 13:03:46
////////////////////////////////////////////////////////////////////////////////
#ifndef FLIPBOOK_HH
#define FLIPBOOK_HH

#include <QGLWidget>
#include <QImage>
#include <QImageWriter>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "ResultsCollector.hh"

class Flipbook {
public:
    typedef ResultsCollector_t::Result Result;

    Flipbook() : m_results(NULL), m_frame(0) { }

    Flipbook(const std::string &directory,
             const ResultsCollector_t *rc,
             const std::vector<std::string> &resultPaths)
        : m_directory(directory), m_results(rc), m_resultPaths(resultPaths),
          m_frame(0) { }

    bool active() const { return m_frame < m_resultPaths.size(); }

    void advance() { if (active()) ++m_frame; }

    const std::string &path() const {
        assert(active());
        return m_resultPaths[m_frame];
    }

    std::shared_ptr<const Result> currentResult() const {
        assert(active() && m_results);
        return m_results->getResultWithPath(path());
    }

    // To be called from the display function after drawing has completed.
    void snapshot(QGLWidget *view) const {
        if (m_frame < m_resultPaths.size()) {
            QImage img = view->grabFrameBuffer();
            QString path = QString::fromStdString(m_directory + "/" +
                                      m_resultPaths[m_frame] + ".png");
            QImageWriter writer(path);
            bool success = writer.write(img);
            if (!success)
                std::cout << "Failed to write image: "
                          << path.toStdString() << std::endl;
        }
    }

private:
    std::string m_directory;
    const ResultsCollector_t *m_results;
    std::vector<std::string> m_resultPaths;
    size_t m_frame;
};

#endif // FLIPBOOK_HH
