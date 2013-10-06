////////////////////////////////////////////////////////////////////////////////
// Flipbook.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Assists in the generation of "flipbooks," renderings of a list of
//      results.
//
//      Flipbooks are driven by the ResultsWindowController (since it must be
//      able to trigger the results selection), but the actual image output is
//      triggered by FEMView's draw() routine.
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
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include "ResultsCollector.hh"
#include "AnalysisSettings.hh"

class Flipbook {
public:
    typedef ResultsCollector_t::Result Result;

    Flipbook() : m_results(NULL), m_frame(0), m_masked(true) { }

    Flipbook(const std::string &directory,
             const ResultsCollector_t *rc,
             const std::vector<std::string> &resultPaths)
        : m_directory(directory), m_results(rc), m_resultPaths(resultPaths),
          m_frame(0) { }

    void setMasked(bool masked) { m_masked = masked; }
    bool active() const { return !m_masked &&
                                 (m_frame < m_resultPaths.size()); }

    void advance() { if (active()) ++m_frame; }

    const std::string &path(size_t frame) const {
        assert(frame < m_resultPaths.size());
        return m_resultPaths[frame];
    }
    const std::string &path() const { return path(m_frame); }

    std::shared_ptr<const Result> currentResult() const {
        assert(active() && m_results);
        return m_results->getResultWithPath(path());
    }

    std::string imagePath(size_t frame) const {
        return m_directory + "/" + path(frame) + ".png";
    }
    std::string imagePath() const { return imagePath(m_frame); }

    // To be called from the display function after drawing has completed.
    void snapshot(QGLWidget *view) const {
        if (m_frame < m_resultPaths.size()) {
            QImage img = view->grabFrameBuffer();
            QString path = QString::fromStdString(imagePath());
            QImageWriter writer(path);
            bool success = writer.write(img);
            if (!success)
                std::cout << "Failed to write image: "
                          << path.toStdString() << std::endl;
        }
    }

    void writeFlipperJSON(const std::string &title,
                          const std::vector<std::string> &settingNames) const {
        std::ofstream jsonOut(m_directory + "/frames.js");
        if (!jsonOut.is_open()) {
            std::cout << "Failed to open output file '"
                      << m_directory + "/frames.js" << '\'' << std::endl;
            return;
        }

        jsonOut << "title = '" << escapedString(title) << "';" << std::endl;
        jsonOut << "statistics = ['model', 'result max', 'result min'";
        for (size_t s = 0; s < settingNames.size(); ++s) {
            jsonOut << ", '" << escapedString(settingNames[s]) << "'";
        }
        jsonOut << "];" << std::endl;
        jsonOut << "variants = ['Plain'];" << std::endl;
        jsonOut << "frames = [" << std::endl;
        
        assert(m_results);
        for (size_t f = 0; f < m_resultPaths.size(); ++f) {
            jsonOut << "    {'image': ['" << escapedString(imagePath(f)) << "']";

            jsonOut << ", 'model': '"
                    << escapedString(getModelPathComponent(path(f))) << "'";
            jsonOut << ", 'result max': '"
                    << m_results->getResultWithPath(path(f))->getMaxScalar(Result::PER_ELEM) << "'";
            jsonOut << ", 'result min': '"
                    << m_results->getResultWithPath(path(f))->getMinScalar(Result::PER_ELEM) << "'";

            AnalysisSettings settings;
            m_results->getSettings(getSettingsPathComponent(path(f)), settings);

            for (size_t s = 0; s < settingNames.size(); ++s) {
                std::string name = escapedString(settingNames[s]);
                jsonOut << ", '" << name << "': '"
                        << settings.displayString(name) << "'";
            }

            jsonOut << "}," << std::endl;
        }

        jsonOut << "];" << std::endl;
    }

private:
    std::string m_directory;
    const ResultsCollector_t *m_results;
    std::vector<std::string> m_resultPaths;
    size_t m_frame;
    bool m_masked;
};

#endif // FLIPBOOK_HH
