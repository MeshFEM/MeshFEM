////////////////////////////////////////////////////////////////////////////////
// Timer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a dynamic collection of timers.
//      Timers can be organized in sections. Timer T within section S is
//      reported as S:T. If a new section, S2, is started while another, S1, is
//      still running, the new section and its timers are reported as S1:S2.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
////////////////////////////////////////////////////////////////////////////////
#ifndef TIMER_HH
#define TIMER_HH
#include <map>
#include <iostream>
#include <string>
#include <cassert>
#include <list>

#include <sys/timeb.h>
#ifndef WIN32
#include <sys/time.h>
#endif

// Get time in seconds
inline double Time(void) {
#ifdef WIN32
    struct _timeb t;
    _ftime(&t);
    return double(t.time) + double(t.millitm) / 1000.0;
#else // WIN32
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec/1.0e6;
#endif // WIN32
}

class Timer
{
private:
    struct _Timer {
        bool running;
        double startTime, time;
        int invocations;
        _Timer() : running(false), time(0), invocations(0) { start(); }
        // gets elapsed time (even if currently running)
        double elapsed() const { return running ? time + (Time() - startTime) : time; }
        void stop()  { assert(running); time += Time() - startTime; running = false; }
        void start() {
            if (running) {
                std::cerr << "ERROR: timer already running. Reported timings will be inaccurate." << std::endl;
                stop();
            }
            assert(!running); running = true; ++invocations; startTime = Time();
        }
    };

    typedef std::map<std::string, _Timer> TimerMap;
    typedef TimerMap::iterator           TimerIterator;
    typedef TimerMap::const_iterator     TimerConstIterator;

    struct _Section : public _Timer {
        TimerMap timers;
        _Section() : _Timer() { }
        void startTimer(std::string name) {
            TimerIterator it = timers.find(name);
            if (it != timers.end()) {
                if (it->second.running) {
                    std::cerr << "ERROR: timer " << name << " already started. "
                              << std::endl;
                }
                it->second.start();
            }
            else {
                timers[name] = _Timer();
            }
        }

        using _Timer::start;
        void stop() {
            _Timer::stop();
            // Also stop all our sub-timers...
            for (auto &entry: timers) {
                if (entry.second.running) {
                    std::cerr << "WARNING: stopping timer " << entry.first
                              << " implicitly in enclosing section's stop()"
                              << std::endl;
                    entry.second.stop();
                }
            }
        }

        void start(const std::string &name) {
            auto lb = timers.lower_bound(name);
            if ((lb == timers.end()) || (lb->first != name))
                timers.emplace_hint(lb, name, _Timer());
            else
                lb->second.start(); // The full section timer must be started too...
        }
        void stop(const std::string &name) { timers.at(name).stop(); }

        void report(std::ostream &os) {
            for (auto &entry: timers)
                os << displayName(entry.first) << '\t' << entry.second.elapsed() << '\t'
                   << entry.second.invocations << std::endl;
        }
    };

    typedef std::map<std::string, _Section> SectionMap;
    typedef SectionMap::iterator            SectionIterator;
    typedef SectionMap::const_iterator      SectionConstIterator;

    SectionMap                              m_sections;
    std::list<std::string>                  m_sectionStack;

    static std::string displayName(std::string name) {
        size_t levels = 0;
        for (char c : name)
            if (c == ':') ++levels;
        if (levels == 0) return name;

        std::string result(4 * levels, ' ');
        result.append(name, name.rfind(':') + 1, std::string::npos);
        return result;
    }

public:
    Timer() { reset(); }

    const SectionMap &sections() { return m_sections; }

    void startSection(std::string name) {
        if (!m_sectionStack.empty())
            name = m_sectionStack.back() + ':' + name;
        m_sectionStack.push_back(name);
        auto lb = m_sections.lower_bound(name);
        if ((lb == m_sections.end()) || (lb->first != name))
            m_sections.emplace_hint(lb, name, _Section());
        else
            lb->second.start(); // The full section timer must be started too...
    }

    void stopSection(std::string name) {
        assert(!m_sectionStack.empty());
        std::string currentName = m_sectionStack.back();
        m_sectionStack.pop_back();

        if (!m_sectionStack.empty())
            name = m_sectionStack.back() + ':' + name ;
        
        if (name != currentName) {
            std::cerr << "ERROR: sections must be stopped in the reverse of "
                         "the order they were started." << std::endl;
            std::cerr << "(Expected " << currentName << ", but got " << name
                      << ")" << std::endl;
        }

        m_sections.at(name).stop();
    }

    void start(std::string timer) {
        std::string sectionName;
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        m_sections.at(sectionName).start(timer);
    }

    void stop(std::string timer) {
        std::string sectionName;
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        m_sections.at(sectionName).stop(timer);
    }

    // Remove all timers and restart the global section
    void reset() {
        m_sections.clear();
        m_sectionStack.clear();
        // (re)start global section
        m_sections.insert(std::make_pair("", _Section()));
    }

    void report(std::ostream &os) {
        for (SectionIterator it = m_sections.begin(); it != m_sections.end(); ++it) {
            if (it->first != "") { // Skip global section... this is reported at the end
                os << displayName(it->first) << '\t' << it->second.elapsed()
                   << '\t' << it->second.invocations << '\n';
                it->second.report(os);
            }
        }

        m_sections.at("").report(os);
        os << "Full time\t" << m_sections.at("").elapsed() << '\n';
    }
};

#endif // TIMER_HH
