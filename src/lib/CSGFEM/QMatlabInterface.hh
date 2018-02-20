////////////////////////////////////////////////////////////////////////////////
// QMatlabInterface.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      QT interface for MATLAB Engine
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/07/2013 00:56:10
////////////////////////////////////////////////////////////////////////////////
#ifndef QMATLAB_INTERFACE_HH
#define QMATLAB_INTERFACE_HH

#include "MatlabInterface/MatlabInterface.h"
#include "QCommandLine.hh"
#include <QObject>
#include <QWidget>
#include <iostream>
#include <string>

class QTextEdit;

class QMatlabInterface : public QWidget, public MatlabInterface
{
    Q_OBJECT

public:
    QMatlabInterface(QWidget *parent = NULL);
    ~QMatlabInterface() { }

    void appendNotification(const char *note, bool error = false);
    void appendText(const char *note);

    void setEcho(bool echo) { m_echo = echo; }
    bool echo() const { return m_echo; }

    virtual bool putVar(const char *name, const mxArray *pm) {
        bool success = MatlabInterface::putVar(name, pm);
        if (m_echo) {
            QString note;
            if (success)
                note.sprintf("Added variable '%s'", name);
            else
                note.sprintf("ERROR: failed to add variable '%s'", name);
            appendNotification(note.toLatin1(), !success);
        }

        return success;
    }

    using MatlabInterface::Eval;
    virtual int Eval(const char *command, std::string &output_str,
                     std::string &error_str) {
        int ret = MatlabInterface::Eval(command, output_str, error_str);
        if (m_echo) {
            QString note;
            note.sprintf(">> %s\n", command);
            appendNotification(note.toLatin1(), false);
            appendText(output_str.c_str());
            if (ret) {
                appendNotification(error_str.c_str(), true);
            }
        }
        return ret;
    }

    void keyPressEvent(QKeyEvent *event) {
        // Redirect all key presses other than control (Apple) or copy/paste to
        // the command line.
        if (event->matches(QKeySequence::Copy) ||
            (event->key() == Qt::Key_Control)) {
            QWidget::keyPressEvent(event);
        }
        else {
            g_commandLine->setFocus();
            g_commandLine->keyPressEvent(event);
        }
    }
public slots:
    void commandEntered(QString cmd);

private:
    QCommandLine *g_commandLine;
    QTextEdit *g_outputView;
    bool m_echo;
protected:
    void changeEvent(QEvent *event);
};

#endif // QMATLAB_INTERFACE_HH

