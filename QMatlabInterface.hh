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

    virtual bool putVar(const char *name, const mxArray *pm) {
        bool success = MatlabInterface::putVar(name, pm);
        QString note;
        if (success)
            note.sprintf("Added variable '%s'", name);
        else
            note.sprintf("ERROR: failed to add variable '%s'", name);
        appendNotification(note.toAscii(), !success);

        return success;
    }

    using MatlabInterface::Eval;
    virtual int Eval(const char *command, std::string &output_str,
                     std::string &error_str) {
        int ret = MatlabInterface::Eval(command, output_str, error_str);
        QString note;
        note.sprintf(">> %s\n", command);
        appendNotification(note.toAscii(), false);
        appendText(output_str.c_str());
        if (ret) {
            appendNotification(error_str.c_str(), true);
        }
        return ret;
    }

    void keyPressEvent(QKeyEvent *event) {
        g_commandLine->setFocus();
        g_commandLine->keyPressEvent(event);
    }
public slots:
    void commandEntered(QString cmd);

private:
    QCommandLine *g_commandLine;
    QTextEdit *g_outputView;
protected:
    void changeEvent(QEvent *event);
};

#endif // QMATLAB_INTERFACE_HH

