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

class QTextEdit;

class QMatlabInterface : public QWidget, public MatlabInterface
{
    Q_OBJECT

public:
    QMatlabInterface(QWidget *parent = NULL);
    ~QMatlabInterface() { delete[] m_outputBuffer; }

    void appendNotification(const char *note, bool error = false);

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

    virtual int Eval(const char *command) {
        int ret = MatlabInterface::Eval(command);
        bool success = (ret == 0);
        QString note;
        if (success)
            note.sprintf(">> %s\n", command);
        else
            note.sprintf("ERROR: failed to run command '%s'", command);
        appendNotification(note.toAscii(), !success);
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
    char *m_outputBuffer;
    size_t m_outputBufferSize;

    void m_terminateOutput();
protected:
    void changeEvent(QEvent *event);
};

#endif // QMATLAB_INTERFACE_HH

