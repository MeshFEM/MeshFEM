////////////////////////////////////////////////////////////////////////////////
// QMatlabInterface.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      QT interface for MATLAB Engine
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
////////////////////////////////////////////////////////////////////////////////
#include "QMatlabInterface.hh"
#include "QCommandLine.hh"

#include <cstring>
#include <cstdio>
#include <QtGui>

QMatlabInterface::QMatlabInterface(QWidget *parent)
    : QWidget(parent), MatlabInterface(), m_echo(true)
{
    g_commandLine = new QCommandLine();
    g_outputView = new QTextEdit();
    g_outputView->setReadOnly(true);
    QFont font("Courier");
    font.setStyleHint(QFont::Monospace);
    g_outputView->setFont(font);

    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(g_outputView);
    layout->addWidget(g_commandLine);

    setLayout(layout);
    this->layout()->setContentsMargins(10, 10, 10, 10);
    setWindowTitle("MATLAB Instance");

    setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);

    setFocusPolicy(Qt::StrongFocus);
    QObject::connect(g_commandLine, SIGNAL(commandEntered(QString)),
                     this, SLOT(commandEntered(QString)));
    QObject::connect(g_commandLine, SIGNAL(clearOutput()),
                     g_outputView, SLOT(clear()));
}

void QMatlabInterface::appendText(const char *text)
{
    QTextCursor cursor = g_outputView->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(text);
    g_outputView->setTextCursor(cursor);
}

void QMatlabInterface::appendNotification(const char *note, bool error)
{
    QString noteStr = Qt::escape(QString(note));
    if (error)
        noteStr.sprintf("<b><font color='red'>%s</font></b><br>",
                        (const char *)noteStr.toAscii());
    else
        noteStr.sprintf("<b>%s</b><br>",
                        (const char *)noteStr.toAscii());

    QTextCursor cursor = g_outputView->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertHtml(noteStr);
    g_outputView->setTextCursor(cursor);
}

void QMatlabInterface::commandEntered(QString cmd)
{
    QRegExp exitFinder("(^|;)\\s*exit\\b.*");
    bool hasExit = cmd.contains(exitFinder);
    if (hasExit) {
        cmd.replace(exitFinder, "");
    }

    Eval(cmd.toAscii());

    if (hasExit) {
        appendNotification("WARNING: exit command disabled", true);
    }
}

void QMatlabInterface::changeEvent(QEvent *event)
{
    if ((event->type() == QEvent::ActivationChange) && isActiveWindow()) {
        g_commandLine->setFocus();
    }
    QWidget::changeEvent(event);
}

