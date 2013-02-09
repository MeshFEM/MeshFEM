////////////////////////////////////////////////////////////////////////////////
// QCommandLine.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Command line-like QLineEdit variant.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/07/2013 01:24:43
////////////////////////////////////////////////////////////////////////////////
#ifndef QCOMMAND_LINE_HH
#define QCOMMAND_LINE_HH

#include <QLineEdit>
#include <QKeyEvent>

#include <string>
#include <iostream>

class QCommandLine : public QLineEdit
{
    Q_OBJECT
signals:
    void commandEntered(QString cmd);
    void clearOutput();

public:
    QCommandLine(QWidget *parent = NULL)
        : QLineEdit(parent), m_browsingHistory(false) { }

    void keyPressEvent(QKeyEvent *event) {

// QT on Mac OS X Swaps Control/Command...
#ifdef Q_WS_MACX
        bool control = (event->modifiers() == Qt::MetaModifier);
#else
        bool control = (event->modifiers() == Qt::ControlModifier);
#endif
        int key = event->key();
        // Emacs-style keybinding
        QKeyEvent *remappedEvent = NULL;
        if (control && (key == Qt::Key_H)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Backspace,
                                          Qt::NoModifier);
        }
        else if (control && (key == Qt::Key_D)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Delete,
                                          Qt::NoModifier);
        }
        else if (control && (key == Qt::Key_P)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Up,
                                          Qt::KeypadModifier);
        }
        else if (control && (key == Qt::Key_N)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Down,
                                  Qt::KeypadModifier);
        }
        else if (control && (key == Qt::Key_B)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Left,
                                  Qt::KeypadModifier);
        }
        else if (control && (key == Qt::Key_F)) {
            remappedEvent = new QKeyEvent(event->type(), Qt::Key_Right,
                                  Qt::KeypadModifier);
        }
        else if (control && (key == Qt::Key_L)) {
            emit(clearOutput());
        }
        if (remappedEvent) {
            event = remappedEvent;
        }

        // Modifying the text buffer ends history search...
        m_browsingHistory &= !isModified();

        if (event->key() == Qt::Key_Up) {
            // Go backward in history
            if (m_browsingHistory) {
                History::iterator next = m_history_iterator;
                if (++next != m_history.end()) {
                    setText(*next);
                    m_history_iterator = next;
                }
            }
            else {
                m_currentString = text();
                if (m_history.size() > 0) {
                    m_history_iterator = m_history.begin();
                    setText(*m_history_iterator);
                    m_browsingHistory = true;
                }
            }
            setModified(false);
        }
        else if (event->key() == Qt::Key_Down) {
            // Go forward in history
            if (m_browsingHistory) {
                if (m_history_iterator != m_history.begin()) {
                    --m_history_iterator;
                    setText(*m_history_iterator);
                }
                else {
                    setText(m_currentString);
                    m_browsingHistory = false;
                }
                setModified(false);
            }
        }
        else if ((event->key() == Qt::Key_Return) ||
                 (event->key() == Qt::Key_Enter)) {
            QString cmd = text();
            clear();
            // TODO: Add to history
            emit commandEntered(cmd);
            m_history.push_front(cmd);
            setModified(false);
        }
        else{
            // default handler for event
            QLineEdit::keyPressEvent(event);
        }

        delete remappedEvent;
    }

private:
    typedef std::list<QString> History;
    History m_history;
    History::iterator m_history_iterator;
    QString m_currentString;
    bool m_browsingHistory;
};

#endif // QCOMMAND_LINE_HH
