#ifndef RESULT_TREE_VIEW_HH
#define RESULT_TREE_VIEW_HH

#include <QObject>
#include <QTreeWidget>
#include <QKeyEvent>

class ResultTreeView : public QTreeWidget
{
    Q_OBJECT

public:
    ResultTreeView(QWidget *parent = NULL)
        : QTreeWidget(parent) { }

    void keyPressEvent(QKeyEvent *event) {
        // Redirect Return key to control + o on mac so that it activates the
        // widget item as on other platforms.
        QKeyEvent *redirEvent = NULL;
#ifdef Q_WS_MACX
        if ((event->key() == Qt::Key_Return) ||
                (event->key() == Qt::Key_Enter)) {
            redirEvent = new QKeyEvent(event->type(), Qt::Key_O, Qt::ControlModifier);
            event = redirEvent;
        }
#endif
        QTreeWidget::keyPressEvent(event);

        delete redirEvent;
    }
};


#endif // RESULT_TREE_VIEW_HH
