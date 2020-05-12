QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++1z

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    tracker/fern.cpp \
    tracker/fern_fext.cpp \
    tracker/integrator.cpp \
    tracker/object_detector.cpp \
    tracker/object_model.cpp \
    tracker/opt_flow_tracker.cpp \
    tracker/scanning_grid.cpp \
    tracker/tld_tracker.cpp \
    tracker/tld_utils.cpp

HEADERS += \
    mainwindow.h \
    profile.h \
    test_runner.h \
    tracker/augmentator.h \
    tracker/common.h \
    tracker/fern.h \
    tracker/fern_fext.h \
    tracker/integrator.h \
    tracker/object_classifier.h \
    tracker/object_detector.h \
    tracker/object_model.h \
    tracker/opt_flow_tracker.h \
    tracker/scanning_grid.h \
    tracker/tld_tracker.h \
    tracker/tld_utils.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_videoio \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_features2d \
        -lopencv_video \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
