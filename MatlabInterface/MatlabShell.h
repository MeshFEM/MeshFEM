////////////////////////////////////////////////////////////////////////////////
// MatlabShell.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a minimal shell that can be used to communicate with MATLAB.
//      This shell is needed on non-Windows platforms because the MATLAB engine
//      can't open an interactive window. The shell will likely fail on Windows
//      and should not be used.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  11/17/2010 01:43:11
//  Revision History:
//      11/17/2010  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////
#ifndef MATLAB_SHELL_H
#define MATLAB_SHELL_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <readline/readline.h>
#include <readline/history.h>

#include "MatlabInterface.h"

class MatlabShell : public MatlabInterface
{
    private:
        /** MATLAB output buffer  */
        char           *m_outputBuffer;
        /** Size of MATLAB output buffer  */
        int             m_bufferSize;

    public:
        ////////////////////////////////////////////////////////////////////////
        /*! Opens a new connection with MATLAB and attaches a new output buffer
        //  to it.
        //  @param[in]  bufferSize  Size of output buffer (defaults to 4k)
        *///////////////////////////////////////////////////////////////////////
        MatlabShell(int bufferSize = 4 * 1024)
            : MatlabInterface(bufferSize) { }

        ////////////////////////////////////////////////////////////////////////
        /*! Runs the interactive shell loop, presenting a prompt and passing
        //  commands to MATLAB. Run() returns when the user types Control-D or
        //  "exit"
        *///////////////////////////////////////////////////////////////////////
        void run()
        {
            char *cmd;
            while ((cmd = readline(">> ")))   {
                if (cmd[0])    {
                    add_history(cmd);
                    bool shouldExit = extractExitCommand(cmd);
                    std::string outputString, errorString;
                    Eval(cmd, outputString, errorString);

                    printf("%s", errorString.c_str());
                    printf("%s", skipPromptGarbage(outputString.c_str()));

                    if (shouldExit)
                        break;
                }

                free(cmd);
            }

            printf("\n");
        }

        ////////////////////////////////////////////////////////////////////////
        /*! Remove the exit command and any subsequent commands from a line,
        //  leaving the preceeding commands intact.
        //  @param[inout]   cmd     input command, trimmed command destination
        //  @return     true if the line contained an exit command
        *///////////////////////////////////////////////////////////////////////
        bool extractExitCommand(char *cmd)
        {
            bool exiting = false;

            int cmdLen = strlen(cmd);

            char *cmdRemainder = (char *) malloc(cmdLen + 1);
            strcpy(cmdRemainder, cmd);

            int trimmedLen = 0;
            int semicolonFinder = -1;
            int numMatched = 2;

            while (numMatched == 2)
            {
                semicolonFinder = -1;
                numMatched = sscanf(cmdRemainder, " %[^;] ;%n %[^\n]"
                        , &cmd[trimmedLen]
                        , &semicolonFinder
                        , cmdRemainder);

                int exitFinder = 0;
                sscanf(&cmd[trimmedLen], " exit %n", &exitFinder);

                if (exitFinder > 0) {
                    // Strip exit command from sequence
                    cmd[trimmedLen] = 0;
                    exiting = true;
                    break;
                }

                trimmedLen = strlen(cmd);

                if (semicolonFinder > 0)    {
                    cmd[trimmedLen++] = ';';
                    cmd[trimmedLen] = 0;
                }
            }

            free(cmdRemainder);

            return exiting;
        }
};

#endif
