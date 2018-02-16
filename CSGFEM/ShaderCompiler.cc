////////////////////////////////////////////////////////////////////////////////
// ShaderCompiler.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Compiles glsl shaders into GPU programs.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/09/2012 23:45:35
////////////////////////////////////////////////////////////////////////////////
#include "ShaderCompiler.hh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! Print an info log on the creation of a vertex or fragment shader
//  @param[in]  obj   which shader whose status to query
*///////////////////////////////////////////////////////////////////////////////
void printShaderInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    char *infoLog;
    
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    
    if (infologLength > 0) {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Print an info log on the creation of a GPU program
//  @param[in]  obj   which program whose status to query
*///////////////////////////////////////////////////////////////////////////////
void printProgramInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    char *infoLog;
    
    glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    
    if (infologLength > 0) {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Read a vertex and fragment shader sources and compile them into a program.
//  @param[in]  vertProgFilename    vertex shader source file
//  @param[in]  fragProgFilename    fragment shader source
//  @param[out] program             shader program index
//  @return     true on success
*///////////////////////////////////////////////////////////////////////////////
bool readShader(const string &vertProgFilename,
                const string &fragProgFilename,
                GLuint &program)
{
    string vertProgramSource, fragProgramSource;
    ifstream vertProgFile(vertProgFilename.c_str());
    if (!vertProgFile) {
        cerr << "Error opening vertex shader program file." << endl;
        return false;
    }

    ifstream fragProgFile(fragProgFilename.c_str());
    if (!fragProgFile) {
        cerr << "Error opening fragment shader program file." << endl;
        return false;
    }
    
    getline(vertProgFile, vertProgramSource, '\0');
    getline(fragProgFile, fragProgramSource, '\0');
    
    GLuint vertShader, fragShader;
    program = glCreateProgram();
    const char *vertSource = vertProgramSource.c_str();
    const char *fragSource = fragProgramSource.c_str();
    
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertSource, NULL);
    glCompileShader(vertShader);
    cerr << "Compiling " << vertProgFilename << "..." << endl;
    printShaderInfoLog(vertShader);
    
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragSource, NULL);
    glCompileShader(fragShader);
    cerr << "Compiling " << fragProgFilename << "..." << endl;
    printShaderInfoLog(fragShader);
    
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);
    cerr << "Enabling program." << endl;
    printProgramInfoLog(program);

    return true;
}

