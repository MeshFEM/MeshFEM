#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>

#include "MatlabInterface.h"

#include "engine.h"  // Matlab engine header

#include <sys/stat.h>
// from http://www.codeproject.com/KB/files/filesize.aspx
static long FileSize(const char* fname)
{
    struct stat finfo;
    if (stat(fname, &finfo) == 0)
        return finfo.st_size;
    else
        return 0;
}

MatlabInterface::MatlabInterface()
{
    // Start the MATLAB engine locally by executing the string
    // "matlab"
    //
    // To start the session on a remote host, use the name of
    // the host as the string rather than \0
    //
    // For more complicated cases, use any string with whitespace,
    // and that string will be executed literally to start MATLAB
    //
    if (!(m_ep = engOpen("\0"))) {
        std::cerr << "\nCan't start MATLAB engine\n";
        exit(EXIT_FAILURE);
    }

#ifdef WIN32
    // Try to open a MATLAB session window
    if (engSetVisible(m_ep, 1))
        std::cout << "engSetVisible worked" << std::endl;
    else
        std::cout << "engSetVisible failed" << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/*! Tell MATLAB to write command output to the passed character buffer.
//  @param[in]  buffer  Pointer to a pre-allocated character buffer
//  @param[in]  len     Length of the character buffer    
*///////////////////////////////////////////////////////////////////////////////
void MatlabInterface::AttachOutputBuffer(char *buffer, size_t len)
{
    engOutputBuffer(m_ep, buffer, len);
}

////////////////////////////////////////////////////////////////////////////////
/*! Tell MATLAB to stop writting command output to the output buffer
*///////////////////////////////////////////////////////////////////////////////
void MatlabInterface::AttachOutputBuffer()
{
    engOutputBuffer(m_ep, NULL, 0);
}



MatlabInterface::~MatlabInterface()
{
    engClose(m_ep);
}



// This also shifts all the values by 1 to account for the difference
// between matlab and C/C++ indexing
template <typename T>
mxArray*
MatlabInterface::CreateIndexMatrix(unsigned int m, unsigned int n, const T *vals, bool colmaj)
{
    mxArray *M = mxCreateDoubleMatrix(m, n, mxREAL);
    double *pM = mxGetPr(M);
    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            pM[idxM] = double(vals[idx]+1);
        }
    return M;
}



template <typename T>
mxArray*
MatlabInterface::CreateRealMatrix(unsigned int m, unsigned int n, const T *vals, bool colmaj)
{
    mxArray *M = mxCreateDoubleMatrix(m, n, mxREAL);
    double *pM = mxGetPr(M);
    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            pM[idxM] = double(vals[idx]);
        }
    return M;
}



template <typename T>
mxArray*
MatlabInterface::CreateComplexMatrix(unsigned int m, unsigned int n, const std::complex<T> *vals, bool colmaj)
{
    mxArray *M = mxCreateDoubleMatrix(m, n, mxCOMPLEX);
    double *pMr = mxGetPr(M);
    double *pMi = mxGetPi(M);
    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            pMr[idxM] = double(vals[idx].real());
            pMi[idxM] = double(vals[idx].imag());
        }
    return M;
}



template <typename IndexType, typename ValueType>
mxArray*
MatlabInterface::CreateEncodedSparseRealMatrix(unsigned int n,
        const IndexType *rowind, const IndexType *colind, const ValueType *vals)
{
    mxArray *M = mxCreateDoubleMatrix(n, 3, mxREAL);
    double *pM = mxGetPr(M);
    // note that matlab expects the data in column-major order
    for (unsigned int i = 0; i < n; ++i)
    {
        pM[0*n+i] = double(rowind[i]+1);
        pM[1*n+i] = double(colind[i]+1);
        pM[2*n+i] = double(  vals[i]  );
    }
    return M;
}



template <typename IndexType, typename ValueType>
mxArray*
MatlabInterface::CreateEncodedSparseComplexMatrix(unsigned int n,
        const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals)
{
    mxArray *M = mxCreateDoubleMatrix(n, 3, mxCOMPLEX);
    double *pMr = mxGetPr(M);
    double *pMi = mxGetPi(M);
    // note that matlab expects the data in column-major order
    for (unsigned int i = 0; i < n; ++i)
    {
        pMr[0*n+i] = double(rowind[i]+1); pMi[0*n+i] = double(0);
        pMr[1*n+i] = double(colind[i]+1); pMi[1*n+i] = double(0);
        pMr[2*n+i] = double(vals[i].real());
        pMi[2*n+i] = double(vals[i].imag());
    }
    return M;
}


// This also shifts all the values by 1 to account for the difference
// between matlab and C/C++ indexing
template <typename T>
void
MatlabInterface::CopyFromIndexMatrix(mxArray *M, unsigned int m, unsigned int n, T *dest, bool colmaj)
{
    assert(mxGetM(M) == m);
    assert(mxGetN(M) == n);
    double *pM = mxGetPr(M);
    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            dest[idx] = T(pM[idxM]-1);
        }
}

template <typename T>
void
MatlabInterface::CopyFromRealMatrix(mxArray *M, unsigned int m, unsigned int n, T *dest, bool colmaj)
{
    assert(mxGetM(M) == m);
    assert(mxGetN(M) == n);
    double *pM = mxGetPr(M);
    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            dest[idx] = T(pM[idxM]);
        }
}


template <typename T>
void
MatlabInterface::CopyFromComplexMatrix(mxArray *M, unsigned int m, unsigned int n, std::complex<T> *dest, bool colmaj)
{
    assert(mxGetM(M) == m);
    assert(mxGetN(M) == n);
    double *pMr = mxGetPr(M);
    double *pMi = mxGetPi(M);
    bool pure_real = (pMi == NULL); // seems to be NULL when the matrix doesn't have complex values

    // note that matlab expects the data in column-major order
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < m; ++i)
        {
            unsigned int idxM = j*m+i;
            unsigned int idx = colmaj ? idxM : i*n+j;
            dest[idx] = Complex(T(pMr[idxM]), pure_real ? T(0) : T(pMi[idxM]));
        }
}



// Creates a matrix in the Matlab engine.
template <typename T>
void
MatlabInterface::SetEngineIndexMatrix(const char *name, unsigned int m, unsigned int n, const T *vals, bool colmaj) // shifts indices by 1 (C/C++ vs matlab)
{
    mxArray *ary = CreateIndexMatrix(m, n, vals, colmaj);
    assert(ary);
    engPutVariable(m_ep, name, ary);
    mxDestroyArray(ary);
}

template <typename T>
void
MatlabInterface::SetEngineRealMatrix(const char *name, unsigned int m, unsigned int n, const T *vals, bool colmaj)
{
    mxArray *ary = CreateRealMatrix(m, n, vals, colmaj);
    assert(ary);
    engPutVariable(m_ep, name, ary);
    mxDestroyArray(ary);
}

template <typename T>
void
MatlabInterface::SetEngineComplexMatrix(const char *name, unsigned int m, unsigned int n, const std::complex<T> *vals, bool colmaj)
{
    mxArray *ary = CreateComplexMatrix(m, n, vals, colmaj);
    assert(ary);
    engPutVariable(m_ep, name, ary);
    mxDestroyArray(ary);
}

template <typename T>
void MatlabInterface::SetEngineComplexMatrix(const char *name
        , const std::vector<std::vector<std::complex<T> > > &mat)
{
    typedef std::complex<T> CType;
    // Construct flattened row-major matrix
    unsigned int m = mat.size();
    unsigned int n = (m == 0) ? 0 : mat[0].size();

    CType *vals = new CType[m * n];
    assert(vals != NULL);
    CType *curr = vals;

    for (unsigned int i = 0; i < m; ++i)	{
        const std::vector<CType> &row = mat[i];
        assert(row.size() == n); // All the rows should be the same size
        for (unsigned int j = 0; j < n; ++j)
            *(curr++) = row[j];
    }

    // Move the flattened matrix into MATLAB
    SetEngineComplexMatrix(name, m, n, vals, false);
    delete[] vals;
}

template <typename IndexType, typename ValueType>
void
MatlabInterface::SetEngineEncodedSparseRealMatrix(const char *name, unsigned int n,
        const IndexType *rowind, const IndexType *colind, const ValueType *vals)
{
    mxArray *ary = CreateEncodedSparseRealMatrix(n, rowind, colind, vals);
    assert(ary);
    engPutVariable(m_ep, name, ary);
    mxDestroyArray(ary);
}

template <typename IndexType, typename ValueType>
void
MatlabInterface::SetEngineSparseRealMatrix(const char *name, unsigned int n,
        const IndexType *rowind, const IndexType *colind, const ValueType *vals, unsigned int nrows, unsigned int ncols)
{
    SetEngineEncodedSparseRealMatrix(name, n, rowind, colind, vals);
    char cmd[1024];
//    if(ncols == 0 || nrows == 0) { 
//      sprintf(cmd, "%s = sparse(%s(:,1), %s(:,2), %s(:,3),)", name, name, name, name);
//} else { 
//      assert(ncols > 0 && nrows > 0); 
      sprintf(cmd, "%s = sparse(%s(:,1), %s(:,2), %s(:,3), %d, %d)", name, name, name, name, nrows, ncols);
//    }
    engEvalString(m_ep, cmd);
}

template <typename IndexType, typename ValueType>
void
MatlabInterface::SetEngineEncodedSparseComplexMatrix(const char *name, unsigned int n,
        const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals)
{
    mxArray *ary = CreateEncodedSparseComplexMatrix(n, rowind, colind, vals);
    assert(ary);
    engPutVariable(m_ep, name, ary);
    mxDestroyArray(ary);
}

template <typename IndexType, typename ValueType>
void
MatlabInterface::SetEngineSparseComplexMatrix(const char *name, unsigned int n,
        const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals, unsigned int nrows, unsigned int ncols)
{
    SetEngineEncodedSparseComplexMatrix(name, n, rowind, colind, vals);
    char cmd[1024];
    //sprintf(cmd, "%s = spconvert(%s)", name, name);
    if(ncols == 0 || nrows == 0) { 
      sprintf(cmd, "%s = sparse(%s(:,1), %s(:,2), %s(:,3), %d, %d)", name, name, name, name, nrows, ncols);
    }
    engEvalString(m_ep, cmd);
}

// Reads a matrix from the matlab engine.
template <typename T>
void
MatlabInterface::GetEngineIndexMatrix(const char *name, unsigned int m, unsigned int n, T *dest, bool colmaj)
{
    mxArray *ary = engGetVariable(m_ep, name);
    assert(ary);
    CopyFromIndexMatrix(ary, m, n, dest, colmaj);
    mxDestroyArray(ary);
}

template <typename T>
void
MatlabInterface::GetEngineRealMatrix(const char *name, unsigned int m, unsigned int n, T *dest, bool colmaj)
{
    mxArray *ary = engGetVariable(m_ep, name);
    assert(ary);
    CopyFromRealMatrix(ary, m, n, dest, colmaj);
    mxDestroyArray(ary);
}


template <typename T>
void
MatlabInterface::GetEngineComplexMatrix(const char *name, unsigned int m, unsigned int n, std::complex<T> *dest, bool colmaj)
{
    mxArray *ary = engGetVariable(m_ep, name);
    assert(ary);
    CopyFromComplexMatrix(ary, m, n, dest, colmaj);
    mxDestroyArray(ary);
}


// Executes a matlab script.
// Returns non-zero on error.
int
MatlabInterface::RunScript(const char *fname)
{
    long size = FileSize(fname);
    if (size <= 0) {
        std::cerr << "ERROR: Matlab script \"" << fname << "\" has size 0\n";
        return -1;
    }
    std::ifstream fin(fname, std::ios::in|std::ios::binary);
    if (!fin.good() || !fin.is_open()) {
        std::cerr << "ERROR: Unable to read from matlab script \"" << fname << "\"\n";
        return -2;
    }

    char *code = new char[size+1];
    fin.read(code, size);
    code[size] = '\0';
    int res;
    res = engEvalString(m_ep, code);
    delete [] code;
    if (res != 0)
        std::cerr << "ERROR: Error running matlab script \"" << fname << "\"\n";
    return res;
}

// Evaluate a single line
// Returns non-zero on error.
int
MatlabInterface::Eval(const char *matlab_code)
{
    int res = engEvalString(m_ep, matlab_code);
    if (res != 0)
        std::cerr << "ERROR: Error running matlab command \"" << matlab_code << "\"\n";
    return res;
}

void MatlabInterface::GetEncodedSparseRealMatrix(const char* name, unsigned int*& rowind, unsigned int*& colind, double*& vals, 
                                                 unsigned int& nentries) { 

    mxArray *M = engGetVariable(m_ep, name);
  
    if(M == 0) { 
      std::cerr << "warning: matrix " << name << " could not be loaded " << std::endl;
      return; 
    }
    nentries =  mxGetM(M);
    int ncols =  mxGetN(M);
    assert(ncols == 3); 
    double *pM = mxGetPr(M);
    rowind = new unsigned int[nentries]; 
    colind = new unsigned int[nentries];
    vals   = new       double[nentries];

    for (unsigned int i = 0; i < nentries; ++i)
    {
        rowind[i] = (unsigned int)(pM[           i]) - 1; assert(rowind[i] >= 0); 
        colind[i] = (unsigned int)(pM[  nentries+i]) - 1; assert(colind[i] >= 0); 
        vals  [i] = (double      )(pM[2*nentries+i]);
    }
}

////////////////////////////////////////////////////////////
// Explicit instantiation of exported template methods

// Creates a matrix in matlab.
template void MatlabInterface::SetEngineIndexMatrix         <int>(const char *name, unsigned int m, unsigned int n, const                  int *vals, bool colmaj); // shifts indices by 1 (C/C++ vs matlab)
template void MatlabInterface::SetEngineIndexMatrix<unsigned int>(const char *name, unsigned int m, unsigned int n, const         unsigned int *vals, bool colmaj); // shifts indices by 1 (C/C++ vs matlab)
template void MatlabInterface::SetEngineRealMatrix       <double>(const char *name, unsigned int m, unsigned int n, const               double *vals, bool colmaj);
template void MatlabInterface::SetEngineComplexMatrix    <double>(const char *name, unsigned int m, unsigned int n, const std::complex<double> *vals, bool colmaj);
template void MatlabInterface::SetEngineComplexMatrix    <double>(const char *name, const std::vector<std::vector<std::complex<double> > > &vals);
template void MatlabInterface::SetEngineComplexMatrix    <float> (const char *name, const std::vector<std::vector<std::complex<float > > > &vals);

template void MatlabInterface::SetEngineRealMatrix        <float>(const char *name, unsigned int m, unsigned int n, const                float *vals, bool colmaj);

template
void MatlabInterface::SetEngineEncodedSparseRealMatrix<unsigned int, double>(const char *name, unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const double *vals);

template
void MatlabInterface::SetEngineSparseRealMatrix<int, double>(const char *name, unsigned int n,
        const int *rowind, const int *colind, const double *vals, unsigned int nrows, unsigned int ncols);
template
void MatlabInterface::SetEngineSparseRealMatrix<unsigned int, double>(const char *name, unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const double *vals, unsigned int nrows, unsigned int ncols);
template
void MatlabInterface::SetEngineSparseRealMatrix<size_t, float>(const char *name, unsigned int n,
        const size_t *rowind, const size_t *colind, const float *vals, unsigned int nrows, unsigned int ncols);
template
void MatlabInterface::SetEngineSparseRealMatrix<size_t, double>(const char *name, unsigned int n,
        const size_t *rowind, const size_t *colind, const double *vals, unsigned int nrows, unsigned int ncols);

template
void MatlabInterface::SetEngineEncodedSparseComplexMatrix<unsigned int, double>(const char *name, unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const std::complex<double> *vals);

template
void MatlabInterface::SetEngineSparseComplexMatrix<unsigned int, double>(const char *name, unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const std::complex<double> *vals, unsigned int nrows, unsigned int ncols);

// Reads a matrix from matlab.
template void MatlabInterface::GetEngineIndexMatrix<unsigned int>(const char *name, unsigned int m, unsigned int n,         unsigned int *vals, bool colmaj); // shifts indices by 1 (C/C++ vs matlab)
template void MatlabInterface::GetEngineRealMatrix       <double>(const char *name, unsigned int m, unsigned int n,               double *vals, bool colmaj);
template void MatlabInterface::GetEngineRealMatrix       <float >(const char *name, unsigned int m, unsigned int n,               float  *vals, bool colmaj);
template void MatlabInterface::GetEngineComplexMatrix    <double>(const char *name, unsigned int m, unsigned int n, std::complex<double> *vals, bool colmaj);

// matrix creation helper functions
template mxArray* MatlabInterface::CreateIndexMatrix<unsigned int>(unsigned int m, unsigned int n, const         unsigned int *vals, bool colmaj); // shifts indices by 1 (C/C++ vs matlab)
template mxArray* MatlabInterface::CreateRealMatrix       <double>(unsigned int m, unsigned int n, const               double *vals, bool colmaj);
template mxArray* MatlabInterface::CreateComplexMatrix    <double>(unsigned int m, unsigned int n, const std::complex<double> *vals, bool colmaj);

template
mxArray* MatlabInterface::CreateEncodedSparseRealMatrix<unsigned int, double>(unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const double *vals);

template
mxArray* MatlabInterface::CreateEncodedSparseComplexMatrix<unsigned int, double>(unsigned int n,
        const unsigned int *rowind, const unsigned int *colind, const std::complex<double> *vals);

// matrix copy-back helper functions
template void MatlabInterface::CopyFromIndexMatrix<unsigned int>(mxArray *M, unsigned int m, unsigned int n,         unsigned int *dest, bool colmaj); // shifts indices by 1 (C/C++ vs matlab)
template void MatlabInterface::CopyFromRealMatrix       <double>(mxArray *M, unsigned int m, unsigned int n,               double *dest, bool colmaj);
template void MatlabInterface::CopyFromComplexMatrix    <double>(mxArray *M, unsigned int m, unsigned int n, std::complex<double> *dest, bool colmaj);

////////////////////////////////////////////////////////////

