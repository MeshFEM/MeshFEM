#ifndef MATLAB_INTERFACE_H
#define MATLAB_INTERFACE_H

#include <complex>
#include <cassert>
#include <map>
#include <vector>

// from Matlab
struct engine;
struct mxArray_tag;
typedef struct engine Engine;
typedef struct mxArray_tag mxArray;


class MatlabInterface
{
public:
    typedef std::complex<double> Complex;

    MatlabInterface();
    ~MatlabInterface();

    ////////////////////////////////////////////////////////////////////////////
    /*! Tell MATLAB to write command output to the passed character buffer.
    //  @param[in]  buffer  Pointer to a pre-allocated character buffer
    //  @param[in]  len     Length of the character buffer    
    *///////////////////////////////////////////////////////////////////////////
    void AttachOutputBuffer(char *buffer, size_t len);

    ////////////////////////////////////////////////////////////////////////////
    /*! Tell MATLAB to stop writting command output to the output buffer
    *///////////////////////////////////////////////////////////////////////////
    void AttachOutputBuffer();

    // Creates a matrix in matlab.
    template <typename T>
    void SetEngineIndexMatrix(const char *name, unsigned int m, unsigned int n, const T *vals, bool colmaj=false); // shifts indices by 1 (C/C++ vs matlab)
    template <typename T>
    void SetEngineRealMatrix(const char *name, unsigned int m, unsigned int n, const T *vals, bool colmaj=false);
    template <typename T>
    void SetEngineComplexMatrix(const char *name, unsigned int m, unsigned int n, const std::complex<T> *vals, bool colmaj=false);
    template <typename T>
    void SetEngineComplexMatrix(const char *name, const std::vector<std::vector<std::complex<T> > > &vals);

    template <typename IndexType, typename ValueType>
    void SetEngineEncodedSparseRealMatrix(const char *name, unsigned int n,
            const IndexType *rowind, const IndexType *colind, const ValueType *vals);

    template <typename IndexType, typename ValueType>
    void SetEngineSparseRealMatrix(const char *name, unsigned int n,
            const IndexType *rowind, const IndexType *colind, const ValueType *vals, unsigned int nrows=0, unsigned int ncols=0);

    template <typename IndexType, typename ValueType>
    void SetEngineEncodedSparseComplexMatrix(const char *name, unsigned int n,
            const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals);

    template <typename IndexType, typename ValueType>
    void SetEngineSparseComplexMatrix(const char *name, unsigned int n,
            const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals, unsigned int nrows=0, unsigned int ncols=0);

    // Reads a matrix from matlab.
    template <typename T>
    void GetEngineIndexMatrix(const char *name, unsigned int m, unsigned int n, T *vals, bool colmaj=false); // shifts indices by 1 (C/C++ vs matlab)
    template <typename T>
    void GetEngineRealMatrix(const char *name, unsigned int m, unsigned int n, T *vals, bool colmaj=false);
    template <typename T>
    void GetEngineComplexMatrix(const char *name, unsigned int m, unsigned int n, std::complex<T> *vals, bool colmaj=false);

    void GetEncodedSparseRealMatrix(const char* name, unsigned int*& rowind, unsigned int*& colind, double*& vals, unsigned int& nentries);

    // Executes a matlab script. Returns non-zero on error.
    int RunScript(const char *fname);

    // Eval in-place string
    int Eval(const char *matlab_code);

private:
    // Note that the arrays created by these matrices *must be destroyed*
    // afterward.
    // matrix creation helper functions
    template <typename T>
    static mxArray* CreateIndexMatrix(unsigned int m, unsigned int n, const T *vals, bool colmaj=false); // shifts indices by 1 (C/C++ vs matlab)
    template <typename T>
    static mxArray* CreateRealMatrix(unsigned int m, unsigned int n, const T *vals, bool colmaj=false);
    template <typename T>
    static mxArray* CreateComplexMatrix(unsigned int m, unsigned int n, const std::complex<T> *vals, bool colmaj=false);

    template <typename IndexType, typename ValueType>
    static mxArray* CreateEncodedSparseRealMatrix(unsigned int n,
            const IndexType *rowind, const IndexType *colind, const ValueType *vals);

    template <typename IndexType, typename ValueType>
    static mxArray* CreateEncodedSparseComplexMatrix(unsigned int n,
            const IndexType *rowind, const IndexType *colind, const std::complex<ValueType> *vals);

    // matrix copy-back helper functions
    template <typename T>
    static void CopyFromIndexMatrix(mxArray *M, unsigned int m, unsigned int n, T *dest, bool colmaj=false); // shifts indices by 1 (C/C++ vs matlab)
    template <typename T>
    static void CopyFromRealMatrix(mxArray *M, unsigned int m, unsigned int n, T *dest, bool colmaj=false);
    template <typename T>
    static void CopyFromComplexMatrix(mxArray *M, unsigned int m, unsigned int n, std::complex<T> *dest, bool colmaj=false);


private:
    Engine *m_ep;
};

#endif /* MATLAB_INTERFACE_H */

