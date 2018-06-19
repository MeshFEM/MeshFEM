#! /bin/bash
#
# Submit job as (build defaults to Release):
#
#   sbatch compile.sh
#   sbatch --export=BUILD='Debug',ALL compile.sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=16GB

# Load modules
module purge
module load mercurial/intel/4.0.1
module load gcc/6.3.0
module load cmake/intel/3.7.1

export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

# MeshFEM code
module load boost/intel/1.62.0
module load suitesparse/intel/4.5.4

# Run job
cd "${SLURM_SUBMIT_DIR}"

# Compile main program
mkdir build
cd build
echo ${BUILD}

if [ -z "${BUILD}" ]; then
	BUILD=Release
fi

mkdir ${BUILD}
pushd ${BUILD}
cmake -DCMAKE_BUILD_TYPE=${BUILD} ../..
make -j8
popd
