Getting MeshFEM to run in Visual Studio 2019:

1.) This is specifically for VS2019; for older version one would need significant changes in the code
2.) Pre-requisites:
	- Download boost, extract it, and use the following to commands in the boost root folder in a terminal:
		bootsrap
		bjam --toolset=msvc-14.2 --build-type=complete --prefix=[installation dir]
	  More details on the installation can be found here: https://theboostcpplibraries.com/introduction-installation
	  Add the folder [installation dir] as an environment variable called BOOST_ROOT,
	  add the folder [installation dir]\lib as an environment variable called BOOST_LIBRARYDIR
	- Download or clone the CMake version of suitesparse from here: https://github.com/jlblancoc/suitesparse-metis-for-windows
		Configure and Generate the project files for Visual Studio 16 2019 - x64, there are some options,
		(like using CUDA and OpenMP) but these depend on your preferences.
		Just make sure you set the installation path (CMAKE_INSTALL_PREFIX) to something you remember
		Open the generated project file, build ALL_BUILD and then INSTALL.
		The libraries, shared libraries and include files will now be found in the installation path.
		To later be able to run executables without copying the .dlls into the program folder, you can
		add the subdirectories /bin and /lib64/lapack_blas_windows to your PATH environment variable.
		IMPORTANT: after building and installing you have to rename (or duplicate with different names)
		the following .lib libraries in the /lib subdirectory:
			libcholmod.lib -> cholmod.lib
			libccolamd.lib -> ccolamd.lib
			libcolamd.lib -> colamd.lib
			libamd.lib -> amd.lib
			libcamd.lib -> camd.lib
			libumfpack.lib -> umfpack.lib
		Now add the directory [suitesparse installation dir]/include/suitesparse as an environment
		variable called SUITESPARSE_INC, [suitesparse installation dir]/lib as SUITESPARSE_LIB
		and the itself [suitesparse installation dir] as SUITESPARSE_ROOT.
3.) Configure and generate the project files for MeshFEM
	- Aside from a warning, because of having BOOST_ROOT as an environment variable, this worked
	without any required changes, of course only after following the above rigorously
4.) Open the MeshFEM.sln and try to compile, you will get quite a lot of errors:
	- For some reason I had to manually add the include directory of boost for the MeshFEM project
		(these are found in the boost [installation dir]\include\boost_XX_X\)
	- There is a problem when compiling tinyexpr in Visual Studio ("initializer not costant")
	with a very simple workaround discribed here https://github.com/codeplea/tinyexpr/issues/34
		add these two lines anywhere above static const te_variable functions[] = {}
			#pragma function (ceil)
			#pragma function (floor)