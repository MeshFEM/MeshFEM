################################################################################
include(MeshFEMCoreDownloadExternal)

################################################################################

## Catch2
function(meshfem_download_catch)
    meshfem_download_project(Catch2
        URL     https://github.com/catchorg/Catch2/archive/v2.13.10.tar.gz
        URL_MD5 7a4dd2fd14fb9f46198eb670ac7834b7
    )
endfunction()

## Ceres
function(meshfem_download_ceres)
    meshfem_download_project(ceres
        GIT_REPOSITORY https://github.com/jdumas/ceres-solver.git
        GIT_TAG        2ba66a2c22959d9c455a8f2074dc7a605c4a92e8
    )
endfunction()

## Json
function(meshfem_download_json)
    meshfem_download_project(json
        URL https://github.com/nlohmann/json/releases/download/v3.10.1/include.zip
        URL_HASH SHA256=144268f7f85afb0f0fbea7c796723c849724c975f9108ffdadde9ecedaa5f0b1
    )
endfunction()

## Optional
function(meshfem_download_optional)
    meshfem_download_project(optional
        URL     https://github.com/martinmoene/optional-lite/archive/v3.0.0.tar.gz
        URL_MD5 a66541380c51c0d0a1e593cc2ca9fe8a
    )
endfunction()

## Tinyexpr
function(meshfem_download_tinyexpr)
    meshfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
        GIT_TAG        ffb0d41b13e5f8d318db95feb071c220c134fe70
    )
endfunction()

## Triangle
function(meshfem_download_triangle)
    meshfem_download_project(triangle
        GIT_REPOSITORY https://github.com/libigl/triangle.git
        GIT_TAG        3ee6cac2230f0fe1413879574f741c7b6da11221
    )
endfunction()

## Spectra
function(meshfem_download_spectra)
    meshfem_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        8c7242e08f0fb7f6a0022cfe232e3dc5b5bd4eb4
    )
endfunction()

## IPC-Toolkit
function(meshfem_download_ipc_toolkit)
    meshfem_download_project(ipc_toolkit
        GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
        GIT_TAG        e3707832f83f5576c07e3bb9f748a4c75835ca85
    )
endfunction()

## TinyAD
function(meshfem_download_tinyad)
    meshfem_download_project(TinyAD
        GIT_REPOSITORY https://github.com/jpanetta/TinyAD
        GIT_TAG edeb8cd5a978413ce10ad42092a666ca43aec663
    )
endfunction()
