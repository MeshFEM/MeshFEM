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

# ## Catamari
# DISABLED until BlockCatamari is released.
# function(meshfem_download_catamari)
#     meshfem_download_project(catamari
#         GIT_REPOSITORY https://github.com/MeshFEM/catamari.git
#         GIT_TAG        23b858e18f639261bae4a2fe15d00a286cda963d)
# endfunction()

## catamari_legacy
function(meshfem_download_catamari_legacy)
    meshfem_download_project(catamari_legacy
        GIT_REPOSITORY https://github.com/MeshFEM/catamari.git
        GIT_TAG        2f427c264cc42bf31b1cc23fc7f34d904f130384)
endfunction()
