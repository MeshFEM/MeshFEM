# MESHFEM build customization system:
# In general, library include/lib paths are configured using environment
# variables. These are of the format:
# LIBRARYNAME_INC
# LIBRARYNAME_LIB
# However, the names and dependencies of the shared libraries themselves vary
# from platform to platform. Thus, we use the platform-specific configuration
# files to create the following variables:
# LIBRARYNAME_LFLAGS
# These should look something like:
# LIBRARYNAME_LFLAGS=-L$(LIBRARYNAME_LIB) -lname ...
# These platform-specific config files are located in platform_config/(host).mk

# Read in definitions based on hostname
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# determine the hostname
HOST=$(shell hostname -s)

# Assume hostnames like "login-0-2" are the Mercer cluster
ifneq (,$(findstring login,$(HOST)))
    HOST=mercer
endif

# Assume hostnames like "compute-0-2" are the Mercer cluster
ifneq (,$(findstring compute,$(HOST)))
    HOST=mercer
endif

# The VGL cluster is all the same
VLG_HOSTS=banquo blakey cassio ceres django duncan horatio humair iago iris \
		  juno macbeth othello rose1 rose2 rose3 rose4 texier
ifneq (,$(findstring $(HOST), $(VLG_HOSTS)))
    HOST=vlg_cluster
endif

# The geoprocess machine in Pisa
ifneq (,$(findstring geoprocess,$(HOST)))
    HOST=ubuntu
endif

# The Linux_m3800 machine 
ifneq (,$(findstring Linux,$(HOST)))
    HOST=ubuntu
endif

ifneq (,$(findstring ubuntu,$(HOST)))
    HOST=ubuntu
endif

# directory of local definitions
LOCALDEFSDIR = $(TOP)/platform_config

# filename of local definitions file
LOCALDEFSFILE = $(LOCALDEFSDIR)/$(HOST).mk

# check existence of definitions file
ifneq ($(shell test -f $(LOCALDEFSFILE) && echo 'true'),true)
	LOCALDEFSFILE = $(LOCALDEFSDIR)/default.mk
endif

# inform user
$(info Using platform definitions $(LOCALDEFSFILE))

# include local definitions into this file
include $(LOCALDEFSFILE)
