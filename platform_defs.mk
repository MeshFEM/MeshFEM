# Read in definitions based on hostname
# Adapted from openFTL's build system
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
