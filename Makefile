include platform_defs.mk

CONVERT_OBJS=mesh_convert.o
PERHOMO_OBJS=TestPeriodicHomogenization.o
SOURCES=TestPeriodicHomogenization.cc mesh_convert.cc
TARGETS=mesh_convert TestPeriodicHomogenization

# CPPFLAGS+=-std=c++11 -O0 -fno-inline -g $(INCLUDES)
CPPFLAGS+=-std=c++11 -O2 $(INCLUDES)

all: $(TARGETS)

mesh_convert: $(CONVERT_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

TestPeriodicHomogenization: $(PERHOMO_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

depend:
	@touch Makefile.depend;
	makedepend -Y -f Makefile.depend --  -- $(SOURCES) &> /dev/null

clean:
	rm -f $(TARGETS) $(CONVERT_OBJS) $(PERHOMO_OBJS) *.bak

.PHONY: clean depend

# Read in the dependency file, if it exists
sinclude Makefile.depend
