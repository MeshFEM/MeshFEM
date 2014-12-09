include ../platform_defs.mk

CPPFLAGS+=-Wall -pedantic -std=c++11 $(INCLUDES) -I..
CPPFLAGS+=-O2
# CPPFLAGS+=-O0 -g
TARGETS=msh_processor selector

MSH_PROC_OBJS=msh_processor.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
SELECTOR_OBJS=selector.o ../MeshIO.o ../MSHFieldParser.o ../Types.o

all: $(TARGETS)

selector: $(SELECTOR_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

msh_processor: $(MSH_PROC_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm $(TARGETS)
