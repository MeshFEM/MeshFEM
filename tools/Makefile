include ../platform_defs.mk

CPPFLAGS+=-Wall -pedantic -std=c++11 $(INCLUDES) -I..
CPPFLAGS+=-O2
TARGETS=msh_processor

MSH_PROC_OBJS=msh_processor.o ../MeshIO.o ../MSHFieldParser.o ../Types.o

all: $(TARGETS)

msh_processor: $(MSH_PROC_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm $(TARGETS)
