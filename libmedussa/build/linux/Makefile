ROOTDIR = ../..
LIBDIR = $(ROOTDIR)/lib
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
FLAGS = -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

CC = gcc

all: medussa.o
	$(CC) -shared -Wl,-soname,libmedussa.so -o libmedussa.so ./*.o -lportaudio -lsndfile

medussa.o: $(SRCDIR)/medussa.c medussa_callbacks.o medussa_matrix.o medussa_types.o randomkit.o
	$(CC) -c -fPIC $(SRCDIR)/medussa.c $(FLAGS)

medussa_callbacks.o: $(SRCDIR)/medussa_callbacks.c $(SRCDIR)/randomkit.o
	$(CC) -c -fPIC $(SRCDIR)/medussa_callbacks.c $(FLAGS)

medussa_matrix.o: $(SRCDIR)/medussa_matrix.c
	$(CC) -c -fPIC $(SRCDIR)/medussa_matrix.c $(FLAGS)

medussa_types.o: $(SRCDIR)/medussa_types.c
	$(CC) -c -fPIC $(SRCDIR)/medussa_types.c $(FLAGS)

randomkit.o: $(SRCDIR)/randomkit.c
	$(CC) -c -fPIC $(SRCDIR)/randomkit.c $(FLAGS)

clean:
	rm libmedussa.so
	rm medussa.o
	rm medussa_callbacks.o
	rm medussa_types.o
	rm medussa_matrix.o
	rm randomkit.o
