ROOTDIR = ../..
LIBDIR = $(ROOTDIR)/lib
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src

CC = gcc

all: medussa.o
	gcc -shared -Wl,-soname,libmedussa.so -o libmedussa.so ./medussa.o -lportaudio -lsndfile

medussa.o: $(SRCDIR)/medussa.c medussa_callbacks.o medussa_matrix.o medussa_types.o
	gcc -c -fPIC $(SRCDIR)/medussa.c -o medussa.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

medussa_callbacks.o: $(SRCDIR)/medussa_callbacks.c $(SRCDIR)/randomkit.o
	gcc -c $(SRCDIR)/medussa_callbacks.c -o medussa_callbacks.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

medussa_matrix.o: $(SRCDIR)/medussa_matrix.c
	gcc -c $(SRCDIR)/medussa_matrix.c -o medussa_matrix.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

medussa_types.o: $(SRCDIR)/medussa_types.c
	gcc -c $(SRCDIR)/medussa_types.c -o medussa_types.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

randomkit.o: $(SRCDIR)/randomkit.c
	gcc -c $(SRCDIR)/randomkit.c -o randomkit.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

clean:
	rm libmedussa.so
	rm medussa.o
	rm medussa_callbacks.o
	rm medussa_types.o
	rm medussa_matrix.o
