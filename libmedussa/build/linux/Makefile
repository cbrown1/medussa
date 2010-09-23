ROOTDIR = ../..
LIBDIR = $(ROOTDIR)/lib
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src

CC = gcc

all: medussa.o
	gcc -shared -Wl,-soname,libmedussa.so -o libmedussa.so ./medussa.o -lportaudio -lsndfile

medussa.o: $(SRCDIR)/medussa.c
	gcc -c -fPIC $(SRCDIR)/medussa.c -o medussa.o -I/usr/include/python2.6 -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

clean:
	rm libmedussa.so
	rm medussa.o