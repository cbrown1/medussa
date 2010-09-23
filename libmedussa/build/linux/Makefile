ROOTDIR = ../..
LIBDIR = $(ROOTDIR)/lib
INCLUDEDIR = $(ROOTDIR)/lib
SRCDIR = $(ROOTDIR)/src

CC = gcc

all: medussa.o
	echo "building all"
	gcc -shared -Wl,-soname,libmedussa.so -o libmedussa.so ./medussa.o

medussa.o: 
	echo "building `medusa.o`..."
	gcc -c -fPIC $(SRCDIR)/medussa.c ./medussa.o -I/usr/include/python2.6 -lportaudio

