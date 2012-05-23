ROOTDIR = ../..
LIBDIR = $(ROOTDIR)/lib
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
override FLAGS += -Wall -L/usr/lib -L/usr/local/lib -lportaudio -lsndfile

CC = gcc

all: libmedussa.so

libmedussa.so: libmedussa.so.1
	ln -s libmedussa.so.1 libmedussa.so

libmedussa.so.1: libmedussa.so.1.0.1
	ln -s libmedussa.so.1.0.1 libmedussa.so.1

libmedussa.so.1.0.1: medussa.o
	gcc -shared -Wl,-soname,libmedussa.so -o libmedussa.so.1.0.1 medussa.o medussa_callbacks.o medussa_matrix.o medussa_types.o randomkit.o pink.o pa_ringbuffer.o $(FLAGS)

medussa.o: $(SRCDIR)/medussa.c medussa_callbacks.o medussa_matrix.o medussa_types.o randomkit.o pink.o pa_ringbuffer.o
	gcc -c -fPIC $(SRCDIR)/medussa.c $(FLAGS) -o medussa.o

medussa_callbacks.o: $(SRCDIR)/medussa_callbacks.c $(SRCDIR)/randomkit.o
	gcc -c -fPIC $(SRCDIR)/medussa_callbacks.c $(FLAGS)

medussa_matrix.o: $(SRCDIR)/medussa_matrix.c
	gcc -c -fPIC $(SRCDIR)/medussa_matrix.c $(FLAGS)

medussa_types.o: $(SRCDIR)/medussa_types.c
	gcc -c -fPIC $(SRCDIR)/medussa_types.c $(FLAGS)

randomkit.o: $(SRCDIR)/randomkit.c
	gcc -c -fPIC $(SRCDIR)/randomkit.c $(FLAGS)

pink.o: $(SRCDIR)/pink.c
	gcc -c -fPIC $(SRCDIR)/pink.c $(FLAGS)

pa_ringbuffer.o: $(ROOTDIR)/lib/pa_ringbuffer/pa_ringbuffer.c
	gcc -c -fPIC $(ROOTDIR)/lib/pa_ringbuffer/pa_ringbuffer.c $(FLAGS)

clean:
	rm randomkit.o
	rm medussa_types.o
	rm medussa_matrix.o
	rm medussa_callbacks.o
	rm medussa.o
	rm pink.o
	rm pa_ringbuffer.o
	rm libmedussa.so.1.0.1
	rm libmedussa.so.1
	rm libmedussa.so

