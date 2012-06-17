# Copyright (c) 2010-2012 Christopher Brown
#
# This file is part of Medussa.
#
# Medussa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Medussa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Medussa.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#

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
	gcc -shared -Wl,-soname,libmedussa.so -o libmedussa.so.1.0.1 medussa.o medussa_callbacks.o medussa_matrix.o medussa_types.o randomkit.o pink.o pa_ringbuffer.o disk_streaming.o $(FLAGS)

medussa.o: $(SRCDIR)/medussa.c medussa_callbacks.o medussa_matrix.o medussa_types.o randomkit.o pink.o pa_ringbuffer.o disk_streaming.o 
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

disk_streaming.o: $(SRCDIR)/disk_streaming.c
	gcc -c -fPIC -lpthread -lrt $(SRCDIR)/disk_streaming.c $(FLAGS)

clean:
	rm randomkit.o
	rm medussa_types.o
	rm medussa_matrix.o
	rm medussa_callbacks.o
	rm medussa.o
	rm pink.o
	rm pa_ringbuffer.o
	rm disk_streaming.o
	rm libmedussa.so.1.0.1
	rm libmedussa.so.1
	rm libmedussa.so

