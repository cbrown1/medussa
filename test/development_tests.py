import medussa
import numpy as np
from time import sleep
import sys
import random

fs = 44100.0

d = medussa.open_device()


def playAndStopStream( s ):
    #print "is_playing = " + str(s.is_playing) # BUG this errors with RuntimeError: PaError(-9988): Invalid stream pointer -- rossb 10 April 2012
    #assert( s.is_playing == False )
    assert( s.is_muted == False )

    print "s.fs = " + str(s.fs)
    
    if 0: # test redundant (repeated) open()
        print "opening"
        s.open()
        
    if 0: # test open() then start()
        print "opening"
        
        s.open()
        print "is_playing = " + str(s.is_playing)
        assert( s.is_playing == False )

        print "starting"
        s.start()
    
        print "is_playing = " + str(s.is_playing)
        assert( s.is_playing == True )

    else: # test play()
        print "playing"
        s.play()
    
        print "is_playing = " + str(s.is_playing)
        assert( s.is_playing == True )

    if 1: # run the stream for 5 seconds
        print "running for 5 seconds"
        for i in range( 0, 10 ):
            #print s.pa_time() # BUG prints "Here" "There" garbage debug messages -- rossb 10 April 2012
            sleep(.5)
    else: # play and pause the stream every half second for 5 seconds
        print "alternating pause() and play()"
        for i in range( 0, 10 ):
            #print s.pa_time()
            if( i % 2 == 0 ):
                print "pausing"
                s.pause()
            else:
                print "playing"
                s.play()
            sleep(.5)
    
    print "stopping"
    s.stop()
    print "is_playing = " + str(s.is_playing)
    assert( s.is_playing == False )

    assert( s.is_muted == False )


# simplest tests: init the stream and play a constant tone or noise
#####################################################################

# play a test tone at a fixed frequency
if 0:
    s = d.create_tone(440,fs)
    playAndStopStream( s )

# play white noise
if 0:
    s = d.create_white(fs)
    playAndStopStream( s )
    
# play pink noise
if 0:
    s = d.create_pink(fs)
    playAndStopStream( s )


# vary parameters while playing
#####################################################################

# vary test tone frequency
if 0:
    s = d.create_tone(440,fs)
    s.play()
    for f in [440, 100, 220, 880, 500]:
        s.tone_freq = f
        sleep(1)
    s.stop()

# mute and unmute the stream every half second

if 0:
    s = d.create_tone(440,fs)
    s.play()
    for i in range(0,10):
        print i % 2
        if i % 2 == 0:
            print "muting"
            #s.mute() # BUG s.mute() doesn't work. s.is_muted = True does -- rossb 10 April 2012
            s.is_muted = True
            print "is_muted = " + str(s.is_muted)
            assert( s.is_muted == True )
            
        else:
            print "unmuting"
            s.unmute()
            print "is_muted = " + str(s.is_muted)
            assert( s.is_muted == False )
            
        sleep(.5);
    s.stop()


# linear amplitude ramp
if 0:
    s = d.create_tone(440,fs)
    s.play()
    for i in range(1,100):
        x = float(i) / 100.
        print x
        s.mix_mat = np.array( [[x, 0], [0, x]] )
        sleep(.1)
    s.stop()


# basic soundfile playback
# test printing out s.frames and s.duration
# test printing out s.cursor and s.time getter
#####################################################################

def printFiniteStreamLengthAttributes( s ):
    print "s.frames = " + str(s.frames) # rename frameCount? what are the numpy conventions?
    print "s.duration = " + str(s.duration) # rename durationSeconds. make sure it's actually in seconds

def printFiniteStreamPositionAttributes( s ):
    print "s.cursor = " + str(s.cursor)

    print "s.time(units='ms') " + str(s.time(units='ms'))
    print "s.time(units='sec') " + str(s.time(units='sec')) # why not 'secs' or 'seconds'?
    print "s.time(units='frames') " + str(s.time(units='frames'))

    
# play soundfile (from array)
if 0:
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)
    print "s.arr = " + str(s.arr)
    playAndStopStream( s )
    printFiniteStreamPositionAttributes(s)
    
# play soundfile (streaming)
if 0:
    s = d.open_file("clean.wav")
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)
    print "s.file_name = " + s.file_name
    #print "s.finfo = " + s.finfo # BUG THIS CRASHES -- rossb 10 April 2012
    playAndStopStream( s )
    printFiniteStreamPositionAttributes(s)


# test restarting playback. 
# repeatedly play for 2 seconds and stop. cursor should reset and
# the stream should start from the beginning each time it plays.
#####################################################################

def playStopPlayStopEtc( s ):
    for i in range( 0, 5 ):
        s.play()
        sleep(2)
        s.stop()
  
# play an array
if 0:
    print "each time the stream starts it should play from the start"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    playStopPlayStopEtc( s )
    # this one doesn't crash at end
    
# play soundfile (streaming)
if 0:
    print "each time the stream starts it should play from the start"
    s = d.open_file("clean.wav")
    print "s.file_name = " + s.file_name
    playStopPlayStopEtc( s )
    # BUG CRASHES AT END OF THIS TEST -- rossb 10 April 2012


# test looped soundfile playback. (s.is_looping and s.loop())
# when turning looping off while playing,
# should stop looping at end of current loop cycle.
#####################################################################

def playLoopedStopLoopThenStop( s ):
    print "s.is_looping = " + str(s.is_looping)
    assert(s.is_looping == False)
    
    s.loop( True )
    print "s.is_looping = " + str(s.is_looping)
    assert(s.is_looping == True)
    
    s.play()
    sleep(13)

    print "s.is_looping = " + str(s.is_looping)
    assert(s.is_looping == True)

    # play out to end of this loop...
    s.loop( False )
    print "s.is_looping = " + str(s.is_looping)
    assert(s.is_looping == False)

    print "should stop playing at end of loop cycle"

    sleep(5)
    
    s.stop()

    
# play an array looped
if 0:
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    playLoopedStopLoopThenStop( s )
    
# play soundfile (streaming) looped
if 0:
    s = d.open_file("clean.wav")
    print "s.file_name = " + s.file_name
    playLoopedStopLoopThenStop( s )


# test dynamic display of stream position.
# print pos out 4 times a second while playing
#####################################################################

# play soundfile (from array)
if 0:
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)

    assert(s.is_looping == False)
    s.play()
    for i in range( 0, 20 ):
        printFiniteStreamPositionAttributes(s)
        sleep(.25)

    # BUG even though the stream is looping, the cursor wraps to 0 here before stop is called -- rossb 10 April 2012 
    printFiniteStreamPositionAttributes(s)
    s.stop()
    
# play soundfile (streaming)
if 0:
    s = d.open_file("clean.wav")
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)

    assert(s.is_looping == False)
    s.play()
    for i in range( 0, 20 ):
        printFiniteStreamPositionAttributes(s)
        sleep(.25)

    # BUG even though the stream is looping, the cursor wraps to 0 here before stop is called -- rossb 10 April 2012
    printFiniteStreamPositionAttributes(s)
    s.stop()


# test seeking while the stream is playing
# random fuzz test. seeks by assigning random seek positions
# to s.cursor or calling time property at random.
#####################################################################

if 0:
    # start the stream. playback looping. choose random locations to
    # seek to every second (based on stream duration)
    # seek using a random choice of assiging to cursor or
    # calling the time property

    s = d.open_file("clean.wav")

    fileDurationSeconds = s.duration / 1000.
    print str(fileDurationSeconds)
              
    s.loop( True )
    assert(s.is_looping == True)
    
    s.play()
    for i in range( 0, 20 ):
        #printFiniteStreamPositionAttributes(s) # BUG sometimes crashes when the file loops if I uncomment this line -- rossb 26 April 2012
        sleep(1)
        t = random.uniform(0, fileDurationSeconds)

        if random.randint(0,1) == 0:
            print "seeking with s.time"
            s.time(units='sec',pos=t)
        else:
            print "seeking by assigning to s.cursor"
            s.cursor = int( t * s.fs )

    s.stop()


# test that properties that should be read-only are read-only
#####################################################################

if 1:
    s = d.create_tone(440,fs)
    playAndStopStream( s )

    # TODO


print "done."
