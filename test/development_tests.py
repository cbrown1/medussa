import medussa
import numpy as np
from time import sleep
import sys
import random

fs = 44100.0

#medussa.init()

d = medussa.open_device()

TEST_ALL = True

# sanity checks: create and immediately delete streams
#####################################################################
if TEST_ALL or 0:
    print "testing: sanity checks: creating and deleting streams"
    
    print "create_tone()"
    s = d.create_tone(440,fs)
    del s
    print "create_tone() DONE"
    
    print "create_white()"
    s = d.create_white(fs)
    del s
    print "create_white() DONE"
    
    print "create_pink()"
    s = d.create_pink(fs)
    del s
    print "create_pink() DONE"

    print "open_file()" # this used to crash when the SFINFO layout was broken
    s = d.open_file("clean.wav")
    del s
    print "open_file() DONE"
    
    print "open_array()"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    del s
    print "open_array() DONE"


    
def playAndStopStream( s ):
    print "> playAndStopStream()"
    
    #print "is_playing: " + str(s.is_playing) # BUG this errors with RuntimeError: PaError(-9988): Invalid stream pointer -- rossb 10 April 2012
    #assert( s.is_playing == False )
    assert( s.is_muted == False )

    print "s.fs: " + str(s.fs)
    
    if 1: # test redundant (repeated) open()
        print "opening"
        s.open()
        
    if 1: # test open() then start()
        print "opening"
        
        s.open()
        print "is_playing: " + str(s.is_playing)
        assert( s.is_playing == False )

        print "starting"
        s.start()
    
        print "is_playing: " + str(s.is_playing)
        assert( s.is_playing == True )

    else: # test play()
        print "playing"
        s.play()
    
        print "is_playing: " + str(s.is_playing)
        assert( s.is_playing == True )

    if 1: # run the stream for 5 seconds
        print "running for 5 seconds"
        for i in range( 0, 10 ):
            #print s.pa_time() # BUG prints "Here" "There" garbage debug messages -- rossb 10 April 2012 [FIXED May 4]
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

    print "< playAndStopStream()"

# simplest tests: init the stream and play a constant tone or noise
#####################################################################

# play a test tone at a fixed frequency
if TEST_ALL or 0:
    print "testing: play a test tone at a fixed frequency"
    s = d.create_tone(440,fs)
    playAndStopStream( s )

# play white noise
if TEST_ALL or 0:
    print "testing: play white noise"
    s = d.create_white(fs)
    playAndStopStream( s )
    
# play pink noise
if TEST_ALL or 0:
    print "testing: play pink noise"
    s = d.create_pink(fs)
    playAndStopStream( s )


# vary parameters while playing
#####################################################################

# vary test tone frequency
if TEST_ALL or 0:
    print "testing: vary test tone frequency"
    s = d.create_tone(440,fs)
    s.play()
    for f in [440, 100, 220, 880, 500]:
        s.tone_freq = f
        sleep(1)
    s.stop()

# mute and unmute the stream every half second

if TEST_ALL or 0:
    print "testing: mute and unmute the stream every half second"
    
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
if TEST_ALL or 0:
    print "testing: linear amplitude ramp"
    s = d.create_tone(440,fs)
    s.play()
    s.mix_mat_fade_duration= .1 # same as freqency we update mix_mat
    for i in range(1,100):
        x = float(i) / 100.
        print x
        s.mix_mat = np.array( [[x, 0], [0, x]] )
        sleep(.1)
    s.stop()


# toggle mix-mat between 0 and 1 every second, increasing fade time
if TEST_ALL or 1:
    print "testing: mix_mat_fade_duration. each time signal fades in/out has longer fade time"
    s = d.create_tone(440,fs)
    s.play()
    n = 20
    fade_time = 0
    fade_inc = 1 / n
    for i in range(1,n):
        x = float(i % 2)
        print x
        s.mix_mat_fade_duration = fade_time
        fade_time += fade_inc
        s.mix_mat = np.array( [[x, 0], [0, x]] )
        sleep(1)
    s.stop()
    

# basic soundfile playback
# test printing out s.frames and s.duration
# test printing out s.cursor and s.time getter
#####################################################################

def printFiniteStreamLengthAttributes( s ):
    print "s.frames: " + str(s.frames) # rename frameCount? what are the numpy conventions?
    print "s.duration: " + str(s.duration) # rename durationSeconds? make sure it's actually in seconds

def printFiniteStreamPositionAttributes( s ):
    print "s.cursor: " + str(s.cursor)

    print "s.time(units='ms'): " + str(s.time(units='ms'))
    print "s.time(units='sec'): " + str(s.time(units='sec')) # why not 'secs' or 'seconds'?
    print "s.time(units='frames'): " + str(s.time(units='frames'))

    
# play soundfile (from array)
if TEST_ALL or 0:
    print "testing: basic soundfile playback (array)"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)
    print "s.arr: " + str(s.arr)
    playAndStopStream( s )
    printFiniteStreamPositionAttributes(s)
    
# play soundfile (streaming from file)
if TEST_ALL or 0:
    print "testing: basic soundfile playback (file)"
    s = d.open_file("clean.wav")
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)
    print "s.file_name: " + s.file_name
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
if TEST_ALL or 0:
    print "testing: restarting playback (array stream)"
    print "each time the stream starts it should play from the start"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    playStopPlayStopEtc( s )
    # this one doesn't crash at end
    
# play soundfile (streaming)
if TEST_ALL or 0:
    print "testing: restarting playback (file stream)"
    print "each time the stream starts it should play from the start"
    s = d.open_file("clean.wav")
    print "s.file_name: " + s.file_name
    playStopPlayStopEtc( s )
    # BUG CRASHES AT END OF THIS TEST -- rossb 10 April 2012 [FIXED May 4]

# test cursor update behavior
#####################################################################

def testCursorBehavior( s ):
    print "testing: When the stream is created, cursor is 0."
    print "s.cursor: " + str(s.cursor)
    assert(s.cursor == 0 )

    s.play()
    sleep(1)
    s.pause()

    print "testing: pause() halts playback but doesn't reset the cursor."
    print "s.cursor: " + str(s.cursor)
    assert(s.cursor != 0 )

    c = s.cursor
    s.play()
    sleep(1)

    print "testing: calling play() on a paused stream doesn't reset the cursor."
    print "testing: play() plays from the current cursor position."
    assert s.cursor > c

    c = s.cursor

    s.play()

    # wait for stream to reach the end (and stop)
    while s.is_playing:
        sleep( 1 )

    print "testing: If the stream ends because it has reached the end (and looping isn't enabled), then cursor remains at the end."
    assert not s.is_playing
    assert s.cursor > c
    assert s.cursor_is_at_end


    print "testing: If the stream is at the end, Play() plays from the start()."

    s.play()

    sleep( 1 )

    assert s.is_playing
    assert not s.cursor_is_at_end
    
    s.stop()

    print "testing: stop() always resets the cursor to zero."
    assert s.cursor == 0 

    print "testing: request_seek() is immediate when the stream is stopped."

    s.request_seek( 0 );
    assert s.cursor == 0;

    s.request_seek( 50 );
    assert s.cursor == 50;

    print "OK"


if TEST_ALL or 0:
    print "testing: cursor behavior (array)"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    testCursorBehavior( s )

if TEST_ALL or 0:
    print "testing: cursor behavior (streaming)"
    s = d.open_file("clean.wav")
    testCursorBehavior( s )


# test looped soundfile playback. (s.is_looping and s.loop())
# when turning looping off while playing,
# should stop looping at end of current loop cycle.
#####################################################################

def playLoopedStopLoopThenStop( s ):
    print "s.is_looping: " + str(s.is_looping)
    assert(s.is_looping == False)
    
    s.loop( True )
    print "s.is_looping: " + str(s.is_looping)
    assert(s.is_looping == True)
    
    s.play()
    sleep(13)

    print "s.is_looping: " + str(s.is_looping)
    assert(s.is_looping == True)

    # play out to end of this loop...
    s.loop( False )
    print "s.is_looping: " + str(s.is_looping)
    assert(s.is_looping == False)

    print "should stop playing at end of loop cycle"

    sleep(5)
    
    s.stop()

    
# play an array looped
if TEST_ALL or 0:
    print "testing: looped playback (array)"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    playLoopedStopLoopThenStop( s )
    
# play soundfile (streaming) looped
if TEST_ALL or 0:
    print "testing: looped playback (streaming)"
    s = d.open_file("clean.wav")
    print "s.file_name: " + s.file_name
    playLoopedStopLoopThenStop( s )


# test dynamic display of stream position.
# print pos out 4 times a second while playing
# test that cursor is at end after non-looped playback before stop() is called
#####################################################################

# play soundfile (from array)
if TEST_ALL or 0:
    print "testing: dynamic update of stream position (array)"
    x,fs = medussa.read_file("clean.wav")
    s = d.open_array(x, fs)
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)

    assert(s.is_looping == False)
    s.play()
    for i in range( 0, 20 ):
        printFiniteStreamPositionAttributes(s)
        sleep(.25)

    # BUG even though the stream is NOT looping, the cursor wraps to 0 here before stop is called -- rossb 10 April 2012 [FIXED May 25]
    assert( s.cursor != 0 )
    printFiniteStreamPositionAttributes(s)
    s.stop()
    
# play soundfile (streaming)
if TEST_ALL or 0:
    print "testing: dynamic update of stream position (streaming)"
    s = d.open_file("clean.wav")
    printFiniteStreamLengthAttributes(s)
    printFiniteStreamPositionAttributes(s)

    assert(s.is_looping == False)
    s.play()
    for i in range( 0, 20 ):
        printFiniteStreamPositionAttributes(s)
        sleep(.25)

    # BUG even though the stream is NOT looping, the cursor wraps to 0 here before stop is called -- rossb 10 April 2012 [FIXED May 25]
    assert( s.cursor != 0 )
    printFiniteStreamPositionAttributes(s)
    s.stop()


# test seeking while the stream is playing
# random fuzz test. seeks by assigning random seek positions
# to s.cursor or calling time property at random.
#####################################################################

if TEST_ALL or 0:
    print "testing: random seeking file while stream is playing"
    
    # start the stream. playback looping. choose random locations to
    # seek to every second (based on stream duration)
    # seek using a random choice of assiging to cursor or
    # calling the time property

    s = d.open_file("clean.wav")

    fileDurationSeconds = s.duration
    print str(fileDurationSeconds)
              
    s.loop( True )
    assert(s.is_looping == True)
    
    s.play()
    for i in range( 0, 20 ):
        printFiniteStreamPositionAttributes(s) # BUG sometimes crashed when the file loops if I uncomment this line -- rossb 26 April 2012 [FIXED May 4?]
        sleep(1)
        t = random.uniform(0, fileDurationSeconds)

        if random.randint(0,1) == 0:
            print "seeking with s.time()"
            s.time(units='sec',pos=t)
        else:
            print "seeking by calling s.request_seek()"
            s.request_seek( int( t * s.fs ) )

    s.stop()


# test that properties that should be read-only are read-only
#####################################################################

if TEST_ALL or 0:
    print "testing: read-only properties"
    s = d.create_tone(440,fs)

    print "testing: s.fs (read only)"
    # read:
    print "s.fs: ", s.fs
    # write:
    try:
        s.fs = 22050
        assert(False) # BAD, s.fs= should throw since s.fs should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected fs to be read-only
    print "OK"

    print "testing: s.max_mat (read / write)"
    try:
        m = s.mix_mat
        s.mix_mat = m
        assert(True) # GOOD
    except AttributeError:
        raise # BAD. problem getting or setting mix_mat
    print "OK"

    print "testing: s.mute_mat (no longer public)"
    try:
        print s.mute_mat
        assert(False) # BAD, s.mute_mat shouldn't be there
    except AttributeError:
        assert(True) # GOOD, we expected mute_mat to be absent
    print "OK"

    
    x,fs = medussa.read_file("clean.wav")
    s2 = d.open_array(x, fs)

    print "testing: s2.arr (read only)"
    # read:
    print "s2.arr: ", s2.arr
    # write:
    try:
        s2.arr = [1,2,3,4]
        assert(False) # BAD, s2.arr= should throw since s2.arr should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected s2.arr to be read-only
    print "OK"
    

    s3 = d.open_file("clean.wav")
    
    print "testing: s3.file_name (read only)"
    # read:
    print "s3.file_name: ", s3.file_name
    # write:
    try:
        s3.file_name = "xyz"
        assert(False) # BAD, s3.file_name= should throw since s3.file_name should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected s3.file_name to be read-only
    print "OK"

    print "testing: s3.frames (read only)"
    # read:
    print "s3.frames: ", s3.frames
    # write:
    try:
        s3.frames = 123
        assert(False) # BAD, s3.frames= should throw since s3.frames should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected s3.frames to be read-only
    print "OK"

    print "testing: s3.duration (read only)"
    # read:
    print "s3.duration: ", s3.duration
    # write:
    try:
        s3.duration = 123
        assert(False) # BAD, s3.duration= should throw since s3.duration should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected s3.duration to be read-only
    print "OK"

    print "testing: s3.cursor (read only)"
    # read:
    print "s3.cursor: ", s3.cursor
    # write:
    try:
        s3.cursor = 123
        assert(False) # BAD, s3.cursor= should throw since s3.cursor should be read-only
    except AttributeError:
        assert(True) # GOOD, we expected s3.cursor to be read-only
    print "OK"

    print "testing: s3.fin (no longer public)"
    try:
        print s3.fin
        assert(False) # BAD, s3.fin shouldn't be there
    except AttributeError:
        assert(True) # GOOD, we expected s3.fin to be absent
    print "OK"

    print "testing: s3.finfo (no longer public)"
    try:
        print s3.finfo
        assert(False) # BAD, s3.finfo shouldn't be there
    except AttributeError:
        assert(True) # GOOD, we expected s3.finfo to be absent
    print "OK"
    

    print "ATTRIBUTE TESTS PASSED"

    playAndStopStream( s )
    playAndStopStream( s2 )
    playAndStopStream( s3 )

print "done."
