"""Utils for ffmpeg-python
"""

### Progress bar
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


### Get video stats using ffmpeg output
def gvs(iVid):
    AUDIO = False
    process = subprocess.Popen(['ffmpeg', '-hide_banner', '-i', iVid, '-y' ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
    for line in process.stdout:
    print(line)
    if ' Video:' in line:
        l_split = line.split(',')
        #print('---------printing line ", line)
        for segment in l_split[1:]:
            if 'fps' in segment:
                    s = segment.strip().split(' ')
                    fps = float(s[0])
            if 'x' in segment:
                    s = segment.strip().split('x')
                    width = int(s[0])
                    s2 = s[1].split(' ')
                    height = int(s2[0])
    if 'Duration:' in line:
        s = line.split(',')
        ss = s[0].split(' ')
        sss = ss[3].strip().split(':')
        seconds = float(sss[0])*60*60 + float(sss[1])*60 + float(sss[2])
    if 'Audio:' in line:
        AUDIO = True

    print('fps = ', str(fps))
    print('width = ', str(width))
    print('height = ', str(height))
    print('seconds = ', str(seconds))
    print('AUDIO = ', AUDIO)

### Functions based on ffmpeg-python video tensorflow example
def readFrameAsNp(ffmpegDecode, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = ffmpegDecode.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def writeFrameAsByte(ffmpegEncode, frame):
    logger.debug('Writing frame')
    ffmpegEncode.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def vid2np(in_filename):
    logger.info('vid2np() -- Decoding to pipe')
    codec = 'h264'
    args = (
            ffmpeg
            .input(in_filename,
                **{'c:v': codec})
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .global_args("-hide_banner")
            .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def np2vid(out_filename, fps_out, in_file, widthOut, heightOut):
    logger.info('np2vid() encoding from pipe')
    global AUDIO
    codec = 'hevc'
    if AUDIO == True :
        pipeline2 = ffmpeg.input(in_file)
        audio = pipeline2.audio
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                s='{}x{}'.format(widthOut, heightOut),
                framerate=fps_out )
            .output(audio, out_filename , pix_fmt='yuv420p', **{'c:v': codec}, 
                shortest=None, acodec='copy')
            .global_args("-hide_banner")
            .overwrite_output()
            .compile()
        )
    else:
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', 
                s='{}x{}'.format(widthOut, heightOut), 
                framerate=fps_out )
            .output(out_filename , pix_fmt='yuv420p', **{'c:v': codec})
            .global_args("-hide_banner")
            .overwrite_output()
            .compile()
        )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

### The model changes the resolution, processes blank to find new resolution
###***
def getOutputResolution():
    #process a blank frame and return dimesions
    blank = np.zeros([height,width,3],dtype=np.uint8)
    blank.fill(255)
    fastAI_image = Image(pil2tensor(blank, dtype=np.float32).div_(255))
    p,img_hr,b = learn.predict(fastAI_image)
    im = image2np(img_hr)
    x = im.shape
    out_height = x[0]
    out_width = x[1]
    return int(out_width), int(out_height)


