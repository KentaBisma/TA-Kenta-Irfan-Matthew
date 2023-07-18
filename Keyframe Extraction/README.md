# slide-detector

Detect and extract slides in a video.

Slides are frames in the video that stay the same for several frames in
a row. They are dumped into the current directory with filenames of the
form `static_at_MM:SS.jpg` where `MM:SS` is the timestamp of the first
frame showing the slide.

```
Usage:
    ./slide-detector.py VIDEO [x y w h] [slide_time_millis]
```
`VIDEO` is the video file. If you want to crop the video before
analysis, the next four arguments are the (x, y) coordinates of the
top-left of the crop rectangle and the (w, h) of its size (**w**idth and
**h**eight). `slide_time_millis` is how long a slide has to be on the
screen for it to count as a slide (default is `3000` = 3 seconds).

There's no command-line option for it, but at the top of the file,
there's a `UI = False` option. When `False` the analysis is run at full
speed. When `True`, the video is shown (in multiple windows showing the
different stages of the analysis) at approximately 1x and the following
keyboard commands are available which can help figure out the proper
crop values and test:
 * `q`: quit
 * `z`/`x`: rewind/fast-forward by 100 frames (usually ~3 seconds).
 * `w`/`a`/`s`/`d`: move the top-left corner by 10px and print
     the new x/y value.
 * `W`/`A`/`S`/`D`: move the top-left corner by 1px and print
     the new x/y value.
 * `i`/`j`/`k`/`l`: move the bottom-right corner by 10px and print
     the new w/h value.
 * `I`/`J`/`K`/`L`: move the bottom-right corner by 1px and print
     the new w/h value.
