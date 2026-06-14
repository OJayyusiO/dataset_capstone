"""
Video writer helper.

HTML5 <video> only decodes H.264 ('avc1') inside an MP4 container, not the
MPEG-4 Part 2 'mp4v' codec OpenCV defaults to. open_video_writer tries 'avc1'
first so the output embeds in a browser, then falls back to 'mp4v' if this
OpenCV build can't open an H.264 encoder (common on some Windows wheels) so a
run never silently produces an unwritable / 0-byte file.
"""

import cv2


def open_video_writer(path, fps, size, prefer='avc1'):
    """Open a cv2.VideoWriter, preferring a browser-friendly codec.

    Returns (writer, codec_used). Raises RuntimeError if neither codec opens.
    """
    path = str(path)
    for codec in (prefer, 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            if codec != prefer:
                print(f"  [video] '{prefer}' (H.264) unavailable in this OpenCV "
                      f"build; wrote '{path}' with '{codec}' instead "
                      f"(may not play in an HTML5 <video> tag).")
            return writer, codec
        writer.release()
    raise RuntimeError(
        f"Could not open a VideoWriter for {path} with 'avc1' or 'mp4v'.")
