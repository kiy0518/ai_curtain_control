"""Frame sources and letterbox preprocessing.

Three sources are supported via ``open_source``:

* an integer / ``/dev/videoN`` path  -> CSI or USB camera (V4L2)
* an explicit GStreamer pipeline string (for the Rockchip ISP path)
* an image or video file path

The IMX415 on the ROCK 4D is fed through the Rockchip ISP. After enabling the
camera overlay and rebooting, the ISP-processed capture node usually appears as
a higher-numbered ``/dev/videoN``. Inspect ``v4l2-ctl --list-devices`` and set
``--source`` accordingly; if a plain device index gives a green/garbled image,
use ``--gst`` with a libcamera/ISP pipeline instead.
"""

import os

import cv2
import numpy as np

from constants import INPUT_SIZE


def open_source(source, width=1280, height=720):
    """Return an opened ``cv2.VideoCapture`` (or raise).

    ``source`` may be an int, a ``/dev/videoN`` path, a file path, or a
    GStreamer pipeline (auto-detected when it contains "!" ).
    """
    if isinstance(source, str) and "!" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    elif isinstance(source, str) and source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    # Best-effort capture size for live cameras (ignored for files).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source!r}")
    return cap


# Confirmed working capture node for the IMX415 on this ROCK 4D:
#   /dev/video11  (== /dev/video-camera0)  -> rkisp_mainpath, NV12, up to 4K@60.
# The Rockchip ISP auto-configures the media pipeline (rkaiq_3A.service must be
# active); no manual media-ctl setup is required. Auto-exposure needs a handful
# of frames to converge, so the first frames come out dark — this is normal.
CSI_DEVICE = "/dev/video-camera0"


def isp_gst_pipeline(device=CSI_DEVICE, width=1920, height=1080, fps=30):
    """GStreamer pipeline for the Rockchip ISP output (IMX415).

    This exact pipeline was verified on the board: it yields BGR frames that
    OpenCV's ``cv2.VideoCapture(..., cv2.CAP_GSTREAMER)`` reads directly.
    Requires gstreamer + the rockchip plugins (preinstalled on Radxa OS).
    """
    return (
        f"v4l2src device={device} io-mode=4 ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=2 sync=false"
    )


def letterbox(image, new_size=INPUT_SIZE, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square ``new_size``.

    Returns ``(padded_image, (ratio, pad_x, pad_y))``. The padding params let
    the postprocessor map predictions back to the original frame.
    """
    h, w = image.shape[:2]
    ratio = min(new_size / h, new_size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (new_size - nw) / 2
    pad_y = (new_size - nh) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, (ratio, pad_x, pad_y)


def is_image_file(path):
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
