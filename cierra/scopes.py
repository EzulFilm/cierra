"""Numeric video and timeline scopes on display-code images.

Each frame gets the same tone, histogram, waveform, vectorscope, neutral-cast, skin and clipping measurements as the
grading loop. The timeline report adds mean-luma flicker, with frame numbers kept beside every measurement so a change
can be found in the source clip.
"""

from . import analysis


def timeline_scopes(frames):
    """Report for ((frame_number, (3, H, W) display image), ...), in the order given."""
    frames = list(frames)
    if not frames:
        raise ValueError("a timeline scope needs at least one frame")
    shape = frames[0][1].shape
    if any(image.shape != shape for _, image in frames):
        raise ValueError("every frame in a timeline scope must have the same shape")
    numbers = [int(n) for n, _ in frames]
    images = [image for _, image in frames]
    return {"first_frame": numbers[0], "last_frame": numbers[-1], "count": len(frames),
            "frames": [{"frame": n, **analysis.stats(image)} for n, image in frames],
            "flicker": analysis.flicker(images)}


def video_scopes(path, first=0, last=None, step=1, code_range=None):
    """Read frame first..last, every step-th frame, and return its timeline report (FFmpeg through cierra.video)."""
    from . import video
    last = first if last is None else last
    if first < 0 or last < first or step < 1:
        raise ValueError("scope frames need 0 <= first <= last and step >= 1")
    return timeline_scopes((n, video.read_frame(path, n, code_range)) for n in range(first, last + 1, step))
