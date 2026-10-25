import math
import cv2
import numpy as np

A = "/Users/afridee/.cursor/projects/Volumes-T7-Dev-stock-material-detection/assets/"
IMGS = {
    "a": A + "WhatsApp_Image_2026-10-25_at_15.15.11__1_-dd6ea2da-65e6-4f69-9a78-94b764bf415e.jpg",
    "b": A + "WhatsApp_Image_2026-10-25_at_15.15.11-2a04a062-78e7-4862-b66d-5abd6d75e9cb.jpg",
}
# (image, butt end xy, tip xy, ring x)
STICKS = [
    ("a", (107, 331), (476, 298), 346),
    ("a", (113, 541), (469, 508), 341),
    ("a", (135, 748), (475, 731), 352),
    ("b", (107, 305), (469, 272), 343),
    ("b", (112, 515), (463, 479), 338),
    ("b", (134, 715), (468, 698), 348),
]

out = []
for img_key, butt, tip, ring_x in STICKS:
    img = cv2.imread(IMGS[img_key])
    dx, dy = tip[0] - butt[0], tip[1] - butt[1]
    ang = math.degrees(math.atan2(dy, dx))
    length = math.hypot(dx, dy)
    # start of band = a little before the ring, along the axis
    t0 = (ring_x - 5 - butt[0]) / dx
    start = (butt[0] + t0 * dx, butt[1] + t0 * dy)
    band_len = length * (1 - t0) + 8
    cx = start[0] + math.cos(math.radians(ang)) * band_len / 2
    cy = start[1] + math.sin(math.radians(ang)) * band_len / 2
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    h = 28
    crop = cv2.getRectSubPix(rot, (int(round(band_len)), h), (cx, cy))
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out.append(crop)

for i, c in enumerate(out):
    cv2.imwrite(f"/Volumes/T7-Dev/stock-material-detection/.scratch/aris_band_{i}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
w = max(c.shape[1] for c in out)
sheet = np.vstack([cv2.copyMakeBorder(c, 4, 4, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in out])
cv2.imwrite("/Volumes/T7-Dev/stock-material-detection/.scratch/aris_bands_sheet.jpg", sheet)
