import math
import cv2
import numpy as np

A = "/Users/afridee/.cursor/projects/Volumes-T7-Dev-stock-material-detection/assets/"
IMGS = {
    "a": A + "WhatsApp_Image_2026-10-25_at_15.23.58-45e4eddb-5ffe-4e32-aced-e86ce47cad54.jpg",
    "b": A + "WhatsApp_Image_2026-10-25_at_15.23.58__1_-3e49550f-dde9-46f9-be66-87f70d2342eb.jpg",
}
# (image, butt end xy, tip xy, ring x)
STICKS = [
    ("a", (68, 238), (512, 214), 352),
    ("a", (63, 455), (516, 429), 353),
    ("a", (51, 603), (511, 576), 348),
    ("a", (70, 854), (539, 818), 371),
    ("b", (65, 251), (501, 228), 346),
    ("b", (63, 463), (505, 437), 346),
    ("b", (53, 609), (500, 579), 343),
    ("b", (75, 852), (527, 814), 366),
]
H = 34
OUT = "/Volumes/T7-Dev/stock-material-detection/.scratch/apple_"

out = []
overlays = {k: cv2.imread(v) for k, v in IMGS.items()}
for img_key, butt, tip, ring_x in STICKS:
    img = cv2.imread(IMGS[img_key])
    dx, dy = tip[0] - butt[0], tip[1] - butt[1]
    ang = math.degrees(math.atan2(dy, dx))
    length = math.hypot(dx, dy)
    t0 = (ring_x - 5 - butt[0]) / dx
    start = (butt[0] + t0 * dx, butt[1] + t0 * dy)
    band_len = length * (1 - t0) + 8
    cx = start[0] + math.cos(math.radians(ang)) * band_len / 2
    cy = start[1] + math.sin(math.radians(ang)) * band_len / 2
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    crop = cv2.getRectSubPix(rot, (int(round(band_len)), H), (cx, cy))
    out.append(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    cv2.line(overlays[img_key], butt, tip, (0, 0, 255), 1)

for i, c in enumerate(out):
    cv2.imwrite(f"{OUT}band_{i}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
w = max(c.shape[1] for c in out)
sheet = np.vstack([cv2.copyMakeBorder(c, 4, 4, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in out])
cv2.imwrite(f"{OUT}sheet.jpg", sheet)
cv2.imwrite(f"{OUT}overlay.jpg", np.hstack(list(overlays.values())))
