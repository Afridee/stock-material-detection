import math
import cv2
import numpy as np

A = "/Users/afridee/.cursor/projects/Volumes-T7-Dev-stock-material-detection/assets/"
# (image, butt end xy, tip xy, gold ring xy)
STICKS = [
    (A + "WhatsApp_Image_2026-10-25_at_15.43.16-9cc128e1-45d6-4143-9bd6-d60cc4362d2d.jpg", (273, 885), (302, 288), (295, 461)),
]
_OLD = [
    (A + "WhatsApp_Image_2026-10-25_at_15.36.33__1_-98245c6e-13f7-4237-ab63-953a46091c70.jpg", (290, 870), (250, 150), (264, 407)),
    (A + "WhatsApp_Image_2026-10-25_at_15.36.33-bea250b2-6c71-4b1c-a11a-11139d5d069f.jpg", (248, 860), (227, 308), (234, 502)),
    (A + "WhatsApp_Image_2026-10-25_at_15.36.32-56353eda-81d0-44cc-939b-4ad28a7de51b.jpg", (283, 872), (257, 370), (266, 548)),
    (A + "WhatsApp_Image_2026-10-25_at_15.36.33__2_-06044aca-45e2-4a6b-bc73-85e59069025b.jpg", (281, 870), (281, 270), (281, 480)),
]
H = 72
OUT = "/Volumes/T7-Dev/stock-material-detection/.scratch/bd_"

out = []
for path, butt, tip, ring in STICKS:
    img = cv2.imread(path)
    dx, dy = tip[0] - butt[0], tip[1] - butt[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    ang = math.degrees(math.atan2(dy, dx))
    s0 = (ring[0] - butt[0]) * ux + (ring[1] - butt[1]) * uy - 8
    band_len = length - s0 + 6
    cx = butt[0] + ux * (s0 + band_len / 2)
    cy = butt[1] + uy * (s0 + band_len / 2)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    crop = cv2.getRectSubPix(rot, (int(round(band_len)), H), (cx, cy))
    out.append(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

for i, c in enumerate(out):
    cv2.imwrite(f"{OUT}band_{i}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
w = max(c.shape[1] for c in out)
sheet = np.vstack([cv2.copyMakeBorder(c, 4, 4, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in out])
cv2.imwrite(f"{OUT}sheet.jpg", sheet)
