def iou(b1, b2):
    size1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    size2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    x = max(0, min(b1[2], b2[2])-max(b1[0], b2[0]))
    y = max(0, min(b1[3], b2[3])-max(b1[1], b2[1]))
    return x*y/(size1+size2-x*y)
