import numpy as np
import glob
import os


data_path = "D:/Ear/dataset/eval_results/"
types = ["mixed(only left ear is normal)/", "otosclerosis/", "normal/part1/", "normal/part2/"]
ears = ["left ear", "right ear"]
SCOREs = [0.99]
RATIOs = [0.25]


def parse_type(T, E):
    if 'otosclerosis' in T:
        return 1
    elif 'normal/' in T:
        return 2
    elif 'mixed' in T:
        if 'left' in E:
            return 2
        elif 'right' in E:
            return 1


def get_data(file):
    file = open(file, 'r')
    info = []
    for line in file:
        line = line.strip('\n').split(',')
        line = [l.split(':')[-1] for l in line]
        line.append(line[0])
        line[0] = int(line[0].split('_i')[-1].split('.')[0])
        line[1] = int(line[1])
        line[2] = float(line[2])
        if line[2] > 0.5:
            info.append(line)
    return info


def find_list(info):
    current_start, current_end, current_max, current_peak = [info[0][0], info[0][0], 1, info[0][2]]
    max_len, max_peak = [1, info[0][2]]
    current_index_s, current_index_e = 0, 0
    index_s, index_e = 0, 0
    for i in range(len(info)-1):
        layer = info[i+1][0]
        score = info[i+1][2]
        if layer - current_end == 1:
            current_end = layer
            current_index_e = i+1
            current_max = current_end - current_start + 1
            if score > current_peak:
                current_peak = score
        else:
            current_start = layer
            current_index_s = i+1
            current_max = 1
            current_end = layer
            current_index_e = i+1
            current_peak = score
        if current_max > max_len:
            index_s = current_index_s
            index_e = current_index_e
            max_len = current_max
            max_peak = current_peak
        elif current_max == max_len:
            if current_peak > max_peak:
                index_s = current_index_s
                index_e = current_index_e
                max_len = current_max
                max_peak = current_peak
    return index_s, index_e


def find_type(info, score, ratio):
    result = []
    ill_count, normal_count = 0, 0
    if score == 0:
        info = np.array(info)
        score = max(info[:, 2]) * 0.9
    for i in range(len(info)):
        if info[i][2] >= score:
            result.append(info[i])
    for r in result:
        if r[1] == 1:
            ill_count += 1
        elif r[1] == 2:
            normal_count += 1
    if ill_count == 0 and normal_count == 0:
        ratio = 0
        return 0
    else:
        rat = ill_count / (ill_count + normal_count)
    if rat >= ratio:
        return 1
    else:
        return 2


for S in SCOREs:
    for R in RATIOs:
        RST = [0, 0, 0, 0, 0, 0]    # TPP, FPP, FNP, TPN, FPN, FNN
        PN = [0, 0]
        for T in types:
            for E in ears:
                CTs = os.listdir(data_path + T + '/' + E)
                for CT in CTs:
                    path = data_path + T + '/' + E + '/' + CT
                    file = glob.glob(path + '/' + 'result.txt')
                    if file:
                        info = get_data(file[0])
                        if len(info) == 0:
                            print("Length is zero:{}".format(path))
                            RST[(parse_type(T, E) - 1) * 3 + 2] += 1
                            PN[parse_type(T, E) - 1] += 1
                        else:
                            start, end = find_list(info)
                            info = info[start: end+1]
                            if parse_type(T, E) == find_type(info, S, R):
                                RST[(parse_type(T, E) - 1) * 3] += 1
                                PN[parse_type(T, E) - 1] += 1
                            else:
                                print("False:{}".format(path))
                                RST[(parse_type(T, E) - 1) * 3 + 2] += 1
                                RST[(find_type(info, S, R) - 1) * 3 + 1] += 1
                                PN[parse_type(T, E) - 1] += 1
                    else:
                        print("No file:{}".format(path))
        print("Score:{}\tRatio:{}\tPrecision P:{:.3F}%\tRecall P:{:.3F}%\tPrecision N:{:.3F}%\t Recall N:{:.3F}%\tTotal P:{}\tTotal N:{}"
              .format(S, R, RST[0]/(RST[0]+RST[1])*100, RST[0]/(RST[0]+RST[2])*100, RST[3]/(RST[3]+RST[4])*100, RST[3]/(RST[3]+RST[5])*100, PN[0], PN[1]))



