import numpy as np
import torch
import struct
import operator
import os


class Node():
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def __lt__(self, other):
        if self.u == other.u:
            return operator.lt(self.v, other.v)
        else:
            return operator.lt(self.u, other.u)


class ReadFileFromBin:
    '''
    This class is responsible for reading data from some bin-files which
    has been processed by C++ program.
    Those files are:
    metadata.bin
    row_indices.bin , column_indices.bin,
    adjacent_list.bin
    '''
    # file_name = ['metadata.bin', '/row_indices.bin', '/column_indices.bin', '/adjacent_list.bin']

    def __init__(self, root_path, file_name):

        bin_file = root_path + '/' + file_name.split('.')[-2] + '.bin'
        if os.path.exists(root_path+ '/' + file_name.split('.')[-2] + '.bin') is False:
            bin_file = self.generatebinfile(root_path + '/', file_name,  0, False)

        with open(bin_file, 'rb') as f:
            self.node_num = struct.unpack('i', f.read(4))[0]
            self.edge_num = struct.unpack('i', f.read(4))[0]
            self.offset = struct.unpack(str(self.node_num + 1) + 'i', f.read((self.node_num+1)*4))
            self.row_indices = struct.unpack(str(self.edge_num) + 'i', f.read(self.edge_num*4))
            self.column_indices = struct.unpack(str(self.edge_num) + 'i', f.read(self.edge_num*4))

    def generatebinfile(self, root, file_name, line2lip, directed=False):
        with open(root+'/'+file_name, 'r') as f:
            print('---open file---')
            while line2lip >= 1:
                f.readline()
                line2lip -= 1
            line = f.readline().split(' ')
            if line.__len__() == 1:
                line = line[-1].split('\t')
            node_num, edge_num = int(line[0]), int(line[1])

            offset = np.zeros([node_num + 1], dtype=int)
            print('----read line-----')
            edge = []
            for line in f:
                line = line.split(' ')
                if line.__len__() == 1:
                    line = line[-1].split('\t')
                offset[int(line[0])] += 1
                if directed:
                    offset[int(line[1])] += 1
                edge.append(Node(int(line[0]), int(line[1])))
            print('sort the edge')

            edge.sort()
            print(edge.__len__())

            output_name = str(file_name).split('.')[-2] + '.bin'
            tmp = 0
            print('prefix sum of offset')
            for i in range(0, node_num+1):
                t = offset[i]
                offset[i] = tmp
                tmp += t
            print('write bin file')
            with open(root+'/'+output_name, 'wb') as of:
                of.write(node_num.to_bytes(4, byteorder='little'))
                of.write(edge_num.to_bytes(4, byteorder='little'))
                of.write(offset.tobytes())
                for ii in range(0, edge.__len__()):
                    of.write(int(edge[ii].u).to_bytes(4, byteorder='little'))
                for ii in range(0, edge.__len__()):
                    of.write(int(edge[ii].v).to_bytes(4, byteorder='little'))
            return root+'/'+output_name


if __name__ == '__main__':
    readfile = ReadFileFromBin('./train', 'test.txt')
    print(readfile.node_num)
    print(readfile.edge_num)
    print(readfile.offset)
    print(readfile.row_indices)
    print(readfile.column_indices)




