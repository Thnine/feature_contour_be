from numpy import linalg
import numpy as np


#  计算顶点值的过程说明。给定点位置和扰动向量，
# （a）构造一个三角形网格，并将每个向量解释为函数梯度的线性约束，
# （b）函数给出每个顶点的值，
# （c）从这些值中，我们可以使用Marching Squares提取与扰动向量垂直的线。

class Grid:
    def __init__(self, points, perturbVect):
        self.points = points
        self.perturbVect = perturbVect
        # 这里假设网格顶点坐标是个二维数组，x从左到右增加，y从下向上增加
        self.gridCoords = self.generateGrid(points)

    def generateGrid(self, projPts):
        # 计算出网格顶点的坐标
        gridSize = 30
        # 控制边界外扩
        paddingSize = 2

        projPts = np.array(projPts)
        xmax = max(projPts[:, 0])
        xmin = min(projPts[:, 0])
        ymax = max(projPts[:, 1])
        ymin = min(projPts[:, 1])

        gridCoord = []
        if xmin == xmax:
            xmax += 1
            xmin -= 1
        if ymin == ymax:
            ymin -= 1
            ymax += 1

        yrange = ymax - ymin
        xrange = xmax - xmin

        xstep = float(xrange) / (gridSize - 1)
        ystep = float(yrange) / (gridSize - 1)
        # print("ystep", ystep)
        # print("ymax", ymax, "ymin", ymin)

        for i in range(gridSize + paddingSize * 2):
            gridCoord.append([])
            for j in range(gridSize + paddingSize * 2):
                gridCoord[i].append(
                    [xmin - paddingSize * xstep + xstep * j, ymax + paddingSize * ystep - ystep * i])
        return gridCoord

    def addAvgNeighbors(self, c):
        print("邻居平均值初始化...")
        for i in range(self.nrow * self.ncol):
            c.append([])
            self.perturbVect.append(0)
            for j in range(self.nrow * self.ncol):
                c[2 * len(self.points) + i].append(0)

        # 增加方程组，使得顶点和其邻居的平均值相等，平滑顶点值分布
        for i in range(len(c[0])):
            cInd = 2 * len(self.points) + i
            c[cInd][i] = 1
            if i % self.ncol == 0:
                # 当前顶点在最左边（左边没有邻居了）
                if i < self.ncol:
                    # 当前顶点在最上面，即左上方的顶点
                    c[cInd][i + 1] = -.5
                    c[cInd][i + self.ncol] = -.5

                elif i >= self.ncol * (self.nrow - 1):
                    # 当前顶点在最下面，即左下方的顶点
                    c[cInd][i + 1] = -.5
                    c[cInd][i - self.ncol] = -.5

                else:
                    # 最左边其他的顶点
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0
                    c[cInd][i + self.ncol] = -1.0 / 3.0

            elif (i + 1) % self.ncol == 0:
                # 当前顶点在最右边（右边没有邻居了）
                if i < self.ncol:
                    # 当前顶点在最上面，即左上方的顶点
                    c[cInd][i - 1] = -.5
                    c[cInd][i + self.ncol] = -.5
                elif i >= self.ncol * (self.nrow - 1):
                    # 当前顶点在最下面，即左下方的顶点
                    c[cInd][i - 1] = -.5
                    c[cInd][i - self.ncol] = -.5

                else:
                    # 最右边其他的顶点
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + self.ncol] = -1.0 / 3.0

            else:
                if i < self.ncol:
                    # 最上边其他的顶点
                    c[cInd][i + self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0

                elif i >= self.ncol * (self.nrow - 1):
                    # 最下边其他的顶点
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0
                else:
                    # 所有其他顶点
                    c[cInd][i + self.ncol] = -.25
                    c[cInd][i - self.ncol] = -.25
                    c[cInd][i - 1] = -.25
                    c[cInd][i + 1] = -.25

    def calcGridVertices(self):

        # 假设网格顶点是个方阵
        self.nrow = len(self.gridCoords)
        self.ncol = len(self.gridCoords[0])

        c = []
        # 每两行，第1行为x约束，第2行为y约束，其余使得顶点值平均化
        print("开始计算顶点值")
        for i in range(2 * len(self.points)):
            c.append([])
            for j in range(self.nrow * self.ncol):
                c[i].append(0)

        cubes = []

        # 每个正方形上迭代，cubes[i][j]为一个四元数组，表示第i行第j列，网格四个点的坐标，分别为右上角、左上角、左下角、右下角
        for i in range(self.nrow - 1):
            cubes.append([])
            for j in range(self.ncol - 1):
                cubes[i].append([self.gridCoords[i][j + 1], self.gridCoords[i][j], self.gridCoords[i + 1][j],
                                 self.gridCoords[i + 1][j + 1]])

        print("初始化网格坐标")

        # 迭代每个点
        for k in range(len(self.points)):
            # print(k)
            p = self.points[k]
            i = 0
            j = 0
            found = False

            # 最大迭代次为 max(rows,cols)
            while (not found):
                # 找到网格的最小X、最大X、最小Y、最大Y，四个坐标
                maxX = cubes[i][j][0][0]
                minX = cubes[i][j][0][0]
                maxY = cubes[i][j][0][1]
                minY = cubes[i][j][0][1]
                for l in range(1, 3):
                    if (cubes[i][j][l][0] > maxX):
                        maxX = cubes[i][j][l][0]
                    elif (cubes[i][j][l][0] < minX):
                        minX = cubes[i][j][l][0]
                    if (cubes[i][j][l][1] > maxY):
                        maxY = cubes[i][j][l][1]
                    if (cubes[i][j][l][1] < minY):
                        minY = cubes[i][j][l][1]

                # 如果点在当前的网格内
                if (p[0] <= maxX or abs(p[0] - maxX) <= pow(10, -5)) and (
                        p[0] >= minX or abs(p[0] - minX) <= pow(10, -5)) and (
                        p[1] >= minY or abs(p[1] - minY) <= pow(10, -5)) and (
                        p[1] <= maxY or abs(p[1] - maxY) <= pow(10, -5)):

                    # 点坐标x轴的比例
                    xscaled = (p[0] - cubes[i][j][2][0]) / (cubes[i][j][3][0] - cubes[i][j][2][0])
                    # 点坐标Y轴的比例
                    yscaled = (p[1] - cubes[i][j][2][1]) / (cubes[i][j][1][1] - cubes[i][j][2][1])

                    # 根据点的位置，给三角网格中对应的顶点增加一个方程，
                    # x轴两个顶点，左边取值-1，右边取值1，Y轴两个顶点，上面取值1，下面取值-1
                    # 例如：若点在下三角中
                    # 则方程为  grid[right bottom] -grid[left bottom]= perturbVect.X
                    # 和      grid[right top] - grid[right bottom] = perturbVect.Y 也就是扰动向量X大于0时，
                    # 右边的顶点值要大于左边的顶点值，扰动向量Y大于0时，上面的顶点值要大于下面的顶点值，这样向量所指的方向的顶点值就更大
                    # 以此求得的系数矩阵则为每个顶点的值，以此值使用Marching squares即可画出等高线图
                    if xscaled >= yscaled:
                        # 点在下面的三角网格中

                        weightYv1 = 1.0
                        weightYv2 = -1.0
                        weightYv3 = 0

                        weightXv1 = 0
                        weightXv2 = 1.0
                        weightXv3 = -1.0

                        # 对应的右下方的顶点设置值
                        c[2 * k][(i + 1) * self.ncol + j + 1] = weightXv2
                        c[2 * k + 1][(i + 1) * self.ncol + j + 1] = weightYv2

                    else:
                        # 点在上面的三角网格中
                        weightYv1 = 0
                        weightYv2 = 1.0
                        weightYv3 = -1.0

                        weightXv1 = 1.0
                        weightXv2 = -1.0
                        weightXv3 = 0

                        # 对应的左上方的顶点设置值，X:-1 Y:1
                        c[2 * k][i * self.ncol + j] = weightXv2
                        c[2 * k + 1][i * self.ncol + j] = weightYv2
                    # 右上方
                    c[2 * k][i * self.ncol + (j + 1)] = weightXv1
                    c[2 * k + 1][i * self.ncol + (j + 1)] = weightYv1

                    # 左下方
                    c[2 * k][(i + 1) * self.ncol + j] = weightXv3
                    c[2 * k + 1][(i + 1) * self.ncol + j] = weightYv3
                    found = True

                # 从左上角开始搜索，如果点在网格下面，则i+1，到下面Y更小的网格
                elif p[1] < minY:
                    # 如果点在网格右边，则j+1，到右边X更大的网格
                    if (p[0] > maxX):
                        j += 1
                    i += 1
                # 此时点的Y位置正确，如果点在网格右边，则j+1，到右边X更大的网格
                elif (p[0] > maxX):
                    j += 1

        # 扩增X矩阵，使顶点的值与其邻居的平均值相当，平滑等高线分布
        self.addAvgNeighbors(c)

        # 解线性方程组，求得系数矩阵，即每个顶点的值
        c = np.array(c)
        perturbVect = np.array(self.perturbVect)

        # print("c",c.shape,c[0])
        # print("perturbVect",perturbVect.shape,perturbVect)
        self.verticesVect = linalg.lstsq(c, perturbVect, rcond=0.000001)
        # print("self.verticesVect",self.verticesVect)
        # print("self.verticesVect[0]",self.verticesVect[0])
        print("等高线计算完成")
        return self.verticesVect[0]
