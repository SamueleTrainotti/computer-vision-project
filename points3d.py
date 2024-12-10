centerCubes3d = [(-8, -4, 0), (-4, -4, 0), (0, -4, 0), (4, -4, 0), (8, -4, 0), (8, 0, 0), (8, 4, 0), (-8, 4, 0), (-8, 0, 0), (4, 0, 0), (-4, 0, 0)]

# Define the 3D points
points3dZ3, points3dZ5, points3dZ7, points3dZ9, points3dZ11 = [], [], [], [], []
points3dZ11 += [(x+0.2, y+0.2, z + 10.8) for x, y, z in centerCubes3d[1:4]]
points3dZ9 += [(x-0.2, y-0.2, z + 9.2) for x, y, z in centerCubes3d[0:5]]
points3dZ7 += [(x-0.2, y-0.2, z + 7.2) for x, y, z in centerCubes3d[0:5]]
points3dZ5 += [(x-0.2, y-0.2, z + 5.2) for x, y, z in centerCubes3d[0:5]]
points3dZ3 += [(x-0.2, y-0.2, z + 3.2) for x, y, z in centerCubes3d[0:5]]

points3dZ11 += [(centerCubes3d[5][0]-0.2, centerCubes3d[5][1]+0.2, centerCubes3d[5][2] + 10.8)]
points3dZ11 += [(centerCubes3d[6][0]+0.2, centerCubes3d[6][1]+0.2, centerCubes3d[6][2] + 11.2)]
points3dZ9 += [(x+0.2, y+0.2, z + 9.2) for x, y, z in centerCubes3d[5:7]]
points3dZ7 += [(x+0.2, y+0.2, z + 7.2) for x, y, z in centerCubes3d[5:7]]
points3dZ3 += [(x+0.2, y+0.2, z + 3.2) for x, y, z in centerCubes3d[5:7]]
points3dZ5 += [(centerCubes3d[5][0]-0.2, centerCubes3d[5][1]+0.2, centerCubes3d[5][2] + 4.8)]

points3dZ11 += [(4.2, 4.2, 11.2), (-4.2, 4.2, 11.2)]

points3dZ11 += [(centerCubes3d[7][0]-0.2, centerCubes3d[7][1]+0.2, centerCubes3d[7][2] + 11.2)]
points3dZ11 += [(centerCubes3d[8][0]+0.2, centerCubes3d[8][1]+0.2, centerCubes3d[8][2] + 10.8)]
points3dZ9 += [(x-0.2, y+0.2, z + 9.2) for x, y, z in centerCubes3d[7:9]]
points3dZ7 += [(x-0.2, y+0.2, z + 7.2) for x, y, z in centerCubes3d[7:9]]
points3dZ3 += [(x-0.2, y+0.2, z + 3.2) for x, y, z in centerCubes3d[7:9]]
points3dZ5 += [(centerCubes3d[8][0]+0.2, centerCubes3d[8][1]+0.2, centerCubes3d[8][2] + 4.8)]

points3dZ11 += [(centerCubes3d[9][0]+0.2, centerCubes3d[9][1]+0.2, centerCubes3d[9][2] + 11.2), (centerCubes3d[10][0]-0.2, centerCubes3d[10][1]+0.2, centerCubes3d[10][2] + 11.2)]
points3dZ9 += [(centerCubes3d[9][0]+0.2, centerCubes3d[9][1]+0.2, centerCubes3d[9][2] + 9.2), (centerCubes3d[10][0]-0.2, centerCubes3d[10][1]+0.2, centerCubes3d[10][2] + 9.2)]
points3dZ7 += [(centerCubes3d[9][0]+0.2, centerCubes3d[9][1]+0.2, centerCubes3d[9][2] + 7.2), (centerCubes3d[10][0]-0.2, centerCubes3d[10][1]+0.2, centerCubes3d[10][2] + 7.2)]
points3dZ5 += [(centerCubes3d[9][0]+0.2, centerCubes3d[9][1]+0.2, centerCubes3d[9][2] + 5.2), (centerCubes3d[10][0]-0.2, centerCubes3d[10][1]+0.2, centerCubes3d[10][2] + 5.2)]
points3dZ3 += [(centerCubes3d[9][0]+0.2, centerCubes3d[9][1]+0.2, centerCubes3d[9][2] + 3.2), (centerCubes3d[10][0]-0.2, centerCubes3d[10][1]+0.2, centerCubes3d[10][2] + 3.2)]

points3dZ11 += [(3.8, 0.2, 12.8), (-3.8, +0.2, 12.8)]
#points3d = points3dZ3 + points3dZ5 + points3dZ7 + points3dZ9 + points3dZ11