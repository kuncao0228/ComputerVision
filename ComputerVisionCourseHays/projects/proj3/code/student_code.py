import numpy as np



def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    A = []
    for i in range (0, points_2d.shape[0]):
        
        A_index = 0
        temp_3d = np.append(points_3d[i].reshape(1,-1), np.ones(1))

        temp = np.append(temp_3d, np.zeros((1,4)))
        multTemp1 = np.asarray([-1* points_2d[i,0] * points_3d[i,0], -1* points_2d[i,0] * points_3d[i,1], -1*points_2d[i,0] * points_3d[i,2]])
        A.append(np.append(temp, multTemp1))
            
        temp2 = np.append(np.zeros((1,4)), temp_3d)
        multTemp2 = np.asarray([-1* points_2d[i,1] * points_3d[i,0], -1* points_2d[i,1] * points_3d[i,1], -1*points_2d[i,1] * points_3d[i,2]])
        A.append(np.append(temp2, multTemp2))
        
        A_index += 2
         
    A = np.asarray(A)

    
    b = points_2d.flatten()
    M,_,_,_ = np.linalg.lstsq(A, b, rcond=-1.0)
    M=np.append(M,1).reshape(3,4)


    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.


    Q = M[0:3, 0:3]

    m_4 = M[:,3]
    
    inverse = np.linalg.inv(Q)
    cc = np.dot(-inverse, m_4)
    
    
    return cc

def normalizePoints(points0):
    
    
    
    points = np.zeros(points0.shape)

    

    meanU = np.mean(points0[:,0])

    meanV = np.mean(points0[:,1])
    
    points = points0-[meanU, meanV]
    

    scale = 1.0/np.std(points)
    
    
    norm = points * scale
    T = np.asarray([[scale, 0, 0], [0,scale,0], [0,0,1]]) @ np.asarray([[1,0,-meanU],[0,1,-meanV], [0,0,1]])
    

    return norm, T

    
    

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    a_temp = points_a.copy()
    b_temp = points_b.copy()
    A = []
    norm_a, T_a = normalizePoints(a_temp)
    norm_b, T_b = normalizePoints(b_temp)
    


    
    for i in range (0, norm_a.shape[0]):
        u = norm_a[i,0]
        u_prime = norm_b[i,0]
        v = norm_a[i,1]
        v_prime = norm_b[i,1]
        
        A.append([u*u_prime, u_prime*v, u_prime, v_prime*u, v_prime*v, v_prime, u, v])
    A = np.asarray(A)


    temp = -1 * np.ones(A.shape[0])

    F,_,_,_ = np.linalg.lstsq(A,temp, rcond=-1.0)

    
    
    F = np.append(F,1).reshape(3,3)
    
    U,S,Vh = np.linalg.svd(F)

    S = np.diag(S)
    S[2,2] = 0

    
    F_norm = U @ S @ Vh
    
    F = T_b.transpose() @ F_norm @ T_a


    return F

def getRandomElements(matches_a, matches_b):
    rand_a = []
    rand_b = []
    
    for i in range(0,9):
        index = np.random.randint(1,matches_a.shape[0])
        rand_a.append(matches_a[index,:])
        rand_b.append(matches_b[index,:])
    
    rand_a = np.asarray(rand_a)
    rand_b = np.asarray(rand_b)
    

    return rand_a, rand_b
    
    

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    threshold = .05
    best_num_inliers = -1
    for i in range(0, 2000):
        rand_a, rand_b = getRandomElements(matches_a, matches_b)
        F_matrix = estimate_fundamental_matrix(rand_a, rand_b)
        
        inliers_a_temp = []
        inliers_b_temp = []
        total_inliers = 0
        

        for j in range (0, matches_a.shape[0]):
            b_temp = np.append(matches_b[j,:],1)
            a_temp = np.append(matches_a[j,:],1)

            error = b_temp.T @ F_matrix @ a_temp
            if np.abs(error) < threshold:
                inliers_a_temp.append(matches_a[j,:])
                inliers_b_temp.append(matches_b[j,:])
                total_inliers += 1
                
            if total_inliers > best_num_inliers:
                best_num_inliers = total_inliers
                inliers_a = np.asarray(inliers_a_temp)
                inliers_b = np.asarray(inliers_b_temp)
                best_F = np.asarray(F_matrix)
                
                
        
    inliers_a = np.asarray(inliers_a)
            
    inliers_b = np.asarray(inliers_b)

    print(inliers_a.shape)


    return best_F, inliers_a, inliers_b