import torch
import math

def rowswap(M, idxSourceRow, idxTargetRow):
    """
    Swap rows in a matrix.
    
    Args:
        M (matrix/pytorch tensor): Matrix of values
        idxSourceRow (int): Index value of one of the rows to be swapped (indexing from 1)
        idxTargetRow (int): Index value of one of the rows to be swapped (indexing from 1)

    Returns:
        mat: Matrix with the indicated rows swapped
    
    Example:
        >>> rowswap(torch.tensor([[1,2,3],[4,5,6]]), 1,2)
        tensor([[4,5,6],[1,2,3]])
    """
    # print("DOING ROW SWAP")
    mat = M.clone()
    # print('swap: ', idxSourceRow, idxTargetRow)
    temptarget = M[idxTargetRow-1, :]
    mat[idxTargetRow-1] = M[idxSourceRow-1]
    mat[idxSourceRow-1] = temptarget
    return mat

def rowscale(M, idxRow, scaleFactor):
    """
    Scale one row of matrix by scalar value.
    
    Args:
        M (matrix/pytorch tensor): Matrix of values
        idxRow (int): Index (starting from 1) of row to be scaled
        scaleFactor (number int or float): Value to scale each element in the row by 

    Returns:
        mat: Matrix with the indicated row scaled

    Example:
        >>> rowscale(torch.tensor([[1,2,3],[4,5,6]]), 1,5)
        tensor([[5,10,15],[4,5,6]])
    """
    # print('DOING ROW SCALE')
    M[idxRow-1] = M[idxRow-1, :]*scaleFactor
    return M

def rowreplacement(M, idxFirstRow, idxSecondRow, scaleJ, scaleK):
    """
    Replace row in matrix by scaling and adding to other row.

    Args:
        M (matrix/pytorch tensor): Matrix of values
        idxFirstRow (int): Index of the row to be updated
        idxSecondRow (int): Index of the row to be added into the updated row
        scaleJ (number int or float): Scale factor used for the firstRow elements
        scaleK (number int or float): Scale factor used for the secondRow elements

    Returns:
        mat: Matrix with the indicated row scaled and summed with the scale of the second row

    Example:
        >>> rowreplacement(torch.tensor([[1,2,3],[4,5,6]]), 1,2,2,1)
        tensor([[6,9,12],[4,5,6]])
    """
    # print('DOING ROW REPLACE')
    # print(M, idxFirstRow, idxSecondRow, scaleJ, scaleK)
    mat = M.clone()
    firstScaled = rowscale(mat, idxFirstRow, scaleJ)[idxFirstRow-1]
    secondScaled = rowscale(mat, idxSecondRow, scaleK)[idxSecondRow-1]
    firstScaled = firstScaled + secondScaled
    M[idxFirstRow-1] = firstScaled
    return M

def rref(M):
    """
    Calculate reduced row echelon form of a matrix

    Args:
        M (matrix/pytorch tensor): Matrix of values

    Returns:
        mat: Matrix in rref
    
    Example:
        >>> rref(torch.tensor([[1,2,3],[4,5,6]]))
        tensor([[ 1,  0, -1],[ 0,  1,  2]])
    """
    mat = M.clone()
    pivotFound = False
    totalRows= mat.shape[0]
    # print(totalRows)
    manualRowidx = 0
    for colidx in range(len(mat[0])):
        if manualRowidx==totalRows:
            break
        for rowidx in range(manualRowidx, totalRows):
            # print(rowidx, colidx)
            # print(mat[rowidx, colidx].item())
            if mat[rowidx, colidx].item() == 1 or mat[rowidx, colidx].item() == 1.0 :
                if rowidx>manualRowidx:
                    mat = rowswap(mat, rowidx+1, manualRowidx+1)
                    # print(mat)
                    rowidx = manualRowidx
                pivot = [rowidx,colidx]
                # mat = reduceCol(mat, pivot)
                pivotFound = True
                break
        if pivotFound == False: #no value in column = 1
            if mat[manualRowidx, colidx].item()==0:
                for rowidx in range(manualRowidx, totalRows):
                    if mat[rowidx,colidx].item() != 0:
                        mat = rowswap(mat, manualRowidx+1, rowidx+1)
                        # print(mat)
                        pivotFound=True
                        pivot = [manualRowidx, colidx]
                        break
            else: #current pivot item not 0
                pivotFound = True
                pivot = [manualRowidx, colidx]
        if pivotFound ==True:
            mat = reduceCol(mat, pivot)
            manualRowidx+=1
            pivotFound = False
        # print('*'*50)
        # print(mat)
    # print()
    return mat



def reduceCol(M, pivot):
    """
    Reduces the column of a matrix based on pivot value to acheive rref.

    Args:
        M (matrix/pytorch tensor): Matrix of values
        pivot ([int, int]): List (len=2) of integer values representing the row, column index for the current pivot

    Returns:
        mat: Matrix with the column of focus reduced

    Example:
        >>> reduceCol(torch.tensor([[1,2],[3,4]]), [1,1])
        tensor([[1,0],[1,0]])
    """
    # print('doing reduction')
    pivotRow = pivot[0]
    pivotCol = pivot[1]

    # if M[pivotRow,pivotCol].item() == 0:
    #     pivotCol+=1

    if M[pivotRow, pivotCol].item() != 1:
        try:
            scale = 1/M[pivotRow, pivotCol].item()
            # print(scale)
            M = rowscale(M, pivotRow+1, scale)
        except:
            pass


    for rowidx in range(len(M)):
        if rowidx != pivotRow:
            # print(M[rowidx, pivotCol].item())
            if M[rowidx,pivotCol].item() ==0:
                pass
            else:
                fractionCheck, _ = math.modf(M[rowidx, pivotCol].item())
                if fractionCheck != 0: #check if its a fraction 
                    scale = M[rowidx, pivotCol].item()
                    # print(scale)
                    M = rowreplacement(M, rowidx+1, pivotRow+1, 1, scale*-1)
                else:
                    scale = M[rowidx, pivotCol].item()/1
                    # print(M, rowidx, pivotRow, 1, scale*-1)
                    M = rowreplacement(M, rowidx+1, pivotRow+1, 1, scale*-1)
    return M


# Matrix = torch.tensor([[0,1,1],[1,1,6],[2,1,8]], dtype=torch.float16)

# print(rowswap(Matrix, 4,1))
# print(rowscale(Matrix, 1, 10))
# print(rowreplacement(Matrix, 1,6,16,-1))



# Matrix = torch.tensor([[1,3,0,0,3], [0,0,1,0,9], [0,0,0,1,-4]], dtype=torch.float16)
# print(Matrix)
# M = rowswap(Matrix, 1,2)
# print(M)
# M = rowscale(M, 1,(1/3))
# print(M)
# M = rowreplacement(M, 3, 1, 1, -3)
# print(M)

# Matrix = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float16)


# print(rref(Matrix))
