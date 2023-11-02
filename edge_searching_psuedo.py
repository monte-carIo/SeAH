import cv2
import numpy as np
import sys
MIN_ANGLE_DIFF = 12
W_SIZE = 35
MIN_DENSITY=0.07
DISTANCE = 10

def findStartingPoint(mask_img, canny_img, img_draw):
    
    minA = 0
    maxA = 32
    startX, startY = .35 , .92 #TODO - not fixed but auto detect starting point
    h_img, w_img = img_draw.shape[:2]
    x1, y1 = int(startX*w_img -W_SIZE/2), int(startY * h_img + W_SIZE/2)
    x2, y2 = x1+W_SIZE, y1-W_SIZE
    for i in range(10):
        w_mask=  mask_img[y2:y1, x1:x2]
        w_edge=  canny_img[y2:y1, x1:x2]
        # cv2.imshow('w_mask', w_mask)
        # cv2.imshow('w_edge', w_edge)
        HTsegments = houghLines(w_edge, vote = 10, minl=5)
        if HTsegments is None:
            # print('cannot find HT')

            y1 = y1-W_SIZE//2
            y2 = y2-W_SIZE//2
            minA +=5
            maxA +=5
            continue
        HTsegments = HTsegments.reshape(-1,4)
        angles = np.arctan2(-(HTsegments[:,3]-HTsegments[:,1]), HTsegments[:,2]-  HTsegments[:,0]) / np.pi * 180 
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] +=180
        # print('angle', angles)
        # angles = angles[angles <36]
        fil = np.where((angles < maxA) & (angles > minA))
        angles = angles[fil]
        # print('min', minA)
        # print('max', maxA)
        

        # drawSegments(HTsegments, prompt='hough')
        
        targetP=medianPoint(w_mask)
        
        cv2.waitKey(0)
        if targetP != None and len(angles): # NOTE: still need stricter conditions 
            # cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,255,255))
            # cv2.imshow('img', img_draw)
            # cv2.waitKey(0)
            return x1, y1, x2, y2, np.mean(angles)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (200,200,200), thickness=2)
        else:
            
            pass
            # cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,0,0), thickness=2)
            # cv2.imshow('img', img_draw)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (100,100,100), thickness= 2)
        y1 = y1-W_SIZE//2
        y2 = y2-W_SIZE//2
        minA +=5
        maxA +=5
        cv2.waitKey(0)
    return None
    

def hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0,0,40)
    upper = (40,50,150)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def houghLines(img, vote = 10, minl=5):
        return cv2.HoughLinesP(img, 1, np.pi/180, vote, minLineLength=minl, maxLineGap=3)

def preprocessing(img):
    return cv2.bilateralFilter(img, 9, 75, 75) 



def drawSegments(segs, col = 255, prompt ='', viz_box=None):
    viz_box_None = False
    if viz_box is None:
        viz_box = np.zeros((W_SIZE, W_SIZE),  np.uint8)
        viz_box_None = True
    for s in segs:
        cv2.line(viz_box, s[:2], s[2:], color=col, thickness=2)
    if viz_box_None:
        cv2.imshow(prompt, viz_box)
    else:
        return viz_box

def sliding(new_p, x1, y1, x2, y2, dx =0, dy = 0):
    '''
    dx, dy: velocity factor
    '''
    diff_x, diff_y = new_p[0]- W_SIZE //2, new_p[1]- W_SIZE //2
    # print('new p', new_p)
    # print('old  pos ', x1, y1, x2, y2)
    # print('diff', diff_x, diff_y)
    x1 = x1 + diff_x + dx
    y1 = y1 + diff_y - dy
    x2, y2 = x1 + W_SIZE, y1 - W_SIZE
    # print('new  pos ', x1, y1, x2, y2)
    return x1, y1, x2, y2

def dist3point(p1,p2,p3):
    return np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

def dist2point(p1,p2):
    return np.linalg.norm(p1 - p2)

def medianPoint(img):
    w,h = img.shape[:2]
    xs = []
    ys = []
    count = 0
    for x in range(w):
        for y in range(h):
            try:
                if img[y][x]:
                    xs.append(x)
                    ys.append(y)
                    count +=1
            except:
                pass
    # skeleton = skimage.morphology.skeletonize(veins)
    if w == 0 or h==0:
        print('not valid mask')
        return None
    density = count / (w * h )
    if density < MIN_DENSITY:
        print('low segment density')
        return None
    else:
        x = np.median(xs).reshape(-1)
        y = np.quantile(ys, .25).reshape(-1)
        return int(x),int(y)

def save_for_train(segs, title='inner'):
    segs = np.array(segs)
    X = np.concatenate((segs[:, 0], segs[:, 2]) , axis = 0).reshape(-1, 1)
    y = np.concatenate((segs[:, 1], segs[:, 3]) , axis = 0).reshape(-1, 1)
    
    with open(f'{title}.npy', 'wb') as f:
        np.save(f, (X,y))

def sliding_window(img, w_size = W_SIZE, debug = True):
    ''' 1. Preprocessing '''
    img = cv2.bilateralFilter(img, 9, 75, 75)
    scale = .5
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)
    h_img, w_img = img.shape[:2]

    ''' canny '''
    canny_img = cv2.Canny(img, 30, 200)

    ''' hsv_mask '''
    mask_img = hsv_mask(img)

    ''' strict ROI '''
    points = np.array([[72/w_img, 1 ], [92/w_img, 692/h_img], 
                   [98/w_img, 687/h_img], [116/w_img,681/h_img], [132/w_img, 671/h_img], [160/w_img, 667/h_img], [1, 1]])
    points *= [w_img, h_img]

    cv2.fillPoly(canny_img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.fillPoly(mask_img, pts=[points.astype(int)], color=(0, 0, 0))


    ''' 2. Sliding Window '''

    ''' Initialize window starting point '''
    w_size = 35
    startX, startY = .193 , .93 #TODO - not fixed but auto detect starting point
    samep_segs, diffp_segs = [], []
    '''Find Starting Point'''
    # x1, y1 = int(startX*w_img -w_size/2), int(startY * h_img + w_size/2)
    # x2, y2 = x1+w_size, y1-w_size
    mask_img_draw = mask_img.copy()
    img_draw = img.copy()
    x1, y1, x2, y2, prev_angle = findStartingPoint(mask_img_draw, canny_img, img_draw)
    cv2.waitKey(0)

    init = True
    inner_segs = []
    outer_segs = []
    centerP = (w_img//4, h_img//2)
    hard_filter_segments = None
    psuedo_inners = []
    psuedo_outers = []
    def psuedo(newSegs:np.ndarray, fromInner = True):
        # global inner_segs, outer_segs, psuedo_inner, psuedo_outer
        if fromInner:
            prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) 
            y = int(DISTANCE *np.sin(prev_angle_inner-np.pi/2))
            x = int(DISTANCE *np.cos(prev_angle_inner-np.pi/2))
            psuedo_outer = newSegs + [x,-y,x,-y]
            # print('MMMMM', prev_angle_inner / np.pi * 180, prev_angle_inner / np.pi * 180 - 90)
            outer_segs.extend(psuedo_outer)
            psuedo_outers.extend(psuedo_outer)
        else:
            prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0])
            # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
            y = int(DISTANCE *np.sin(prev_angle_outer+np.pi/2))
            x = int(DISTANCE *np.cos(prev_angle_outer+np.pi/2))
            # print('-=-=-=', prev_angle_outer / np.pi * 180, prev_angle_outer / np.pi * 180 + 90)
            psuedo_inner = newSegs + [x,-y,x,-y]
            inner_segs.extend(psuedo_inner)
            psuedo_inners.extend(psuedo_inner)
    ''' Slide window '''
    for _ in range(100):

        # print('>>>>>>>>>>>>>>')
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0))

        kernel = np.ones((5, 5), np.uint8) 
        w_edge=  canny_img[y2:y1, x1:x2]
        w_mask=  mask_img[y2:y1, x1:x2]
        # w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_CLOSE,kernel, iterations=1) #FIXME: Incorrect 
        # w_edge_exclusive = cv2.bitwise_and(w_edge, w_mask)
        # w_edge_morpho = cv2.morphologyEx(w_edge_exclusive, cv2.MORPH_CLOSE,kernel, iterations=1) #FIXME: Incorrect 
        # w_edge = w_edge_morpho
        
        
        ''' Finding segments '''
        HTsegments = houghLines(w_edge)
        if HTsegments is None:
            # print('--------------------',  medianPoint(w_mask))
            targetP = medianPoint(w_mask)
            if targetP is None:
                break
            x1, y1, x2, y2 = sliding(targetP, x1, y1, x2, y2) # TODO: maybe this let x1,y1,x2,y2 out of range
            continue
        HTsegments = HTsegments.reshape(-1,4)

        ''' Filter segments '''

        # soft filter

        angles = np.arctan2(-(HTsegments[:,3]-HTsegments[:,1]), HTsegments[:,2]-  HTsegments[:,0]) / np.pi * 180 
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] +=180
        # print('angle', angles)
        # print('prev angle', prev_angle)
        # print('diff angle', angles-prev_angle)
        filter_idx = np.where((angles-prev_angle > 0-5) & (angles-prev_angle < MIN_ANGLE_DIFF)) # TODO - bright intensity change a little
        soft_filter_segments = HTsegments[filter_idx]
        segments = soft_filter_segments
        angles = angles[filter_idx]
        prev_angle = np.mean(angles, axis= 0) if len(angles) else prev_angle # update previous angle
        for s in segments:
            if s[1] < s[3]: # y1 is higher than y2 => swap
                s[0],s[1],s[2],s[3] = s[2],s[3],s[0], s[1]
        #  harder filter
        
        if len(inner_segs) and len(outer_segs): #TODO: or , not and

            if update_inner or update_outer: # FIXME: thieu dk soft filter, latter segs too depend on this -> must correct at the first inner outer seg
                dist2outer = []
                dist2inner = []
                for s in segments:
                    s_global = s + [x1, y2, x1, y2]
                    dist2inner.append(np.abs(dist3point(inner_segs[-1][:2], inner_segs[-1][2:], (s_global[:2]+s_global[2:])//2)))
                    dist2outer.append(np.abs(dist3point(outer_segs[-1][:2], outer_segs[-1][2:], (s_global[:2]+s_global[2:])//2)))
                # print('dist new point to outer:', dist2outer, update_outer)
                # print('dist new point to inner:', dist2inner, update_inner)
                if update_inner and update_outer:
                    filter_idx= np.where(((np.asarray(dist2outer) < 3) | (np.asarray(dist2inner) < 3)))
                elif update_outer:
                    filter_idx= np.where(((np.asarray(dist2outer) < 3))) 
                else:
                    filter_idx= np.where(((np.asarray(dist2inner) < 3)))
            # filter_idx= np.where(((np.asarray(dist2outer) < 3) | (np.asarray(dist2inner) < 3)))
            
                hard_filter_segments = soft_filter_segments[filter_idx]
                segments = hard_filter_segments
            else:
    
                soft_filter_segments_global = soft_filter_segments + [x1, y2, x1, y2]
                angles_inner = np.arctan2(-(soft_filter_segments_global[:,1]-inner_segs[-1][3]), soft_filter_segments_global[:,0]-  inner_segs[-1][2]) / np.pi * 180 
                # for s in soft_filter_segments_global: # FIXME : not show all of them
                    # cv2.line(img, (inner_segs[-1][2], inner_segs[-1][3]), (s[0], s[1]), color = (0,125,255), thickness = 3)
                    # cv2.line(img, (outer_segs[-1][2], outer_segs[-1][3]), (s[0], s[1]), color = (125,0,255), thickness = 3)
                for i in range(len(angles_inner)):
                    if angles_inner[i] < 0:
                        angles_inner[i] +=180

                angles_outer = np.arctan2(-(soft_filter_segments_global[:,1]-outer_segs[-1][3]), soft_filter_segments_global[:,0]-  outer_segs[-1][2]) / np.pi * 180 
                for i in range(len(angles_outer)):
                    if angles_outer[i] < 0:
                        angles_outer[i] +=180
                # print('outer-prev',angles_outer-prev_angle)
                # print('inner-prev',angles_inner-prev_angle)
                prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                # print('prev outer', prev_angle_outer)
                # print('prev inner', prev_angle_inner)
                filter_idx = np.where(((angles_outer-prev_angle_outer > 0-5))
                    | ((angles_inner-prev_angle_inner > 0-5) & (angles_inner-prev_angle < MIN_ANGLE_DIFF)) ) # TODO - bright intensity change a little
                hard_filter_segments = soft_filter_segments[filter_idx]
                segments = hard_filter_segments

        update_outer = update_inner = False
        
        ''' Split into 2 group '''
        if len(segments) >=2:
            p = segments[0]
            other = segments[1:]
            dist2p = np.array([np.abs(dist3point(p[:2], p[2:], (op[:2]+op[2:])/2)) for op in other]) # absolute distance from p to other segs

            
            diff_idx = np.where((dist2p > 8) & (dist2p < 20)) # TODO - bright intensity change a little FIXME: not correct, in diff, there may be some diff between them -> check it and filter
            diffp_segs = other[diff_idx]
            # diffp_segs = diffp_segs[dist2p < 8]
            samep_segs = other[dist2p < 3.3]

            # print('dist', dist2p)

            # if len()
            # p2 = diffp_segs[0]
            # diffp_other = segments[1:]
            # dist2p = np.array([dist3point(p2[:2], p2[2:], (op[:2]+op[2:])/2) for op in diffp_other]) # absolute distance from p to other segs

            

            samep_segs = np.concatenate((samep_segs.reshape(-1,4), p.reshape(-1,4)), axis= 0)

            # mean_same = np.mean(samep_segs, axis= 0, dtype=int)
            idx_max = np.argmin(samep_segs[:,3]) # y min -> highest
            # print('same', samep_segs)
            mean_same = samep_segs[idx_max]
            # print('max', mean_same)
            center_mean_same_global = (mean_same[:2] + mean_same[2:])//2  + [x1,y2]
            radius_same = dist2point(centerP, center_mean_same_global)
            samep_segs_global = samep_segs  + [x1,y2,x1,y2]

            if len(diffp_segs):
                # mean_diff = np.mean(diffp_segs, axis= 0)
                idx_max = np.argmin(diffp_segs[:,3]) # y min -> highest
                mean_diff = diffp_segs[idx_max]
                mean_ = np.mean((mean_same, mean_diff), axis= 0, dtype = int)
                diffp_segs_global = diffp_segs  + [x1,y2,x1,y2]
                center_mean_diff_global = (mean_diff[:2] + mean_diff[2:])//2 + [x1,y2]
                radius_diff = dist2point(centerP, center_mean_diff_global)

                # print('center_dist', radius_diff, radius_same)
                if radius_diff  < radius_same:
                    # if len(inner_segs) and len(outer_segs): # or use diff angle
                    #     dist2outer = np.abs(dist3point(outer_segs[-1][:2], outer_segs[-1][2:], mean_samep_global[:2]))
                    #     dist2inner = np.abs(dist3point(inner_segs[-1][:2], inner_segs[-1][2:], mean_diffp_global[:2]))
                        # print('dist new point to outer:', dist2outer)
                    #     print('dist new point to inner:', dist2inner)
                    #     if dist2outer < 5:
                    #         outer_segs.extend(samep_segs_global)
                    #     if dist2inner < 5:
                    #         inner_segs.extend(diffp_segs_global)
                    # else:
                        inner_segs.extend(diffp_segs_global)
                        outer_segs.extend(samep_segs_global)
                        update_inner = True
                        update_outer = True
                else:
                    # if len(inner_segs) and len(outer_segs):
                    #     dist2outer = np.abs(dist3point(outer_segs[-1][:2], outer_segs[-1][2:], mean_diffp_global[:2]))
                    #     dist2inner = np.abs(dist3point(inner_segs[-1][:2], inner_segs[-1][2:], mean_samep_global[:2]))
                    #     print('dist new point to outer:', dist2outer)
                    #     print('dist new point to inner:', dist2inner)
                    #     if dist2outer < 5:
                    #         outer_segs.extend(diffp_segs_global)
                    #     if dist2inner < 5:
                    #         inner_segs.extend(samep_segs_global)
                    # else:
                        inner_segs.extend(samep_segs_global)
                        outer_segs.extend(diffp_segs_global)
                        update_inner = True
                        update_outer = True
                # if len(inner_segs) and len(outer_segs):
                #     mean_samep_global = mean_same  + [x1,y2,x1,y2]
                #     dist2inner = dist3point(inner_segs[-1][:2], inner_segs[-1][2:], mean_samep_global[:2])
                #     dist2outer = dist3point(outer_segs[-1][:2], outer_segs[-1][2:], mean_samep_global[:2])
                #     # if len(inner_segs) >1 and len(outer_segs) >1:
                #     #     dist2inner_2 = dist3point(inner_segs[-2][:2], inner_segs[-2][2:], mean_samep_global[:2])
                #     #     dist2inner = np.mean((dist2inner, dist2inner_2))
                #     #     dist2outer_2 = dist3point(outer_segs[-2][:2], outer_segs[-2][2:], mean_samep_global[:2])
                #     #     dist2outer = np.mean((dist2outer, dist2outer_2))
                #     print('dist2inner', dist2inner)
                #     print('dist2outer', dist2outer)
                #     if dist2inner < dist2outer:
                #         inner_segs.extend(samep_segs_global)
                #         outer_segs.extend(diffp_segs_global)
                #     else:
                #         outer_segs.extend(samep_segs_global)
                #         inner_segs.extend(diffp_segs_global)
                # else:
                #     center_mean_diff_global = (mean_diff[:2] + mean_diff[2:])//2 + [x1,y2]
                #     radius_diff = dist2point(centerP, center_mean_diff_global)

                #     print('center_dist', radius_diff, radius_same)
                #     if radius_diff  < radius_same:
                #         inner_segs.extend(diffp_segs_global)
                #         outer_segs.extend(samep_segs_global)
                #     else:
                #         inner_segs.extend(samep_segs_global)
                #         outer_segs.extend(diffp_segs_global)
                x1, y1, x2, y2 = sliding(mean_[2:], x1, y1,x2,y2)
                
            else:
                if len(inner_segs) and len(outer_segs):
                    # dist2center = dist2point((mean_same[:2]+mean_same[2:])//2, centerP) # assume centerp is center of curve but it is not guarantee
                    mean_samep_global = mean_same  + [x1,y2,x1,y2]
                    dist2inner = dist3point(inner_segs[-1][:2], inner_segs[-1][2:], mean_samep_global[:2])
                    dist2outer = dist3point(outer_segs[-1][:2], outer_segs[-1][2:], mean_samep_global[:2])
                    # print('dist2inner', dist2inner)
                    # print('dist2outer', dist2outer)
                    if dist2inner < dist2outer:
                        inner_segs.extend(samep_segs_global)
                        update_inner = True
                        update_outer = True
                        # psuedo_outer = samep_segs_global + [10,0,10,0]
                        # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                        # y = DISTANCE *np.cos(prev_angle_inner-90)
                        # x = DISTANCE *np.sin(prev_angle_inner-90)
                        # psuedo_inner = samep_segs_global + [x,y,x,y]
                        # outer_segs.extend(psuedo_outer)
                        # psuedo_outers.extend(psuedo_outer)
                        psuedo(samep_segs_global, fromInner = True)
                    else:
                        outer_segs.extend(samep_segs_global)
                        update_outer = True
                        update_inner = True
                        psuedo(samep_segs_global, fromInner = False)
                        # prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                        # # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                        # y = DISTANCE *np.cos(prev_angle_outer+90)
                        # x = DISTANCE *np.sin(prev_angle_outer+90)
                        # psuedo_inner = samep_segs_global + [x,y,x,y]

                        # outer_segs.extend(psuedo_inner)
                        # psuedo_inners.extend(psuedo_inner)
                else: # outer seg is more likely detected by Canny than inner seg
                    # pass
                    outer_segs.extend(samep_segs_global) # TODO: add a unknown_segs list and decide which type of it latter with inner and outer 
                    update_outer = True
                    update_inner = True
                    psuedo(samep_segs_global, fromInner = False)
                x1, y1, x2, y2 = sliding(mean_same[2:], x1, y1,x2,y2)
                    
            # mean_same = np.mean()
            # x1, y1, x2, y2 = slide(segments[0][2:], x1, y1,x2,y2)
            
        elif len(segments) == 1:
            p = segments[0]
            p_global = np.array(p).reshape(-1,4) + [x1, y2, x1, y2]
            # print('p_global', p_global)
            targetP = medianPoint(w_mask)
            if targetP is None:
                break


            if len(inner_segs) and len(outer_segs):
                dist2inner = dist3point(inner_segs[-1][:2], inner_segs[-1][2:], p_global[0][:2])
                dist2outer = dist3point(outer_segs[-1][:2], outer_segs[-1][2:], p_global[0][:2])
                if dist2inner < dist2outer:
                    inner_segs.extend(p_global)
                    update_inner = True
                    update_outer = True
                    # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                    # y = DISTANCE *np.cos(prev_angle_inner-90)
                    # x = DISTANCE *np.sin(prev_angle_inner-90)
                    # psuedo_outer = p_global + [x,y,x,y]
                    # outer_segs.extend(psuedo_outer)
                    # psuedo_outers.extend(psuedo_outer)
                    psuedo(p_global, fromInner = True)
                else:
                    outer_segs.extend(p_global)
                    update_inner = True
                    update_outer = True

                    psuedo(p_global, fromInner = False)
                    # prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                    # # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                    # y = DISTANCE *np.cos(prev_angle_outer+90)
                    # x = DISTANCE *np.sin(prev_angle_outer+90)
                    # psuedo_inner = p_global + [x,y,x,y]
                    # inner_segs.extend(psuedo_inner)
                    # psuedo_inners.extend(psuedo_inner)
            else: # outer seg is more likely detected by Canny than inner seg
                outer_segs.extend(p_global)
                update_inner = True
                update_outer = True
                # prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                # # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                # y = DISTANCE *np.cos(prev_angle_outer+90)
                # x = DISTANCE *np.sin(prev_angle_outer+90)
                # psuedo_inner = p_global + [x,y,x,y]
                # inner_segs.extend(psuedo_inner)
                # psuedo_inners.extend(psuedo_inner)
                psuedo(p_global, fromInner = False)
            x1, y1, x2, y2 = sliding((segments[0][2:]+targetP)//2, x1, y1,x2,y2)
            
        else: # move base on previous inner, outer angle?
            if medianPoint(w_mask) is None:
                break
            targetP = medianPoint(w_mask)
            x1, y1, x2, y2 = sliding(targetP, x1, y1, x2, y2)
    
        if debug:
            cv2.imshow('canny', canny_img)
            cv2.imshow('mask', mask_img)
            # cv2.imshow('img', img)
            cv2.imshow('w edge', w_edge)
            # cv2.imshow('w mask', w_mask)
            # cv2.imshow('w_edge_exclusive', w_edge_exclusive) #

            # cv2.imshow('w_edge_morpho', w_edge_morpho) #
            drawSegments(HTsegments, prompt='all segs')
            drawSegments(soft_filter_segments, prompt='soft filter segs')
            if hard_filter_segments is not None:
                drawSegments(hard_filter_segments, prompt='hard filter segs')
            

            w_mask_draw = w_mask.copy()
            if medianPoint(w_mask) is not None:
                targetP = medianPoint(w_mask)
            cv2.circle(w_mask_draw, targetP, color = (125,125,125), radius = 2)
            cv2.imshow('seg median', w_mask_draw)

            # draw inner/outer lines
            # viz_segs = np.zeros(img.shape)
            # viz_segs = img.copy()
            # viz_segs[:,:,0] = canny_img.copy()
            canny_img_brg = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
            viz_segs = cv2.add(img.copy(),canny_img_brg)    
            for i in inner_segs:
                # print(i)
                cv2.circle(viz_segs, i[:2], color = (100,0, 255), radius = 2 , thickness = 2)
                cv2.circle(viz_segs, i[2:], color = (100,0, 255), radius = 2 , thickness = 2)
            for i in outer_segs:
                # print(i)
                cv2.circle(viz_segs, i[:2], color = (255,100,0), radius = 2 , thickness = 2)
                cv2.circle(viz_segs, i[2:], color = (255,100,0), radius = 2 , thickness = 2)
            for i in psuedo_inners:
                cv2.circle(viz_segs, i[:2], color = (100,0,125), radius = 2 , thickness = 2)
                cv2.circle(viz_segs, i[2:], color = (100,0,125), radius = 2 , thickness = 2)

            for i in psuedo_outers:
                cv2.circle(viz_segs, i[:2], color = (125,255,0), radius = 2 , thickness = 2)
                cv2.circle(viz_segs, i[2:], color = (125,255,0), radius = 2 , thickness = 2)

            # draw distance
            if len(inner_segs) > 0:
                cv2.line(viz_segs, centerP, (inner_segs[-1][:2]+inner_segs[-1][2:])//2, color = (0,0,125))
            if len(outer_segs) > 0:
                cv2.line(viz_segs, centerP, (outer_segs[-1][:2]+outer_segs[-1][2:])//2, color = (125,0,0))
            cv2.imshow('segs', viz_segs)

            # viz_box = np.zeros((W_SIZE, W_SIZE))
            viz_box = np.zeros((W_SIZE, W_SIZE), np.uint8)
            if len(samep_segs):
                viz_box = drawSegments(samep_segs, col = 200, viz_box=viz_box)
            if len(diffp_segs):
                viz_box = drawSegments(diffp_segs, col = 100, viz_box=viz_box)
                cv2.line(viz_box, mean_[:2], mean_[2:], color = 50)
                # viz_box = drawSegments(diffp_segs, prompt = 'diff', viz_box)
                
            
            cv2.imshow('mean_diff', viz_box)
            # print('--------------------')

            cv2.waitKey(0)
            
    return inner_segs, outer_segs
if __name__ == '__main__':

    # img = cv2.imread('/home/pronton/SeAH/data/cam4_20230907_143145/190.jpg') # img = cv2.imread(argv[1])
    # img = cv2.imread('/home/pronton/SeAH/data/cam4_20230907_143145/190.jpg') 
    img = cv2.imread(sys.argv[1])
    inner_segs, outer_segs = sliding_window(img, debug=True) #FIXME - stop without median condition
    
    # save
    if len(inner_segs) > 0:
        save_for_train(inner_segs, 'inner')
    if len(outer_segs) > 0:
        save_for_train(outer_segs, 'outer')