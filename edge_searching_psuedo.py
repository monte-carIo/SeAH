import cv2
import numpy as np
import sys
from curve_fitting import predict_curve, curve_fit, drawopenCV
import imageio
MIN_ANGLE_DIFF = 12
W_SIZE = 35
MIN_DENSITY=0.07
AVERAGE_DENSITY =0.1
DISTANCE = 10
MAX_NON_EDGE = 10

# from startingPointSearching import findStartingPoint

def findStartingPoint(mask_img, canny_img, img_draw, minA=0, maxA=32, startX=.35, startY = .92, debug = False):
    h_img, w_img = img_draw.shape[:2]
    x1, y1 = int(startX*w_img -W_SIZE/2), int(startY * h_img + W_SIZE/2)
    x2, y2 = x1+W_SIZE, y1-W_SIZE
    num_reg =  8
    vote=[(0,())]*num_reg
    for i in range(num_reg):
        # print('i',i)
        w_mask=  mask_img[y2:y1, x1:x2]
        w_edge=  canny_img[y2:y1, x1:x2]
        HTsegments = houghLines(w_edge, vote = 10, minl=15)
        if HTsegments is None:
            print('cannot find HT')
            y1 = y1-W_SIZE//2
            y2 = y2-W_SIZE//2
            minA +=5
            maxA +=5
            vote[i] = 0, ()
            if debug:
                cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,0,0), thickness=2)
                cv2.imshow('img', img_draw)
                # cv2.waitKey(0)
            continue
        HTsegments = HTsegments.reshape(-1,4)
        angles = np.arctan2(-(HTsegments[:,3]-HTsegments[:,1]), HTsegments[:,2]-  HTsegments[:,0]) / np.pi * 180 
        for j in range(len(angles)):
            if angles[j] < 0:
                angles[j] +=180
        # angles = angles[angles <36]
        fil = np.where((angles < maxA) & (angles > minA))
        angles = angles[fil]
        # print('min', minA)
        # print('max', maxA)

        if debug:
            cv2.imshow('mask', mask_img)
            cv2.imshow('w_mask', w_mask)
            cv2.imshow('w_edge', w_edge)
            drawSegments(HTsegments, prompt='hough')
        
        targetP, density =medianPoint(w_mask)
        
        # cv2.waitKey(0)
        if targetP != None and density > AVERAGE_DENSITY and len(angles): # NOTE: still need stricter conditions 
            same_dict = {angles[0]:0}
            if len(angles) > 1:
                same = lambda a,b : np.abs(a - b) < 5
                for a in range(len(angles)-1):
                    for b in range(a, len(angles)):
                        if same(angles[a],angles[b]):
                            if angles[a] not in same_dict.keys():
                                same_dict[angles[a]]=1
                            else:
                                same_dict[angles[a]]+=1
                            if angles[b] not in same_dict.keys():
                                    same_dict[angles[b]]=1
                            else:
                                same_dict[angles[b]]+=1
            # print(same_dict)
            # print('max', max(same_dict.values()))
            
            if debug:
                cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,255,255))
                cv2.imshow('img', img_draw)
                # cv2.waitKey(0)
            # print(x1, y1, x2, y2, np.mean(angles))
            # print('add', i)
            vote[i] = max(same_dict.values())+1, (x1, y1, x2, y2, np.mean(angles))
            # return x1, y1, x2, y2, np.mean(angles)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (200,200,200), thickness=2)
        else:
            
            vote[i] = 0, ()
            if debug:
                cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,0,0), thickness=2)
                cv2.imshow('img', img_draw)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (100,100,100), thickness= 2)
        y1 = y1-W_SIZE//2
        y2 = y2-W_SIZE//2
        minA +=5
        maxA +=5
        # cv2.waitKey(0)
    scores = [s[0] for s in vote]
    idMax = np.argmax(scores)
    return vote[idMax][1]

def hsv_mask(img, lower = (0,0,40), upper = (40,50,150)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def houghLines(img, vote = 10, minl=5):
        return cv2.HoughLinesP(img, 1, np.pi/180, vote, minLineLength=minl, maxLineGap=3)

def preprocessing(img):
    return cv2.bilateralFilter(img, 3, 75, 75) 



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
    global w_mask, w_img_left
    oldx1, oldy1= x1,y1
    diff_x, diff_y = new_p[0]- W_SIZE //2, new_p[1]- W_SIZE //2
    # print('new p', new_p)
    # print('old  pos ', x1, y1, x2, y2)
    # print('diff', diff_x, diff_y)
    x1 = x1 + diff_x + dx
    y1 = y1 + diff_y - dy
    x2, y2 = x1 + W_SIZE, y1 - W_SIZE

    if abs(x1-oldx1) + abs(y1-oldy1) < 5:
        targetP, _ = medianPoint(w_mask)
        if targetP is None:
            return None, None, None, None
        else:
            oldx1, oldy1= x1,y1
            diff_x, diff_y = targetP[0]- W_SIZE //2, targetP[1]- W_SIZE //2
            x1 = x1 + diff_x + dx
            y1 = y1 + diff_y - dy
            if abs(x1-oldx1) + abs(y1-oldy1) < 5:
                return None, None, None, None
            else:
                x2, y2 = x1 + W_SIZE, y1 - W_SIZE
                return x1, y1, x2, y2
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
        return None, None
    density = count / (w * h )
    if density < MIN_DENSITY:
        print('low segment density')
        return None, None
    else:
        x = np.median(xs).reshape(-1)
        y = np.quantile(ys, .15).reshape(-1)
        return (int(x),int(y)), density

def segs_to_points(segs, save = True, title='inner'):
    if len(segs) < 3:
        return None, None
        
    segs = np.array(segs)
    X = np.concatenate((segs[:, 0], segs[:, 2]) , axis = 0).reshape(-1, 1)
    y = np.concatenate((segs[:, 1], segs[:, 3]) , axis = 0).reshape(-1, 1)
    
    if save:
        with open(f'{title}.npy', 'wb') as f:
            np.save(f, (X,y))
        return X,y
    return X,y

def sliding_window(img, w_size = W_SIZE, debug = True, camRight=True):
    global w_mask
    if not camRight:
        img = cv2.flip(img, 1)
    ''' 1. Preprocessing '''
    img = cv2.bilateralFilter(img, 9, 75, 75)
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret3,otsu_thresh = cv2.threshold(gray,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # cv2.imshow('otsu ', otsu_thresh)

    ''' canny & hsv filter & strict ROI '''
    if not camRight:
        canny_img = cv2.Canny(img, 1, 100)
        # mixed_otsu_canny = cv2.bitwise_and(canny_img, otsu_thresh)
        # cv2.imshow('mixed ', mixed_otsu_canny)
        cv2.imshow('canny ', canny_img)
        mask_img = hsv_mask(img,  lower=(0,0,40), upper=(110,100,150))
        # points = np.array([[69/w_img, 1], [75/w_img, 717/h_img], [89/w_img, 711/h_img], [102/w_img, 704/h_img], \
        #     [99/w_img, 710/h_img], [130/w_img, 696/h_img], [142/w_img, 688/h_img], [250/w_img, 596/h_img], \
        #         [272/w_img, 566/h_img], [321/w_img, 468/h_img], [1,400/h_img], [1,1]])
        # points = np.array([[49/w_img, 1], [77/w_img, 709/h_img], [108/w_img, 704/h_img], \
        #         [139/w_img, 693/h_img], [232/w_img, 646/h_img],[315/w_img, 621/h_img],[1, 621/h_img],[1,1]])
        # points = np.array([[112/w_img, 1 ], [142/w_img, 717/h_img], [151/w_img, 713/h_img], \
        #     [166/w_img,703/h_img], [188/w_img, 694/h_img], [210/w_img, 678/h_img], [237/w_img, 659/h_img], \
        #         [262/w_img, 635/h_img], [1, 580/h_img], [1,1]])
        points = np.array([[0.25339367, 1.        ], [0.32126697, 0.99033149], [0.34162896, 0.98480663],\
                    [0.37556561, 0.97099448], [0.42533937, 0.95856354], [0.47511312, 0.93646409], [0.5361991 , 0.91022099],\
                    [0.59276018, 0.87707182], [1. ,        0.80110497], [1.        , 1.        ]]) 
        # print('points camleft', points)
    else:
        canny_img = cv2.Canny(img, 50, 200)
        points = np.array([[0.27970297, 1.        ], [0.3539604,  0.97237569], [0.37376238, 0.97099448],\
                            [0.41089109, 0.96546961], [0.54455446, 0.94337017], [0.68069307, 0.90883978],\
                            [1.        , 0.83701657], [1.        , 1.        ]])
        # points = np.array([[113/w_img, 1 ], [143/w_img, 704/h_img], 
        #            [151/w_img, 703/h_img], [166/w_img,699/h_img], [220/w_img, 683/h_img], [275/w_img, 658/h_img], [1, 606/h_img], [1,1]])
        mask_img = hsv_mask(img)
        # print('points camright', points)

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
    if not camRight:
        res = findStartingPoint(mask_img_draw, canny_img, img_draw, 0,35, startY=.99, debug=debug)
    else:
        res = findStartingPoint(mask_img_draw, canny_img, img_draw, startX=.38, startY=.98, debug=debug)
    if res == ():
        return None, None, None, None
    x1, y1, x2, y2, prev_angle = res
    # cv2.waitKey(0)

    init = True
    inner_segs, outer_segs,unknown_segs = [], [], []
    centerP = (w_img//4, h_img//2)
    hard1_filter_segments = None
    hard2_filter_segments = None
    psuedo_inners = []
    psuedo_outers = []
    non_edge_counter = 0
    def psuedo(newSegs:np.ndarray, fromInner = True):
        # global inner_segs, outer_segs, psuedo_inner, psuedo_outer
        newSegs = newSegs.reshape(-1,4)
        if fromInner:
            prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) 
            y = int(DISTANCE *np.sin(prev_angle_inner-np.pi/2))
            x = int(DISTANCE *np.cos(prev_angle_inner-np.pi/2))
            psuedo_outer = newSegs + [x,-y,x,-y]
            outer_segs.extend(psuedo_outer)
            psuedo_outers.extend(psuedo_outer)
            # print('tt',psuedo_outer)
            # print('pp',outer_segs)
        else:
            prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0])
            # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
            y = int(DISTANCE *np.sin(prev_angle_outer+np.pi/2))
            x = int(DISTANCE *np.cos(prev_angle_outer+np.pi/2))
            psuedo_inner = newSegs + [x,-y,x,-y]
            inner_segs.extend(psuedo_inner)
            psuedo_inners.extend(psuedo_inner)
    ''' Slide window '''
    for _ in range(100):

        if x1 == None:
            break
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
            if non_edge_counter >= MAX_NON_EDGE:
                break
            print('cannot find HT')
            non_edge_counter+=1
            targetP, _ = medianPoint(w_mask)
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
        print('angle, prev ang', angles, prev_angle)
        filter_idx = np.where((angles-prev_angle > -10 )& (angles-prev_angle < MIN_ANGLE_DIFF)) # TODO - bright intensity change a little
        soft_filter_segments = HTsegments[filter_idx]
        segments = soft_filter_segments
        print('sss', len(segments))

        # brightness intensity
        # for r,g,b in 
        # Y = 0.2126* R + 0.7152* G + 0.0722* B

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
                print('dist new point to outer:', dist2outer, update_outer)
                print('dist new point to inner:', dist2inner, update_inner)
                if update_inner and update_outer:
                    filter_idx= np.where(((np.asarray(dist2outer) < 5) | (np.asarray(dist2inner) < 5))) #more filter, lower should be close to outer
                # elif update_outer:
                #     filter_idx= np.where(((np.asarray(dist2outer) < 3))) 
                # else:
                #     filter_idx= np.where(((np.asarray(dist2inner) < 3)))
            # filter_idx= np.where(((np.asarray(dist2outer) < 3) | (np.asarray(dist2inner) < 3)))
            
                hard1_filter_segments = segments[filter_idx]
                segments = hard1_filter_segments
            else:
    
                segments_global = segments + [x1, y2, x1, y2]
                print('global segs', segments_global)
                angles_inner = np.arctan2(-(segments_global[:,1]-inner_segs[-1][3]), segments_global[:,0]-  inner_segs[-1][2]) / np.pi * 180 
                # for s in soft_filter_segments_global: # FIXME : not show all of them
                    # cv2.line(img, (inner_segs[-1][2], inner_segs[-1][3]), (s[0], s[1]), color = (0,125,255), thickness = 3)
                    # cv2.line(img, (outer_segs[-1][2], outer_segs[-1][3]), (s[0], s[1]), color = (125,0,255), thickness = 3)
                for i in range(len(angles_inner)):
                    if angles_inner[i] < 0:
                        angles_inner[i] +=180

                angles_outer = np.arctan2(-(segments_global[:,1]-outer_segs[-1][3]), segments_global[:,0]-  outer_segs[-1][2]) / np.pi * 180 
                for i in range(len(angles_outer)):
                    if angles_outer[i] < 0:
                        angles_outer[i] +=180
                # print('outer-prev',angles_outer-prev_angle)
                # print('inner-prev',angles_inner-prev_angle)
                print('angle inner', angles_inner)
                print('angle outer', angles_outer)
                prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                if prev_angle_outer < 0:
                    prev_angle_outer +=180
                if prev_angle_inner < 0:
                    prev_angle_inner +=180
                print('prev outer', prev_angle_outer)
                print('prev inner', prev_angle_inner)
                print('deviation outer angle', angles_outer-prev_angle_outer)
                print('deviation inner angle', angles_inner-prev_angle_inner)
                filter_idx = np.where(((angles_outer-prev_angle_outer > 1) & (angles_outer-prev_angle_outer < MIN_ANGLE_DIFF-3))
                    | ((angles_inner-prev_angle_inner > 1) & (angles_inner-prev_angle_inner < MIN_ANGLE_DIFF-3)) ) # TODO - bright intensity change a little, FIXME: 2 detected lines are shifted but 1 close to inner/outer
                hard2_filter_segments = segments[filter_idx]
                segments = hard2_filter_segments

        update_outer = update_inner = False
        
        # print('len seg', len(segments))
        ''' Split into 2 group '''
        if len(segments) ==0:# TODO: move base on previous inner, outer angle?
            if non_edge_counter >= MAX_NON_EDGE:
                break
            non_edge_counter+=1
            targetP, _ = medianPoint(w_mask)
            if targetP == None:
                break
            x1, y1, x2, y2 = sliding(targetP, x1, y1, x2, y2)
        else: 
            non_edge_counter=0
            if len(segments) == 1:
                p = segments[0]
                p_global = np.array(p).reshape(-1,4) + [x1, y2, x1, y2]
                # print('p_global', p_global)
                targetP, _ = medianPoint(w_mask)
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
                    unknown_segs.extend(p_global)
                    # outer_segs.extend(p_global)
                    # update_inner = True
                    # update_outer = True
                    # prev_angle_outer = np.arctan2(-(outer_segs[-1][3]-outer_segs[-1][1]), outer_segs[-1][2]-  outer_segs[-1][0]) / np.pi * 180 
                    # # prev_angle_inner = np.arctan2(-(inner_segs[-1][3]-inner_segs[-1][1]), inner_segs[-1][2]-  inner_segs[-1][0]) / np.pi * 180 
                    # y = DISTANCE *np.cos(prev_angle_outer+90)
                    # x = DISTANCE *np.sin(prev_angle_outer+90)
                    # psuedo_inner = p_global + [x,y,x,y]
                    # inner_segs.extend(psuedo_inner)
                    # psuedo_inners.extend(psuedo_inner)
                    # psuedo(p_global, fromInner = False)
                x1, y1, x2, y2 = sliding((segments[0][2:]+targetP)//2, x1, y1,x2,y2)
                
            else: 
                p = segments[0]
                other = segments[1:]
                dist2p = np.array([np.abs(dist3point(p[:2], p[2:], (op[:2]+op[2:])/2)) for op in other]) # absolute distance from p to other segs
                # print(dist2p)

                
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
                            
                    
                    # print('diff')

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
                            if len(unknown_segs):
                                radius_unknown = dist2point(centerP, (unknown_segs[-1][2:]+unknown_segs[-1][:2])/2)
                                if abs(radius_diff-radius_unknown) < (radius_same-radius_unknown):
                                    inner_segs.extend(unknown_segs)
                                    for u in unknown_segs:
                                        psuedo(u, fromInner = True)
                                else:
                                    outer_segs.extend(unknown_segs)

                                    for u in unknown_segs:
                                        psuedo(u, fromInner = False)
                                unknown_segs = []
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
                            if len(unknown_segs):
                                radius_unknown = dist2point(centerP, (unknown_segs[-1][2:]+unknown_segs[-1][:2])/2)
                                if abs(radius_diff-radius_unknown) < (radius_same-radius_unknown):
                                    outer_segs.extend(unknown_segs)
                                    for u in unknown_segs:
                                        psuedo(u, fromInner = False)
                                else:
                                    inner_segs.extend(unknown_segs)
                                    for u in unknown_segs:
                                        psuedo(u, fromInner = True)
                                unknown_segs = []
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
                            # unknown_segs.extend(samep_segs_global) # TODO: add a unknown_segs list and decide which type of it latter with inner and outer 
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
                        # outer_segs.extend(samep_segs_global) # TODO: add a unknown_segs list and decide which type of it latter with inner and outer 
                        # update_outer = True
                        # update_inner = True
                        unknown_segs.extend(samep_segs_global) # TODO: add a unknown_segs list and decide which type of it latter with inner and outer 
                        # psuedo(samep_segs_global, fromInner = False)
                    x1, y1, x2, y2 = sliding(mean_same[2:], x1, y1,x2,y2)
                        
                # mean_same = np.mean()
                # x1, y1, x2, y2 = slide(segments[0][2:], x1, y1,x2,y2)
            
            
    
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
            if hard1_filter_segments is not None:
                drawSegments(hard1_filter_segments, prompt='distance filter segs')
            if hard2_filter_segments is not None:
                drawSegments(hard2_filter_segments, prompt='angle filter segs')
            

            w_mask_draw = w_mask.copy()
            if medianPoint(w_mask) is not None:
                targetP,_ = medianPoint(w_mask)
            cv2.circle(w_mask_draw, targetP, color = (125,125,125), radius = 2)
            cv2.imshow('seg median', w_mask_draw)

            # draw inner/outer lines
            # viz_segs = np.zeros(img.shape)
            # viz_segs = img.copy()
            # viz_segs[:,:,0] = canny_img.copy()
            canny_img_brg = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
            viz_segs = cv2.add(img.copy(),canny_img_brg)    
            for i in unknown_segs:
                # print(i)
                cv2.circle(viz_segs, i[:2], color = (0,100, 255), radius = 2 , thickness = 2)
                cv2.circle(viz_segs, i[2:], color = (0,100, 255), radius = 2 , thickness = 2)
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
            if len(unknown_segs) > 0:
                cv2.line(viz_segs, centerP, (unknown_segs[-1][:2]+unknown_segs[-1][2:])//2, color = (0,125,0))
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

            # cv2.waitKey(0)
    
    X_in, y_in = segs_to_points(inner_segs, save=False) # use y axis to predict x axis
    X_out, y_out = segs_to_points(outer_segs, save=False)
    return X_in, y_in, X_out, y_out

def predict_curve_on_image(img, X_in, y_in, X_out, y_out):
    # inner
    viz = img.copy()
    y_t, X_t = predict_curve(y_in, X_in,1000)
    viz = drawopenCV(X_in, y_in, X_t, y_t,  viz)

    # outer
    y_t, X_t= predict_curve(y_out,X_out,1000)
    viz = drawopenCV(X_out, y_out, X_t, y_t,  viz, dataColor=(255,0,0), predictColor=(255,255,0))
    return viz

def all_in_one_viz(img_left, img_right):
    # left cam
    X_in, y_in, X_out, y_out = sliding_window(img_left, debug=False, camRight=False)
    if X_in is not None:
        img_left = cv2.flip(img_left, 1)     
        viz_left = predict_curve_on_image(img_left, X_in, y_in, X_out, y_out)
    else:
        img_left = cv2.flip(img_left, 1)     
        viz_left = img_left.copy()

    # right cam
    X_in, y_in, X_out, y_out = sliding_window(img_right, debug=False, camRight=True)
    if X_in is not None:
        viz_right = predict_curve_on_image(img_right, X_in, y_in, X_out, y_out)
    else:
        viz_right = img_right.copy()

    h_img_left, w_img_left = viz_left.shape[:2]
    h_img_right, w_img_right = viz_right.shape[:2]
    assert h_img_left == h_img_right, 'h_img_left != h_img_right'

    combine = np.zeros((h_img_left, w_img_left+w_img_right, 3), dtype=np.uint8)
    combine[:, :w_img_left] = cv2.flip(viz_left,1)
    combine[:, w_img_left:w_img_left+w_img_right] = viz_right
    
    return combine
def process_pair_image(range_imgs= list(range(0,1020,10))):
    impath = '/home/pronton/SeAH/data/cam3_20230907_143145/'
    img_left_paths = [impath + str(idx) + '.jpg' for idx in range_imgs]
    impath = '/home/pronton/SeAH/data/cam4_20230907_143145/'
    img_right_paths = [impath + str(idx) + '.jpg' for idx in range_imgs]
    image_lst=[]
    for i, (lp, rp) in enumerate(zip(img_left_paths, img_right_paths)):
        img_left = cv2.imread(lp) 
        img_right = cv2.imread(rp) 

        # scale for fast run speed
        scale = .5
        img_left = cv2.resize(img_left, (0,0), fx = scale, fy = scale)
        img_right = cv2.resize(img_right, (0,0), fx = scale, fy = scale)

        viz_combine = all_in_one_viz(img_left, img_right)
        cv2.imshow('combine', viz_combine)
        cv2.imwrite(f'output/combine/{range_imgs[i]}.jpg', viz_combine)
        image_lst.append(cv2.cvtColor(viz_combine, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
    # imageio.mimsave('demo_6fps.gif', image_lst, fps=6)
    # with imageio.get_writer("smiling.gif", mode="I") as writer:
    #     for idx, frame in enumerate(frames):
    #         print("Adding frame to GIF file: ", idx + 1)
    #         writer.append_data(frame)

if __name__ == '__main__':

    # vid_left = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi') 
    # vid_right = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi') 
    process_pair_image()

    # convert_to_points(inner_segs, save=True, title = 'inner_l')
    # convert_to_points(outer_segs, save=True, title = 'outer_l')
    # convert_to_points(inner_segs2, save=True, title = 'inner_r')
    # convert_to_points(outer_segs2, save=True, title = 'outer_r')
    # img_right = cv2.imread('/home/pronton/SeAH/data/cam4_20230907_143145/700.jpg'
    # X_in, y_in, X_out, y_out = sliding_window(img_right, debug=False, camRight=True)
