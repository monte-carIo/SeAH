import cv2
import numpy as np
MIN_ANGLE_DIFF = 10
W_SIZE = 35

def hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0,0,40)
    upper = (40,50,150)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def sliding_window(img):
    pass

def houghLines(img):
        return cv2.HoughLinesP(img, 2, np.pi/180, 10, minLineLength=5, maxLineGap=3)

def preprocessing(img):
    return cv2.bilateralFilter(img, 9, 75, 75) 

def drawSegments(segs, col = 20, prompt ='', viz_box=None):
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

def slide(new_p, x1, y1, x2, y2, dx =0, dy = 0):
    '''
    dx, dy: velocity factor
    '''
    diff_x, diff_y = new_p- W_SIZE //2
    print('new p', new_p)
    print('old  pos ', x1, y1, x2, y2)
    print('diff', diff_x, diff_y)
    x1 = x1 + diff_x + dx
    y1 = y1 + diff_y - dy
    x2, y2 = x1 + W_SIZE, y1 - W_SIZE
    print('new  pos ', x1, y1, x2, y2)
    return x1, y1, x2, y2

def dist3point(p1,p2,p3):
    return np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

def dist2point(p1,p2):
    return np.linalg.norm(p1 - p2)

def medianPoint(img):
    w,h = img.shape[:2]
    l = []
    for x in range(w):
        for y in range(h):
            try:
                if img[y][x]:
                    l.append([x,y])
            except:
                pass
    return np.median(l, axis = 0)

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
    canny_img = cv2.Canny(img, 50, 200)

    ''' hsv_mask '''
    mask_img = hsv_mask(img)

    ''' 2. Sliding Window '''

    ''' Initialize window starting point '''
    w_size = 35
    startX, startY = .193 , .93 #TODO - not fixed but auto detect starting point
    x1, y1 = int(startX*w_img -w_size/2), int(startY * h_img + w_size/2)
    samep_segs, diffp_segs = [], []
    x2, y2 = x1+w_size, y1-w_size
    prev_angle = 15
    inner_segs = []
    outer_segs = []
    centerP = (w_img//4, h_img//2)

    ''' Slide window '''
    for _ in range(100):
        cv2.rectangle(img, (x1,y1), (x2, y2), (255,255,255))

        w_edge=  canny_img[y2:y1, x1:x2]
        w_mask=  mask_img[y2:y1, x1:x2]
        
        ''' Finding segments '''
        segments = houghLines(w_edge)
        if segments is None:
            targetP = medianPoint(w_mask).astype(int)
            x1, y1, x2, y2 = slide(targetP, x1, y1, x2, y2)
            continue
        segments = segments.reshape(-1,4)

        ''' Filter segments '''
        angles = np.arctan2(-(segments[:,3]-segments[:,1]), segments[:,2]-  segments[:,0]) / np.pi * 180 
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] +=180
        print('angle', angles)
        print('prev angle', prev_angle)
        print('diff angle', np.abs(angles-prev_angle))
        filter_idx = np.abs(angles-prev_angle) < MIN_ANGLE_DIFF # TODO - bright intensity change a little
        segments = segments[filter_idx]
        angles = angles[filter_idx]
        prev_angle = np.mean(angles, axis= 0) if len(angles) else prev_angle # update previous angle
        for s in segments:
            if s[1] < s[3]: # y1 is higher than y2 => swap
                s[0],s[1],s[2],s[3] = s[2],s[3],s[0], s[1]

        ''' Split into 2 group '''
        if len(segments) >=2:
            p = segments[0]
            other = segments[1:]
            dist2p = np.array([np.abs(dist3point(p[:2], p[2:], (op[:2]+op[2:])/2)) for op in other]) # absolute distance from p to other segs
            diffp_segs = other[dist2p > 8]
            samep_segs = other[dist2p < 2]
            samep_segs = np.concatenate((samep_segs.reshape(-1,4), p.reshape(-1,4)), axis= 0)
            print('dist', dist2p)

            # mean_same = np.mean(samep_segs, axis= 0, dtype=int)
            idx_max = np.argmin(samep_segs[:,3]) # y min -> highest
            mean_same = samep_segs[idx_max]
            center_mean_same_global = (mean_same[:2] + mean_same[2:])//2  + [x1,y2]
            radius_same = dist2point(centerP, center_mean_same_global)
            samep_segs_global = samep_segs  + [x1,y2,x1,y2]

            if len(diffp_segs):
                # mean_diff = np.mean(diffp_segs, axis= 0)
                idx_max = np.argmin(diffp_segs[:,3]) # y min -> highest
                mean_diff = diffp_segs[idx_max]
                mean_ = np.mean((mean_same, mean_diff), axis= 0, dtype = int)
                x1, y1, x2, y2 = slide(mean_[2:], x1, y1,x2,y2)

                center_mean_diff_global = (mean_diff[:2] + mean_diff[2:])//2 + [x1,y2]
                radius_diff = dist2point(centerP, center_mean_diff_global)

                diffp_segs_global = diffp_segs  + [x1,y2,x1,y2]
                print('center_dist', radius_diff, radius_same)
                if radius_diff  < radius_same:
                    inner_segs.extend(diffp_segs_global)
                    outer_segs.extend(samep_segs_global)
                else:
                    inner_segs.extend(samep_segs_global)
                    outer_segs.extend(diffp_segs_global)
                
            else:
                x1, y1, x2, y2 = slide(mean_same[2:], x1, y1,x2,y2)
                if len(inner_segs) and len(outer_segs):
                    # dist2center = dist2point((mean_same[:2]+mean_same[2:])//2, centerP) # assume centerp is center of curve but it is not guarantee
                    dist2inner = dist3point(inner_segs[-1][:2], inner_segs[-1][2:], mean_same[:2])
                    dist2outer = dist3point(outer_segs[-1][:2], outer_segs[-1][2:], mean_same[:2])
                    if dist2inner < dist2outer:
                        inner_segs.extend(samep_segs_global)
                    else:
                        outer_segs.extend(samep_segs_global)
                else: # outer seg is more likely detected by Canny than inner seg
                    outer_segs.extend(samep_segs_global)
                    
            # mean_same = np.mean()
            # x1, y1, x2, y2 = slide(segments[0][2:], x1, y1,x2,y2)
            
        elif len(segments) == 1:
            p = segments[0]
            p_global = np.array(p).reshape(-1,4) + [x1, y2, x1, y2]
            # print('p_global', p_global)
            x1, y1, x2, y2 = slide(segments[0][2:], x1, y1,x2,y2)

            if len(inner_segs) and len(outer_segs):
                dist2inner = dist2point(p[:2], inner_segs[-1][2:])
                dist2outer = dist2point(p[:2], outer_segs[-1][2:])
                if dist2inner < dist2outer:
                    inner_segs.extend(p_global)
                else:
                    outer_segs.extend(p_global)
            else: # outer seg is more likely detected by Canny than inner seg
                outer_segs.extend(p_global)
            
        else: # move base on previous inner, outer angle?
            targetP = medianPoint(w_mask).astype(int)
            x1, y1, x2, y2 = slide(targetP, x1, y1, x2, y2)
    
        if debug:
            cv2.imshow('canny', canny_img)
            cv2.imshow('mask', mask_img)
            cv2.imshow('img', img)
            cv2.imshow('w edge', w_edge)
            # cv2.imshow('w mask', w_mask)

            drawSegments(segments, prompt='all segs')
            drawSegments(segments, prompt='filter segs')

            targetP = medianPoint(w_mask).astype(int)
            cv2.circle(w_mask, targetP, color = (125,125,125), radius = 2)
            cv2.imshow('seg median', w_mask)

            # draw inner/outer lines
            viz_segs = np.zeros(img.shape)
            for i in inner_segs:
                cv2.circle(viz_segs, i[:2], color = (0,0, 255), radius = 1 )
                cv2.circle(viz_segs, i[2:], color = (0,0, 255), radius = 1 )
            for i in outer_segs:
                cv2.circle(viz_segs, i[:2], color = (255,0,0), radius = 1 )
                cv2.circle(viz_segs, i[2:], color = (255,0,0), radius = 1 )

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

            cv2.waitKey(0)
            
    return inner_segs, outer_segs
if __name__ == '__main__':

    img = cv2.imread('/home/pronton/SeAH/data/cam4_20230907_143145/320.jpg') # img = cv2.imread(argv[1])
    inner_segs, outer_segs = sliding_window(img)
    
    # save
    save_for_train(inner_segs, 'inner')
    save_for_train(outer_segs, 'outer')