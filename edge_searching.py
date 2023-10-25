import cv2
import numpy as np
MIN_ANGLE_DIFF = 10

def hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0,0,40)
    upper = (40,50,150)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def sliding_window(img):
    pass

def houghLines(img):
        return cv2.HoughLinesP(img, 2, np.pi/180, 10, minLineLength=8, maxLineGap=3)

def preprocessing(img):
    return cv2.bilateralFilter(img, 9, 75, 75) 

def drawSegments(segs, col = (125,125,125), prompt =''):
    viz_box = np.zeros((W_SIZE, W_SIZE))
    for s in segs:
        cv2.line(viz_box, s[:2], s[2:], color=col)
    cv2.imshow(prompt, viz_box)

def slide(new_p, x1, y1, x2, y2, dx = 8, dy = 8):
    '''
    dx, dy: velocity factor
    '''
    diff_x, diff_y = new_p- W_SIZE //2
    x1 = x1 + diff_x + dx
    y1 = y1 + diff_y - dy
    x2, y2 = x1 + W_SIZE, y1 - W_SIZE
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
if __name__ == '__main__':

    img = cv2.imread('/home/pronton/SeAH/data/cam4_20230907_143145/250.jpg') # img = cv2.imread(argv[1])

    ''' 1. Preprocessing '''
    img = cv2.bilateralFilter(img, 9, 75, 75)
    scale = .5
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)
    h_img, w_img = img.shape[:2]

    ''' canny '''
    canny_img = cv2.Canny(img, 50, 200)
    cv2.imshow('canny', canny_img)

    ''' hsv_mask '''
    mask_img = hsv_mask(img)
    cv2.imshow('mask', mask_img)

    ''' 2. Sliding Window '''

    ''' Initialize window starting point '''
    W_SIZE = 35
    startX, startY = .193 , .93 #TODO - not fixed but auto detect starting point
    x1, y1 = int(startX*w_img -W_SIZE/2), int(startY * h_img + W_SIZE/2)
    x2, y2 = x1+W_SIZE, y1-W_SIZE
    prev_angle = 15
    inner_segs = []
    outer_segs = []
    centerP = (w_img//4, h_img//2)

    ''' Slide window '''
    for _ in range(100):
        cv2.rectangle(img, (x1,y1), (x2, y2), (255,255,255))
        cv2.imshow('img', img)

        w_edge=  canny_img[y2:y1, x1:x2]
        w_mask=  mask_img[y2:y1, x1:x2]
        cv2.imshow('w edge', w_edge)
        cv2.imshow('w mask', w_mask)
        
        ''' Finding segments '''
        segments = houghLines(w_edge).reshape(-1,4)
        drawSegments(segments, prompt='all segs')

        ''' Filter segments '''
        angles = np.arctan2(-(segments[:,3]-segments[:,1]), segments[:,2]-  segments[:,0]) / np.pi * 180
        filter_idx = np.abs(angles-prev_angle) < MIN_ANGLE_DIFF
        segments = segments[filter_idx]
        angles = angles[filter_idx]
        prev_angle = np.mean(angles, axis= 0) if len(angles) else prev_angle # update previous angle
        drawSegments(segments, prompt='filter segs')

        ''' Split into 2 group '''
        if len(segments) >=2:
            p = segments[0]
            other = segments[1:]
            dist2p = np.array([np.abs(dist3point(p[:2], p[2:], (op[:2]+op[2:])/2)) for op in other]) # absolute distance from p to other segs
            diffp_segs = other[dist2p > 4]
            samep_segs = other[dist2p < 2]
            samep_segs = np.concatenate((samep_segs.reshape(-1,4), p.reshape(-1,4)), axis= 0)
            print('dist', dist2p)

            drawSegments(samep_segs, prompt = 'same')
            mean_same = np.mean(samep_segs, axis= 0, dtype=int)
            center_mean_same_global = (mean_same[:2] + mean_same[2:])//2  + [x1,y2]
            radius_same = dist2point(centerP, center_mean_same_global)
            samep_segs_global = samep_segs  + [x1,y2,x1,y2]

            if len(diffp_segs):
                drawSegments(diffp_segs, prompt = 'diff')
                mean_diff = np.mean(diffp_segs, axis= 0)
                mean_ = np.mean((mean_same, mean_diff), axis= 0, dtype = int)
                x1, y1, x2, y2 = slide(mean_[2:], x1, y1,x2,y2)

                center_mean_diff_global = (mean_diff[:2] + mean_diff[2:])//2 + [x1,y2]
                radius_diff = dist2point(centerP, center_mean_diff_global)

                diffp_segs_global = diffp_segs  + [x1,y2,x1,y2]

                if radius_diff  < radius_same:
                    inner_segs.extend(diffp_segs_global)
                    outer_segs.extend(samep_segs_global)
                else:
                    inner_segs.extend(samep_segs_global)
                    outer_segs.extend(diffp_segs_global)
                
                # draw inner/outer lines
                viz_segs = np.zeros(img.shape)
                print('same_global', samep_segs_global)
                for i in inner_segs:
                    cv2.circle(viz_segs, i[:2], color = (0,0, 255), radius = 1 )
                    cv2.circle(viz_segs, i[2:], color = (0,0, 255), radius = 1 )
                for i in outer_segs:
                    cv2.circle(viz_segs, i[:2], color = (255,0,0), radius = 1 )
                    cv2.circle(viz_segs, i[2:], color = (255,0,0), radius = 1 )
                cv2.imshow('segs', viz_segs)
                
            else:
                x1, y1, x2, y2 = slide(mean_same[2:], x1, y1,x2,y2)
                if len(inner_segs) and len(outer_segs):
                    dist2inner = dist2point(mean_same[:2], inner_segs[-1][2:])
                    dist2outer = dist2point(mean_same[:2], outer_segs[-1][2:])
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
            print('p_global', p_global)
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
            cv2.circle(w_mask, targetP, color = (125,125,125), radius = 2)
            cv2.imshow('seg median', w_mask)
            x1, y1, x2, y2 = slide(targetP, x1, y1, x2, y2)
            
        cv2.waitKey(0)
    
    # save
    save_for_train(inner_segs, 'inner')
    save_for_train(outer_segs, 'outer')