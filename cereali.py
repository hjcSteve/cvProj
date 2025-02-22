
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter, label
import re
import os
import argparse
import shutil
parser = argparse.ArgumentParser(usage='%(prog)s [options]')

parser.add_argument('-o','--save_output', help='Save output images',action='store_true')
parser.add_argument('-p','--prod_img_dir', help='Path to the input image directory')
parser.add_argument('-s','--shelf_img', help='Path to the test image')
parser.add_argument('-t','--test', choices=['a','b','all'],default='',                
                    required=False, help='test to apply')
parser.add_argument('-m','--multi', help='detect multiple products',action='store_true')


args = parser.parse_args()
# Custom validation logic
if args.test == '' and (not args.shelf_img or not args.prod_img_dir):
    print("Error: -p/--prod_img_dir and -s/--shelf_img are required if -t is empty.")
    parser.print_help()
    exit(1)


save_output=args.save_output
test=args.test  
if test=='':
    shelf_img_path=args.shelf_img
    prod_img_dir=args.prod_img_dir
MULTI=args.multi


class Instance:
    matches_number=0
    position:tuple =(0,0)
    width=0
    height=0
    minx=0
    miny=0
    maxx=0
    maxy=0
class ProductInfo:
    name=""
    instances :list[Instance]= []
    
models_dir="models"
scenes_dir="scenes"
MIN_MATCH_COUNT = 40
FLANN_INDEX_KDTREE = 1
LOCAL_MAXIMA_THRESHOLD= 6
GHT_BINS_NUMBER =20
debug=False
found=[]
output_dir="output"

KNN_DISTANCE=0.7

def extractNameFromExtension(filename):
    return re.sub(r"\.[^.]*$", "", filename)


#given a list of ProductInfo print the output
def print_output(products:list[ProductInfo],scene_filename,save_output=True):
    # open the scene image
    output = cv2.imread(scenes_dir+"/"+scene_filename)
    for p in products:
        if len(p.instances)> 0:
            prod_output = np.copy(output)
            print(f'Product {p.name} - {len(p.instances)} found:')
            for (idx,ins) in enumerate(p.instances):
                print(f"\tInstance {idx+1} {{ position: ({ins.position[0]},{ins.position[1]}), width: {ins.width}px, height: {ins.height}px}}")
                # save the output image
                if save_output:
                    #print(f"position: ({ins.position[0]},{ins.position[1]}), width: {ins.width}px, height: {ins.height}px")
                    cv2.rectangle(prod_output, (ins.position[0]-ins.width//2,ins.position[1]-ins.height//2), (ins.position[0]+ins.width//2,ins.position[1]+ins.height//2), (0, 255, 0), 4)
                    folder_path = output_dir+"/"+extractNameFromExtension(scene_filename)
                    if not(os.path.exists(folder_path)):
                        os.makedirs(folder_path)
                    cv2.imwrite(folder_path+"/"+str(p.name)+".jpg",prod_output)
    return  

#given a list of points and return instance
def getInstance(dst,match_number):
    tl=dst[0][0]
    bl=dst[1][0]
    br=dst[2][0]
    tr=dst[3][0]
    instance=Instance()
    minx=(tl[0]+bl[0])//2
    miny=(tl[1]+tr[1])//2
    maxx=(tr[0]+br[0])//2
    maxy=(bl[1]+br[1])//2
    #print(f"minx:{minx} miny:{miny} maxx:{maxx} maxy:{maxy}")
    instance.matches_number=match_number
    width_box = round(maxx - minx)
    height_box = round(maxy - miny)
    x_center_box=round((minx+maxx)//2)
    y_center_box=round((miny+maxy)//2)
    instance.position= (x_center_box,y_center_box)
    instance.width= width_box
    instance.height = height_box
    instance.minx=minx
    instance.miny=miny
    instance.maxx=maxx
    instance.maxy=maxy

    return instance       

def checkCorners(dst):
    tl = dst[0][0] # x y
    bl = dst[1][0]
    br = dst[2][0]
    tr = dst[3][0]

    # only > 0 coordinate
    if(tl[0]<0):
        tl[0]=0
    if(tl[1]<0):
        tl[1]=0
    if(bl[0]<0):
        bl[0]=0
    if(bl[1]<0):
        bl[1]=0
    if(br[0]<0):
        br[0]=0
    if(br[1]<0):
        br[1]=0
    if(tr[0]<0):
        tr[0]=0
    if(tr[1]<0):
        tr[1]=0
    # check if corners are valid
    # x condition 
    # if left x > right x error
    if(tl[0]>tr[0] or bl[0]>br[0]):
        return False
    if(tl[1]>bl[1] or tr[1]>br[1]):
        return False
    if(tl[1]>br[1]):
        return False
    if(tl[0]>br[0]):
        return False
    return True
# given a query and a train image check if color difference is less than threshold
def colorCheck(query_filename,train_filename,dst, color_threshold=100):
    # read the query and the train images in rgb
    try :
        rgb_query = cv2.cvtColor(cv2.imread(models_dir+'/'+query_filename),cv2.COLOR_BGR2RGB)
        rgb_train = cv2.cvtColor(cv2.imread(scenes_dir+'/'+train_filename),cv2.COLOR_BGR2RGB)
    except:
        print(f"Error reading images {models_dir}/{query_filename} and {scenes_dir}/{train_filename}")
        return False
    # split the model into 3 diffent channels of red, green and blue
    r, g, b = cv2.split(rgb_query)
    # save the means of intensity for each channel
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    tl = dst[0][0] # x y
    bl = dst[1][0]
    br = dst[2][0]
    tr = dst[3][0]

    # only > 0 coordinate
    if(tl[0]<0):
        tl[0]=0
    if(tl[1]<0):
        tl[1]=0
    if(br[0]<0):
        br[0]=0
    if(br[1]<0):
        br[1]=0 
    if(tr[0]<0):
        tr[0]=0
    if(tr[1]<0):
        tr[1]=0
    if(bl[0]<0):
        bl[0]=0
    if(bl[1]<0):
        bl[1]=0
    #crop the region of interest of the scene image rgb, y x 
    crop_img = rgb_train[int(tl[1]):int(br[1]),int(tl[0]):int(br[0])]
    r_crop, g_crop, b_crop = cv2.split(crop_img)
    r_crop_mean = np.mean(r_crop)
    g_crop_mean = np.mean(g_crop)
    b_crop_mean = np.mean(b_crop)
    # compute the difference between the means of the channels in the cropped image and the means of the channels in the model image
    r_diff = r_crop_mean - r_mean
    g_diff = g_crop_mean - g_mean
    b_diff = b_crop_mean - b_mean
    # if the difference is less tha a thereshold, the object is the same
    #print(r_diff,g_diff,b_diff)
    color_d = ((r_diff)**2 +(g_diff)**2+(b_diff)**2)**0.5
    if color_d < color_threshold:
        return True
    return False


def offline_phase2(kp_model):
    # 2 - Choose the baricenter as reference point 
    ref_point= calculate_barycenter(kp_model)
    # 3- for every feature pts calculate the joing vectoor to the refrence pts
    joning_vectors=calculate_joining_vectors(keypoints=kp_model,reference=ref_point)
    return joning_vectors

def online_phase2(img_target_shape,kp_target,kp_model,good,joning_vectors):

    K=GHT_BINS_NUMBER
    n_bins=K
    m_bins=K
    accumulator_array = np.zeros((n_bins,m_bins))
    counter=0
    # Create a dictionary to store values with coordinates as keys
    acc_matches  = [[[] for _ in range(m_bins)] for _ in range(n_bins)]
    for m in good:
        model_pt=kp_model[m.queryIdx]
        target_pt=kp_target[m.trainIdx]
        delta_angle= np.radians(target_pt.angle-model_pt.angle)
        x_vector=joning_vectors[m.queryIdx][1]
        y_vector=joning_vectors[m.queryIdx][0]
        delta_size=target_pt.size/model_pt.size   

        x_rotated= x_vector * np.cos(delta_angle) - y_vector* np.sin(delta_angle)
        y_rotated=x_vector * np.sin(delta_angle) +  y_vector * np.cos(delta_angle)

        x=round(target_pt.pt[1]+delta_size*x_rotated)
        y=round(target_pt.pt[0]+delta_size*y_rotated)
        i=y//(img_target_shape[1]//n_bins)
        j=x//(img_target_shape[0]//m_bins)
        if (i>=accumulator_array.shape[0] or j>=accumulator_array.shape[1] or i<=0 or j<=0):
            counter=counter+1
            continue
        accumulator_array[i][j]+=1
        acc_matches[i][j].append(m)

    local_maxima = find_local_maxima_above_threshold(accumulator_array,LOCAL_MAXIMA_THRESHOLD)

    return local_maxima, acc_matches

def detect(product_filename,scene_filename,output_image,multi=False):
    if multi:
        knn_distance=0.8
    else:
        knn_distance=KNN_DISTANCE
    product_info=ProductInfo()
    product_info.name = extractNameFromExtension(product_filename)
    output = output_image.copy()
    instances=[]
    # Load the query and the train images
    img_train = cv2.imread(scenes_dir+'/'+scene_filename ) # trainImage
    img_query = cv2.imread(models_dir+'/'+product_filename) # queryImage
    rgb_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    rgb_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
    
    train_r,train_g,train_b = cv2.split(rgb_train)
    query_r,query_g,query_b = cv2.split(rgb_query)
    # calculate the mens of intensity for each channel
    r_mean = np.mean(query_r)
    g_mean = np.mean(query_g)
    b_mean = np.mean(query_b)
    rgb_means = [r_mean,g_mean,b_mean]
    # the highest mean is the
    highest_mean_index=rgb_means.index(max(rgb_means))
    #print(r_mean,g_mean,b_mean)


    good = []
    if highest_mean_index==0:
        query= query_r
        train= train_r
    elif highest_mean_index==1:
        query= query_g
        train= train_g
    elif highest_mean_index==2:
        query= query_b
        train= train_b

        ###----- STEP 1: FEATURE DETECTION AND DESCRIPTION -----###
    # Creating SIFT object
    sift = cv2.SIFT_create(sigma=1.6)
    kp_query = sift.detect(query)
    kp_train = sift.detect(train)
    kp_query, des_query = sift.compute(query, kp_query)
    kp_train, des_train = sift.compute(train, kp_train)


    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for m,n in flann.knnMatch(des_query,des_train,k=2):
        if m.distance < knn_distance*n.distance:
            good.append(m)
    #print(len(good))


    ####----- STEP 3: HOMOGRAPHY ESTIMATION -----###
    if multi:
        #query === model
        joning_vectors = offline_phase2(kp_query)
        local_maxima, acc_matches=online_phase2(img_target_shape=img_train.shape,kp_target=kp_train,kp_model=kp_query,good=good,joning_vectors=joning_vectors)
        #if len(local_maxima)>0:
            #print("local maxima "+str(len(local_maxima)) +" acc matches "+ str(len(acc_matches)) + " images " + str(product_filename)) 
        for coord in local_maxima:
            # If we have at least 10 matches we find the box of the object
            good=acc_matches[coord[0]][coord[1]]
               # If we have at least 10 matches we find the box of the object
            if len(good)>LOCAL_MAXIMA_THRESHOLD:
                #print(len(good))
                src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)        
                # Calculating homography based on correspondences
                Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                # Apply homography to project corners of the query image into the image
                h,w = img_query.shape[:2]
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                # dts are the corners of the cereal box in the shelf image
                dst = cv2.perspectiveTransform(pts,Homography)  
                # check if the corners are valid
                if checkCorners(dst):
                    if colorCheck(product_filename,scene_filename,dst,50):
                        # Drawing bounding box
                        output = cv2.polylines(output,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
                        # Add to instances for output
                        instance = getInstance(dst,len(good))
                        instances.append(instance)
                    #else:
                        #print("Color check failed")
    else:
        # If we have at least 10 matches we find the box of the object
        if len(good)>MIN_MATCH_COUNT:
            #print(len(good))
            src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)        
            # Calculating homography based on correspondences
            Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            # Apply homography to project corners of the query image into the image
            h,w = img_query.shape[:2]
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dts are the corners of the cereal box in the shelf image
            dst = cv2.perspectiveTransform(pts,Homography)  
            # check if the corners are valid
            if checkCorners(dst):
                if colorCheck(product_filename,scene_filename,dst,50):
                    # Drawing bounding box
                    output = cv2.polylines(output,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
                    # Add to instances for output
                    instance = getInstance(dst,len(good))
                    instances.append(instance)


    product_info.instances=instances
    return product_info, output


def calculate_barycenter(keypoints):
    # Extract the x and y coordinates from the keypoints
    x_coords = [kp.pt[0] for kp in keypoints]
    y_coords = [kp.pt[1] for kp in keypoints]
    # Calculate the barycenter (centroid) coordinates
    barycenter_x = np.mean(x_coords)
    barycenter_y = np.mean(y_coords)
    return (barycenter_x, barycenter_y)
# Function to calculate the joining vectors from keypoints to the barycenter
def calculate_joining_vectors(keypoints, reference):
    joining_vectors = []
    for kp in keypoints:
        # Keypoint coordinates
        keypoint_x, keypoint_y = kp.pt
        # Calculate the vector from the keypoint to the barycenter
        vector_x = reference[0] - keypoint_x
        vector_y = reference[1] - keypoint_y
        # Append the vector to the list
        joining_vectors.append((vector_x, vector_y))
    
    return joining_vectors
def find_local_maxima_above_threshold(arr, threshold):
    # Create a filter that marks the local maxima
    neighborhood_size = 5  # Neighborhood size for comparison
    local_max = maximum_filter(arr, size=neighborhood_size) == arr
    # Exclude the borders of the array
    background = (arr == 0)
    # Apply the threshold condition
    threshold_mask = arr > threshold
    # Combine the local maxima condition with the threshold condition
    maxima_with_threshold = local_max & threshold_mask & ~background
    # Label the local maxima
    labeled, num_features = label(maxima_with_threshold)
    # Extract the coordinates of the local maxima
    maxima_coords = np.column_stack(np.where(labeled > 0))
    return maxima_coords

#check if two instances overlap, if the position(center) is inside the other instance
def checkOverlaps(product_info:list[ProductInfo]):
    #filtr the products with no instances
    for i in range(len(product_info)):
        for j in range(len(product_info)):
            #print("--- check overlaps of "+product_info[i].name+" and "+product_info[j].name)
            for ii,instance1 in enumerate(product_info[i].instances):
                for ij,instance2 in enumerate(product_info[j].instances):
                         #quando sono lo stesso prodotto e lo stesso istanza salata
                        if (ii==ij) and (i==j): continue
                        if (instance1.position[0]>=instance2.minx and instance1.position[0]<=instance2.maxx and instance1.position[1]>=instance2.miny and instance1.position[1]<=instance2.maxy):
                            if instance1.matches_number>instance2.matches_number:
                                product_info[j].instances.remove(instance2)
                            else:
                                product_info[i].instances.remove(instance1)
    # print("overlaps of "+product_info[i].name+" and "+product_info[j].name)
    return


def print_scene(scene,products):
    for scene_name in scene:
        found=[]
        print(f'------------ scena {extractNameFromExtension(scene_name)} ------------')
        rgb_scene = cv2.imread(scenes_dir+'/'+scene_name,cv2.COLOR_BGR2RGB) # trainImage
        for prod_filename in products:
            product_info,output=detect(prod_filename,scene_name,rgb_scene,MULTI)
            found.append(product_info)
        #filter the products with no instances
        found=[x for x in found if x.instances]
        checkOverlaps(found)
        print_output(products=found,scene_filename=scene_name)
    print("done")



# mainnnn
found=[]
# delete the output folder if exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    # create the output folder
    os.makedirs(output_dir)
test_A={
    "scenes": {"e1.png", "e2.png", "e3.png", "e4.png", "e5.png"},
    "products":{"0.jpg", "1.jpg", "11.jpg", "19.jpg", "24.jpg", "26.jpg", "25.jpg"}
}
test_B={
    "scenes": {"m1.png", "m2.png", "m3.png", "m4.png", "m5.png"},
    "products":{"0.jpg", "1.jpg", "11.jpg", "19.jpg", "24.jpg", "26.jpg", "25.jpg"}
}



if test=='a':
    MULTI=False
    print_scene(test_A["scenes"],test_A["products"])

if test=='b':
    MULTI=True
    print_scene(test_B["scenes"],test_B["products"])
if test=='all':
    MULTI=False
    print_scene(test_A["scenes"],test_A["products"])
    MULTI=True
    print_scene(test_B["scenes"],test_B["products"])
if test=='':
    #split it into directory and file name
    scene_filanames = []
    product_filenames = []
    # Extract directory
    scenes_dir = os.path.dirname(shelf_img_path)
    # Extract file name
    scene_filanames.append(os.path.basename(shelf_img_path))
    # Extract directory
    extensions = (".png", ".jpeg", ".jpg")
    models_dir = prod_img_dir
    files = [f for f in os.listdir(prod_img_dir) if f.lower().endswith(extensions)]
    for f in files:
        product_filenames.append(f)
    print_scene(scene_filanames,product_filenames)

