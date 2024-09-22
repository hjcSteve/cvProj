import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter, label
import re
import os
import argparse

parser = argparse.ArgumentParser(usage='%(prog)s [options] <product images directory> <shelf image>')

parser.add_argument('-s','--save_output', help='Save output images',action='store_true')
parser.add_argument('prod_img_dir', help='Path to the input image directory')
parser.add_argument('shelf_img', help='Path to the test image')
parser.add_argument('-t','--test', choices=['a','b',''],default='',                
                    required=False, help='test to apply')
args = parser.parse_args()
save_output=args.save_output
shelf_img_path=args.shelf_img
prod_img_dir=args.prod_img_dir
test=args.test  
class Instance:
    position:tuple =(0,0)
    width=0
    height=0
class ProductInfo:
    name=""
    instances :list[Instance]= []
    
test_models_dir="models"
test_scenes_dir="scenes"
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
GHT_BINS_NUMBER =10
debug=False
found=[]
output_dir="output"

def print_output(products:list,scene_name):
    for p in products:
        if len(p.instances)> 0:
            print(f'Product {p.name} - {len(p.instances)} found:')
            for (idx,ins) in enumerate(p.instances):
                print(f"\tInstance {idx+1} {{ position: ({ins.position[0]},{ins.position[1]}), width: {ins.width}px, height: {ins.height}px}}")

    return
def detect(product_name,scene_name,output_image):
    product_info=ProductInfo()
    product_info.name = re.sub(r"\.[^.]*$", "", product_name)
    output = output_image.copy()
    instances=[]
    # Load the query and the train images
    img_train = cv2.imread(test_scenes_dir+'/'+scene_name,0) # trainImage
    img_query = cv2.imread(test_models_dir+'/'+product_name,0) # queryImage
    
    # Creating SIFT object
    sift = cv2.SIFT_create()

    # Detecting Keypoints in the two images
    kp_query = sift.detect(img_query)
    kp_train = sift.detect(img_train)
    if debug:
        # Visualizing the found Keypoints
        img_visualization = cv2.drawKeypoints(img_train,kp_train,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img_visualization)
        plt.show()
        img_visualization = cv2.drawKeypoints(img_query,kp_query,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img_visualization)
        plt.show()

    # Computing the descriptors for each keypoint
    kp_query, des_query = sift.compute(img_query, kp_query)
    kp_train, des_train = sift.compute(img_train, kp_train)

    # Initializing the matching algorithm

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Matching the descriptors
    matches = flann.knnMatch(des_query,des_train,k=2)
    # Keeping only good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
    
    
    
    # If we have at least 10 matches we find the box of the object
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)        
        # Calculating homography based on correspondences
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # Matches mask for visualization of only matches used by RANSAC
        matchesMask = mask.ravel().tolist()
        # Apply homography to project corners of the query image into the image
        h,w = img_query.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)      
        # Drawing bounding box
        output = cv2.polylines(output,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        # Add to instances for output
        instance=Instance()
        width_box = dst[2][0][0] - dst[0][0][0]
        height_box = dst[2][0][1] - dst[0][0][1]
        x_center_box=round(dst[0][0][0]+(width_box)/2)
        y_center_box=round(dst[0][0][1]+(height_box)/2)
        instance.position= (x_center_box,y_center_box)
        instance.width=round(width_box)
        instance.height = round(height_box)
        instances.append(instance)        
    else:
        #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    # Drawing matches
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv2.drawMatches(img_query,kp_query,img_train,kp_train,good,None,**draw_params)
    plt.imshow(img3, 'gray')
    plt.show()
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
      ## OFFLINE PHASE 
def offline_phase(img_model):
    # Creating SIFT object
    sift = cv2.SIFT_create(sigma = 1.6)
    # 1 - Detect kpts and compute the descriptors in the model image 
    kp_model = sift.detect(img_model)
    kp_model, des_model = sift.compute(img_model, kp_model)
    # 2 - Choose the baricenter as reference point 
    ref_point= calculate_barycenter(kp_model)
    # 3- for every feature pts calculate the joing vectoor to the refrence pts
    joning_vectors=calculate_joining_vectors(keypoints=kp_model,reference=ref_point)
    return kp_model, des_model,joning_vectors

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
def online_phase(img_target,des_model,kp_model,joning_vectors):
     # Creating SIFT object
    sift = cv2.SIFT_create(sigma = 1.0)
    # 1 - Detect keypoints and compute descriptors in the target image and initialize an accumulator array, A[Pc]
    kp_target = sift.detect(img_target)
    kp_target, des_target = sift.compute(img_target, kp_target)
    # 2 - Match descriptors between target and model features
    # Initializing the matching algorithm
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Matching the descriptors
    matches = flann.knnMatch(des_model,des_target,k=2)
    # Keeping only good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
    K=GHT_BINS_NUMBER
    n_bins=img_target.shape[1]//K
    m_bins=img_target.shape[0]//K
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
        i=y//K
        j=x//K
        if (i>=accumulator_array.shape[0] or j>=accumulator_array.shape[1] or i<=0 or j<=0):
            counter=counter+1
            continue
        accumulator_array[i][j]+=1
        acc_matches[i][j].append(m)

    local_maxima = find_local_maxima_above_threshold(accumulator_array,5)

    return local_maxima, acc_matches, kp_target


def detectMulti(product_name_full, scene_path,output_image):
    product_info=ProductInfo()
    product_info.name = re.sub(r"\.[^.]*$", "", product_name_full)
    output = output_image.copy()
    instances=[]

    img_train = cv2.imread(scene_path,0) # trainImage

    img_model = cv2.imread(test_models_dir+'/'+product_name_full,0) # queryImage
    kp_model, des_model,joning_vectors = offline_phase(img_model=img_model)
    local_maxima, acc_matches, kp_target=online_phase(img_target=img_train,des_model=des_model,kp_model=kp_model,joning_vectors=joning_vectors)
    for coord in local_maxima:
        # If we have at least 10 matches we find the box of the object
        good=acc_matches[coord[0]][coord[1]]
        #print(len(good))
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_model[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_target[m.trainIdx].pt for m in good ]).reshape(-1,1,2)        
            # Calculating homography based on correspondences
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # Apply homography to project corners of the query image into the image
            h,w = img_model.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M) 
            instance=Instance()
            width_box = dst[2][0][0] - dst[0][0][0]
            height_box = dst[2][0][1] - dst[0][0][1]
            x_center_box=round(dst[0][0][0]+(width_box)/2)
            y_center_box=round(dst[0][0][1]+(height_box)/2)
            instance.position= (x_center_box,y_center_box)
            instance.width=round(width_box)
            instance.height = round(height_box)
            instances.append(instance)        
            # Drawing bounding box
            output = cv2.polylines(output,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        else:
            #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            
    product_info.instances=instances

    return product_info, output
found=[]
if test=='a':
    test_A={
    "scenes": {"e1.png", "e2.png", "e3.png", "e4.png", "e5.png"},
    "products":{"0.jpg", "1.jpg", "11.jpg", "19.jpg", "24.jpg", "26.jpg", "25.jpg"}
    }
    for scene_name in test_A["scenes"]:
        scene_name_without_ext = re.sub(r"\.[^.]*$", "", scene_name)
        print(f'scena {scene_name_without_ext}')
        rgb_scene = cv2.imread(test_scenes_dir+'/'+scene_name,cv2.COLOR_BGR2RGB) # trainImage
        for p in test_A["products"]:
            p_name = re.sub(r"\.[^.]*$", "", p)
            product_info,output=detect(p,scene_name,rgb_scene)
            found.append(product_info)
            if len(product_info.instances)> 0:
                folder_path = output_dir+"/"+str(scene_name_without_ext)
                if not(os.path.exists(folder_path)):
                    os.makedirs(folder_path)
                cv2.imwrite(folder_path+"/"+str(p_name)+".jpg",output)
        print_output(products=found,scene_name=scene_name_without_ext)
    
if test=='b':
    test_B={
        "scenes": {"m1.png", "m2.png", "m3.png", "m4.png", "m5.png"},
        "products":{"0.jpg", "1.jpg", "11.jpg", "19.jpg", "24.jpg", "26.jpg", "25.jpg"}
    }

    for scene_name in test_B["scenes"]:
        scene_name_without_ext = re.sub(r"\.[^.]*$", "", scene_name)
        print(f'scena {scene_name_without_ext}')
        rgb_scene = cv2.imread(test_scenes_dir+'/'+scene_name,cv2.COLOR_BGR2RGB) # trainImage
        for p in test_B["products"]:
            p_name = re.sub(r"\.[^.]*$", "", p)
            product_info,output=detectMulti(p,scene_name,rgb_scene)
            found.append(product_info)
            if save_output:
                if len(product_info.instances)> 0:
                    folder_path = output_dir+"/"+str(scene_name_without_ext)
                    if not(os.path.exists(folder_path)):
                        os.makedirs(folder_path)
                    cv2.imwrite(folder_path+"/"+str(p_name)+".jpg",output)
        print_output(products=found,scene_name=scene_name_without_ext)



shelf_name = re.sub(r"\.[^.]*$", "", shelf_img_path.split("/")[-1])
print(f'scena {shelf_name}')
rgb_scene = cv2.imread(shelf_img_path,cv2.COLOR_BGR2RGB) # trainImage
prod_filenames = os.listdir(prod_img_dir)
for p in prod_filenames:
    p_name = re.sub(r"\.[^.]*$", "", p)
    product_info,output=detectMulti(p,shelf_img_path,rgb_scene)
    found.append(product_info)
    if save_output:
        if len(product_info.instances)> 0:
            folder_path = output_dir+"/"+str(shelf_name)
            if not(os.path.exists(folder_path)):
                os.makedirs(folder_path)
            cv2.imwrite(folder_path+"/"+str(p_name)+".jpg",output)
sorted_found = sorted(found, key=lambda p: int(p.name) if (p.name.isdigit()) else p.name)
print_output(products=sorted_found,scene_name=shelf_name)