# Detection for Image (Various Exp. Frame)
import os
import time
import cv2
import numpy as np
import glob
from ultralytics import YOLO
import pyrealsense2 as rs
import math

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # Start streaming
        cfg=self.pipeline.start(config)
        
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()
        
    def show_intrinsics(self):
        
        depth_intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return depth_intrinsics
    
    def camera_cordinates(self,u,v,ppx,ppy,fx,fy,depth):
        x=((u-ppx) *depth)/(fx)
        y=((v-ppy) *depth)/(fy)
        z=depth
        return x,y,z
    


def detection(frame,model, l1, l2, counter): 
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imwrite(f"Result_Image/res_frame{counter}.jpg", annotated_frame)
    bounding_box = results[0]
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    print("All Confidence value - ", confidences)
    # for i in range(len(boxes)):
    #     l1.append(boxes[i])
    # if(len(boxes)>len(l1)):
    #         if(len(l1)<=0):
    #             for i in range(len(boxes)):
    #                 l1.append(boxes[i])
    #             final = confidences
    #         else:
    #             l1.clear()
    #             for i in range(len(boxes)):
    #                 l1.append(boxes[i])
    #             final = confidences

            
    # return(l1)


################
    # Clear previous detections from the list
    if(len(boxes)>=1):
        # l1.clear()
        for box, confidence in zip(boxes, confidences):
            if confidence >= 0.1:
                l1.append(box)
                # Center Point Calcute
                center_x = round((box[0] + box[2]) / 2)
                center_y = round((box[1] + box[3]) / 2)
                l2.append((center_x, center_y))
    
    return l1, l2
    
    # Add new detections to the list if their confidence is above 50%
    # if(len(boxes)>len(l1)):
    #     l1.clear()
    #     for box, confidence in zip(boxes, confidences):
    #         if confidence >= 0.1:
    #             l1.append(box)
    #             # Center Point Calcute
    #             center_x = round((box[0] + box[2]) / 2)
    #             center_y = round((box[1] + box[3]) / 2)
    #             l2.append((center_x, center_y))
    
    # return l1, l2

# Exp. Image Generator Function
def exp(detected_image):
    # Create a list of exposure values
    exposures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # Create a list of images with different exposures
    exposed_images = []
    for exposure in exposures:
        exposed_image = cv2.convertScaleAbs(detected_image, alpha=exposure, beta=0)
        exposed_images.append(exposed_image)
    # Save the images
    for i, exposed_image in enumerate(exposed_images):
        cv2.imwrite("Exposure_Image/exp_image_{}.jpg".format(i), exposed_image)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__=="__main__":
    
    ## Realsense Camera
    # point = (400, 300)
    # dc = DepthCamera()
    # intr=dc.show_intrinsics()  
    # while True:
    #     ret, depth_frame, color_frame = dc.get_frame()

    #     # # Show distance for a specific point
    #     # cv2.circle(color_frame, point, 4, (0, 0, 255))
    #     # cv2.circle(color_frame,(640,360),4,(0,0,255))
        
    #     cv2.imshow("depth frame", depth_frame)
    #     cv2.imshow("Color frame", color_frame)
    #      # Check if the user pressed the 'c' key
    #     if cv2.waitKey(1) & 0xFF == ord('s'):
    #         cv2.imwrite('Capture_Frame/frame.jpg', color_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    # Start Time
    start_time = time.time()
    counter = 0
    # Read a Image
    detected_image = cv2.imread("Capture_Frame/frame_4038.jpg")
    # Call Exp Function
    exp(detected_image)
    l1= []
    l2=[]
    # Load the Best Model
    model = YOLO('Weight_V02/best.pt')
    path = "Exposure_Image"
    # List the files in the folder.
    files = os.listdir(path)
    # Read all images in the folder.
    images = []
    # Detect  Object from each image and save it to list
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        counter=counter+1
        detection(img,model,l1, l2, counter)
    
    # Draw  Rectangle on Images
    for i in range(len(l1)):
        start_point = (round(l1[i][0]), round(l1[i][1]))
        end_point = (round(l1[i][2]), round(l1[i][3]))
        detected_image=cv2.rectangle(detected_image, start_point, end_point, (0,0,255), 3)
            
    print("Final Detected Points - ", l1)
    print("Length of Updated Values - ", len(l1))
    end_time = time.time()  
    print(f"Total work time - {(end_time-start_time)}")
    # Use a set to remove duplicates and then convert it back to a list
    threshold = 8
    unique_data = []
    data = l2

    for point in data:
        if not unique_data:
            unique_data.append(point)
        else:
            distances = [euclidean_distance(point, p) for p in unique_data]
            if all(d >= threshold for d in distances):
                unique_data.append(point)

    print("Center Points - ",unique_data)
    print("length - ", len(unique_data))
    # cv2.line(detected_image, (640,0), (640, 720), (255, 0, 0), 2)
    # cv2.line(detected_image, (0,360), (1280, 360), (255, 0, 0), 2)
    cv2.imwrite('Result_Image/detected_image02.jpg', detected_image)
    cv2.imshow("YOLOv8 Inference", detected_image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

######################################

# import os
# import time
# import cv2
# import numpy as np
# import glob
# from ultralytics import YOLO
# import pyrealsense2 as rs

# class DepthCamera:
#     def __init__(self):
#         # Configure depth and color streams
#         self.pipeline = rs.pipeline()
#         config = rs.config()

#         # Get device product line for setting a supporting resolution
#         pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
#         pipeline_profile = config.resolve(pipeline_wrapper)
#         device = pipeline_profile.get_device()
#         device_product_line = str(device.get_info(rs.camera_info.product_line))

#         config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#         config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#         # Start streaming
#         cfg=self.pipeline.start(config)
        
#     def get_frame(self):
#         frames = self.pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()

#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         if not depth_frame or not color_frame:
#             return False, None, None
#         return True, depth_image, color_image

#     def release(self):
#         self.pipeline.stop()
        
#     def show_intrinsics(self):
        
#         depth_intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
#         return depth_intrinsics
    
#     def camera_cordinates(self,u,v,ppx,ppy,fx,fy,depth):
#         x=((u-ppx) *depth)/(fx)
#         y=((v-ppy) *depth)/(fy)
#         z=depth
#         return x,y,z
    


# def detection(frame,model, l1, counter): 
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     cv2.imwrite(f"Result_Image/res_frame{counter}.jpg", annotated_frame)
#     bounding_box = results[0]
#     # Extract bounding boxes, classes, names, and confidences
#     boxes = results[0].boxes.xyxy.tolist()
#     classes = results[0].boxes.cls.tolist()
#     names = results[0].names
#     confidences = results[0].boxes.conf.tolist()
#     print("All Confidence value - ", confidences)
    
#     # Clear previous detections from the list
#     l1.clear()
    
#     # Add new detections to the list if their confidence is above 50%
#     for box, confidence in zip(boxes, confidences):
#         if confidence >= 0.1:
#             l1.append(box)
    
#     return l1

# # Exp. Image Generator Function
# def exp(detected_image):
#     # Create a list of exposure values
#     exposures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#     # Create a list of images with different exposures
#     exposed_images = []
#     for exposure in exposures:
#         exposed_image = cv2.convertScaleAbs(detected_image, alpha=exposure, beta=0)
#         exposed_images.append(exposed_image)
#     # Save the images
#     for i, exposed_image in enumerate(exposed_images):
#         cv2.imwrite("Exposure_Image/exp_image_{}.jpg".format(i), exposed_image)



# if __name__=="__main__":
    
#     ## Realsense Camera
#     # point = (400, 300)
#     # dc = DepthCamera()
#     # intr=dc.show_intrinsics()  
#     # while True:
#     #     ret, depth_frame, color_frame = dc.get_frame()

#     #     # # Show distance for a specific point
#     #     # cv2.circle(color_frame, point, 4, (0, 0, 255))
#     #     # cv2.circle(color_frame,(640,360),4,(0,0,255))
        
#     #     cv2.imshow("depth frame", depth_frame)
#     #     cv2.imshow("Color frame", color_frame)
#     #      # Check if the user pressed the 'c' key
#     #     if cv2.waitKey(1) & 0xFF == ord('s'):
#     #         cv2.imwrite('Capture_Frame/frame.jpg', color_frame)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         cv2.destroyAllWindows()
#     #         break
#     # Start Time
#     start_time = time.time()
#     counter = 0
#     # Read a Image
#     detected_image = cv2.imread("Capture_Frame/Color_Frame/frame_1389.jpg")
#     # Call Exp Function
#     exp(detected_image)
#     l1= []
#     # Load the Best Model
#     model = YOLO('weights/best.pt')
#     path = "Exposure_Image"
#     # List the files in the folder.
#     files = os.listdir(path)
#     # Read all images in the folder.
#     images = []
#     # Detect  Object from each image and save it to list
#     for file in files:
#         file_path = os.path.join(path, file)
#         img = cv2.imread(file_path)
#         counter=counter+1
#         detection(img,model,l1, counter)
    
#     # Draw  Rectangle on Images
#     for i in range(len(l1)):
#         start_point = (round(l1[i][0]), round(l1[i][1]))
#         end_point = (round(l1[i][2]), round(l1[i][3]))
#         detected_image=cv2.rectangle(detected_image, start_point, end_point, (0,0,255), 3)
            
#     print("Final Detected Points - ", l1)
#     print("Length of Updated Values - ", len(l1))
#     end_time = time.time()  
#     print(f"Total work time - {(end_time-start_time)}")
#     # cv2.line(detected_image, (640,0), (640, 720), (255, 0, 0), 2)
#     # cv2.line(detected_image, (0,360), (1280, 360), (255, 0, 0), 2)
#     cv2.imwrite('Result_Image/detected_image02.jpg', detected_image)
#     cv2.imshow("YOLOv8 Inference", detected_image)
#     if cv2.waitKey(0) & 0xFF == ord("q"):
#         cv2.destroyAllWindows()