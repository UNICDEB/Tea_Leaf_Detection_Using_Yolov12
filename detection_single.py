# # Detection for Single Image

# import os
# import time
# import cv2
# import numpy as np
# import glob
# from ultralytics import YOLO

# def detection(frame,model, l1): 
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     bounding_box = results[0]
#     # Extract bounding boxes, classes, names, and confidences
#     boxes = results[0].boxes.xyxy.tolist()
#     classes = results[0].boxes.cls.tolist()
#     names = results[0].names
#     confidences = results[0].boxes.conf.tolist()
#     print("All Confidence value - ", confidences)
#     # for i in range(len(boxes)):
#     #     l1.append(boxes[i])
#     if(len(boxes)>len(l1)):
#             if(len(l1)<=0):
#                 for i in range(len(boxes)):
#                     l1.append(boxes[i])
#             else:
#                 l1.clear()
#                 for i in range(len(boxes)):
#                     l1.append(boxes[i])              
#     return(l1)


# if __name__=="__main__":
    
#     start_time = time.time()
#     detected_image = cv2.imread("Capture_Frame/frame_1596.jpg")
#     l1= []
#     model = YOLO('weights/best.pt')
#     detection(detected_image,model,l1)
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
#     cv2.imwrite('Result_Image/detected_image01.jpg', detected_image)
#     cv2.imshow("YOLOv8 Inference", detected_image)
#     if cv2.waitKey(0) & 0xFF == ord("q"):
#         cv2.destroyAllWindows()

############################################

import os
import time
import cv2
import numpy as np
import glob
from ultralytics import YOLO

def detection(frame,model, l1 , l2): 
    results = model(frame)
    annotated_frame = results[0].plot()
    bounding_box = results[0]
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    print("All Confidence value - ", confidences)
    
    # Clear previous detections from the list
    l1.clear()
    l2.clear()
    
    # Add new detections to the list if their confidence is above 50%
    for box, confidence in zip(boxes, confidences):
        if confidence >= 0.1:
            l1.append(box)
            # Center Point Calcute
            center_x = round((box[0] + box[2]) / 2)
            center_y = round((box[1] + box[3]) / 2)
            l2.append((center_x, center_y))
    
    return l1, l2


if __name__ == "__main__":
    
    start_time = time.time()
    detected_image = cv2.imread("images/result_20251124_133144.jpg")
    l1= []
    l2=[]
    # model = YOLO('Weight_V02/best.pt')
    model = YOLO('Weight/best.pt')
    detection(detected_image, model, l1, l2)
    
    # Draw bounding boxes for detected objects with confidence above 50%
    for box in l1:
        start_point = (round(box[0]), round(box[1]))
        end_point = (round(box[2]), round(box[3]))
        detected_image = cv2.rectangle(detected_image, start_point, end_point, (0,0,255), 3)
    
    # Draw bounding boxes for detected objects with confidence above 50%
    for center in l2:
        center_point = (int(center[0]), int(center[1]))
        cv2.circle(detected_image, center_point, 5, (0, 255, 0), -1)    
            
    print("Final Detected Points - ", l1)
    print("Length of Updated Values - ", len(l1))
    end_time = time.time()
    print(f"Total work time - {(end_time-start_time)}")
    
    print("Center Point - ", l2)
    
    # Save the resulting image
    cv2.imwrite('Result_Image/detected_image01.jpg', detected_image)
    cv2.imshow("YOLOv8 Inference", detected_image)
    
    # Wait for 'q' key press to close the window
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()