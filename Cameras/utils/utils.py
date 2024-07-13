import h5py
import json
import PIL.Image as Image
import numpy as np
import os
from image import *
from model import CANNet2s
from count_head_and_flow import count_head_and_flow
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from matplotlib import cm
from collections import deque
from torchvision import transforms


consecutive_frames_len = 5
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CANNet2s()

model = model.cuda()

checkpoint = torch.load('/content/drive/MyDrive/crowd/People-Flows/weight/fdst.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()
frame_buffer = deque(maxlen=consecutive_frames_len)
# video_path = '/content/drive/MyDrive/crowd/fdst/test_videos/100.mp4'
# output_video_path = '/content/drive/MyDrive/crowd/fdst/test_results/100.mp4'
def get_flow(frame, frame_width, frame_height):
    

    # Open video file
    # cap = cv2.VideoCapture(video_path)




    

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((640, 360))
        img = transform(img).cuda()
        img = Variable(img).unsqueeze(0)
        # print('img type:', img.type)
        frame_buffer.append(img)

        if len(frame_buffer)==consecutive_frames_len:
        #print('len frame buffer:', len(frame_buffer))
            prev_img = frame_buffer[0]

            prev_flow = model(prev_img,img)
            prev_flow_inverse = model(img,prev_img)

            mask_boundry = torch.zeros(prev_flow.shape[2:])
            mask_boundry[0,:] = 1.0
            mask_boundry[-1,:] = 1.0
            mask_boundry[:,0] = 1.0
            mask_boundry[:,-1] = 1.0

            mask_boundry = Variable(mask_boundry.cuda())

            reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

            reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

            overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).data.cpu().numpy()
            original_image_dens_map = (overall * 255).astype(np.uint8)
            # image_to_save = Image.fromarray(original_image_dens_map)
            # image_to_save = image_to_save.resize((640, 360), Image.ANTIALIAS)
            # image_to_save.save('/content/drive/MyDrive/crowd/video_test/frame_dens_test/img.jpg')

            original_image_dens_map = cv2.resize(original_image_dens_map, (frame_width, frame_height))


            prev_flow= prev_flow.data.cpu().numpy()[0]

            # flow_1 = (prev_flow[0] * 255).astype(np.uint8)
            flow_2 = (prev_flow[1] * 255).astype(np.uint8)
            # flow_3 = (prev_flow[2] * 255).astype(np.uint8)
            flow_4 = (prev_flow[3] * 255).astype(np.uint8)
            # flow_5 = (prev_flow[4] * 255).astype(np.uint8)
            flow_6 = (prev_flow[5] * 255).astype(np.uint8)
            # flow_7 = (prev_flow[6] * 255).astype(np.uint8)
            flow_8 = (prev_flow[7] * 255).astype(np.uint8)
            # flow_9 = (prev_flow[8] * 255).astype(np.uint8)



            flow_2 = cv2.resize(flow_2, (frame_width, frame_height))
            flow_4 = cv2.resize(flow_4, (frame_width, frame_height))
            flow_6 = cv2.resize(flow_6, (frame_width, frame_height))
            flow_8 = cv2.resize(flow_8, (frame_width, frame_height))
            #print(flow_2.shape)

            # print(flow_2.shape)
            current_frame = count_head_and_flow(flow_2, flow_4, flow_6, flow_8, original_image_dens_map, frame)
            # current_frame = cv2.convertScaleAbs(current_frame)
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            frame_processed = jpeg.tobytes()
            # cv2_imshow(current_frame)
            # out.write(cv2.cvtColor(np.array(current_frame), cv2.COLOR_RGB2BGR))
            return frame_processed

