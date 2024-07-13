
from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse,StreamingHttpResponse
from django.template import loader
from django.core.paginator import Paginator
from .models import *
from .forms import *
from .utils import *
import cv2
import os
import time


# Create your views here.
# def save_image(camID,frame):
#     image_dir = f"./images/{camID}"
#     os.makedirs(image_dir, exist_ok=True)
#     frame_count = len(os.listdir(image_dir))
#     # video = cv2.VideoCapture(camID)
#     # while True:
#         # ret, frame = video.read()
#     frame_flip = cv2.flip(frame,1)
#     if (frame_count < 8):
#         image_path = os.path.join(image_dir, f'{frame_count + 1}.jpg')
#         cv2.imwrite(image_path, frame_flip)
#     else:
#         for img in range(2, 9):
#             old_image_path = cv2.imread(os.path.join(image_dir, f'{img}.jpg')) 
#             new_image_path = os.path.join(image_dir, f'{img - 1}.jpg')
#             cv2.imwrite(new_image_path, old_image_path)
#         cv2.imwrite(os.path.join(image_dir, '8.jpg'), frame_flip)
#     #     if not ret:
#     #         break
#     # video.release()
        
def gen(camID):
    images_list = []
    video = cv2.VideoCapture(camID)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while True:
        success, image = video.read()
        # image_dir = f"./images/{camID}"
        # save_image(camID, image)
        # last_image = os.path.join(image_dir, os.listdir(image_dir)[-1])
        # if (len(images_list) < 5):
        #     images_list.append(image)
        # else:
        #     images_list.pop(0)
        #     images_list.append(image)
        # # frame_flip = images_list[-1]
        # text = f"Camera {camID}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # thickness = 2
        # color = (0, 0, 255) 
        # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        # x = frame_flip.shape[1] - text_width - 10 
        # y = text_height + 10  
        # position = (x, y)
        # frame_flip = cv2.putText(frame_flip, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        # frame_flip = draw_bbox(frame_flip)
        # ret, jpeg = cv2.imencode('.jpg', frame_flip)
        # frame = jpeg.tobytes()
        frame = get_flow(image, frame_width, frame_height)
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def check_available_sources(cam_test):
    cap = cv2.VideoCapture(cam_test)
    test, _ = cap.read()
    print("i : "+str(cam_test)+" /// result: "+str(test))
    if test:
        return True
    else:
        return False
def Cam0(request):
    if check_available_sources(0):
        return StreamingHttpResponse(gen(0),
					content_type='multipart/x-mixed-replace; boundary=frame')
def Cam1(request):
    if check_available_sources(1):
        return StreamingHttpResponse(gen(3),
					content_type='multipart/x-mixed-replace; boundary=frame')
    
def Cam2(request):
    if check_available_sources(2):
        return StreamingHttpResponse(gen(5),
					content_type='multipart/x-mixed-replace; boundary=frame')
# def Cam3(request):
#     if check_available_sources(3):
#         return StreamingHttpResponse(gen(3),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
# def Cam4(request):
#     if check_available_sources(4):
#         return StreamingHttpResponse(gen(4),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
# def Cam5(request):
#     if check_available_sources(5):
#         return StreamingHttpResponse(gen(5),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
# def Cam6(request):
#     if check_available_sources(6):
#         return StreamingHttpResponse(gen(6),
# 					content_type='multipart/x-mixed-replace; boundary=frame')


@login_required(login_url='homePage')
def home(request):
    return render(request, 'cameras/home.html')

def homePage(request):
    return render(request, 'cameras/homePage.html')


@login_required(login_url='homePage')
def AlertLogs(request):
    logs = Alert_log.objects.all().order_by('-time')
    paginator = Paginator(logs, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request,'cameras/alertLogs.html', {"page_obj": page_obj})

@login_required(login_url='homePage')
def get_alert_logs(request):
    page_number = request.GET.get('page', 1)
    logs = Alert_log.objects.all().order_by('-time')
    paginator = Paginator(logs, 50)  # Show 50 logs per page

    page_obj = paginator.get_page(page_number)
    logs_list = list(page_obj.object_list.values('camera_number', 'alert', 'time', 'clip_link'))

    return JsonResponse({
        'logs': logs_list,
        'has_next': page_obj.has_next(),
        'has_previous': page_obj.has_previous(),
        'page_number': page_obj.number,
        'total_pages': paginator.num_pages
    })
