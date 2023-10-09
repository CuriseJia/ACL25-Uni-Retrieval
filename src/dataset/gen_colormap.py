import cv2
import os
import random
import json

def mosaic2(image, step=16):
    h, w, _ = image.shape
    image2 = cv2.resize(image, (w // step, h // step))
    image3 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('test.jpg', image3)


if __name__ == '__main__':
	img = cv2.imread('imagenet/val/n01491361/ILSVRC2012_val_00002969.JPEG', 1)
	temp = mosaic2(img)



# def random_choose(classname):
#     prompt = open('imagenet/prompt.txt', 'r')
#     caption = prompt.readlines()
#     random.sample(range(len(caption)), 1)
#     res = caption[random.sample(range(len(caption)), 1)[0]].replace('{}', classname).strip('\n')
#     return res


# def get_classname():
#     convert={}
#     file = open('imagenet/convert.txt', 'r')
#     lines = file.readlines()
#     for line in lines:
#         folder = line.split(' ')[0]
#         classname = line.split(' ')[1].strip('\n')
#         convert[folder]=classname

#     return convert

# filelist = os.listdir('imagenet-r/')

# info={}
# json_data = json.loads(json.dumps(info))

# for fl in filelist:
#     # print(fl)
#     convert = get_classname()
#     cls = convert[fl]
#     json_data[cls]= []
#     path = 'imagenet/train/' + fl
#     if os.path.isdir(path):
#         imagelist = os.listdir(path)
#         for file in imagelist:
#             image = fl+'/'+file
#             short_caption = random_choose(cls)
#             data = {
#                 'image':image,
#                 'label':'original',
#                 'short_caption':short_caption
#             }
#             json_data[cls].append(data)


# with open('test.json', "w") as file:
#     json.dump(json_data, file)

            
# print(len(temp))

# for fl in filelist:
#     path = 'imagenet-p/val/' + fl
#     # print(path)
#     print()
#     if os.path.isdir(path):
#         imagelist = os.listdir(path)
#         for file in imagelist:
#             img_path = path + '/' + file
#             img = cv2.imread(img_path)
#             res = mosaic2(img)
#             # print(img_path)
#             # res_path = 'test/' + file
#             cv2.imwrite(img_path, res)

# img = cv2.imread('imagenet-p/val/n01440764/ILSVRC2012_val_00007197.JPEG')
# res = mosaic2(img)
# cv2.imwrite('ILSVRC2012_val_00007197-p.JPEG', res)


# filelist = os.listdir('imagenet-r/')

# info = {}
# json_data = json.loads(json.dumps(info))

# original = {}
# packages_data = json.loads(json.dumps(original))

# for fl in filelist:
#     # print(fl)
#     allpackage=[]
#     convert = get_classname()
#     cls = convert[fl]
#     path = 'imagenet/train/' + fl
#     if os.path.isdir(path):
#         imagelist = os.listdir(path)
#         for file in imagelist:
#             image = fl+'/'+file
#             short_caption = random_choose(cls)
#             data = {
#                 'image':image,
#                 'short_caption':short_caption
#             }
#             allpackage.append(data)
#     packages_data[cls] = allpackage

# json_data['original']=packages_data

# with open('test3.json', "w") as file:
#     json.dump(json_data, file)