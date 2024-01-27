import os,shutil
import cv2
# folder="/content/extended/train"
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))
iter=0
folders=os.listdir("data/train/")
for folder in folders:
  files=os.listdir(os.path.join("data/train",folder))
  for photo in files:
    photo=photo[0:-4]
    image=cv2.imread(os.path.join("data/train",folder,photo)+".jpg")
    for i in range(4):
      if os.path.isdir(os.path.join("extended/train",str(iter)+"_"+str(i)))==False:
        os.mkdir(os.path.join("extended/train",str(iter)+"_"+str(i)))
      size=image.shape[0]//2
      cropp=image[size*(i//2):size*(i//2+1),size*(i%2):size*(i%2+1)]
      cv2.imwrite(os.path.join("extended/train",str(iter)+"_"+str(i),photo+".jpg"),cropp)
  iter=iter+1