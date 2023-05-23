import os, shutil

original_dataset_path = './dogsVcats'
copy_train_path = './datasets'

def copy_files(dogORcat_path,start_num,end_num,trainOrval_path):
    
    image_paths = [os.path.join(original_dataset_path,"train",dogORcat_path + '.' + str(i) + '.jpg')
                  for i in range(start_num,end_num)]
    
    target_copy_paths = os.path.join(copy_train_path,trainOrval_path,dogORcat_path)
    
    if not os.path.isdir(target_copy_paths):
        os.makedirs(target_copy_paths)
        
    for image_path in image_paths:
        shutil.copy(image_path, target_copy_paths)

copy_files("dog",0,10000,"train")
copy_files("cat",0,10000,"train")
copy_files("dog",10000,12500,"validation")
copy_files("cat",10000,12500,"validation")

print("훈련데이터 Dog 개수",len(os.listdir('./datasets/train/dog')))
print("훈련데이터 Cat 개수",len(os.listdir('./datasets/train/cat')))
print("검증데이터 Dog 개수",len(os.listdir('./datasets/validation/dog')))
print("검증데이터 Cat 개수",len(os.listdir('./datasets/validation/cat')))