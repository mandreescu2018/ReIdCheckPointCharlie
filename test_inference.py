import os
import torch
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from models.mobilenetV2 import MobileNetV2
from processor.engine_reid import do_inference, inference_one_pic
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gallery_path = 'D:/datasets/market1501/bounding_box_test/'
query_path = 'D:/datasets/market1501/query/'

class Person:
    def __init__(self, file_name, query=False):
        # self.file_name = file_name
        if query:
            self.img_path = os.path.join(query_path, file_name)
        else:
            self.img_path = os.path.join(gallery_path, file_name)
        self.person_id = int(file_name.split("_")[0])
        cam_seq_ID = file_name.split("_")[1]
        self.camera_ID = int(cam_seq_ID[1])
    
    def __str__(self):
        return f'(Person {self.person_id} on camera {self.camera_ID})'
    
    

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/mobilenet.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    # model = BuildModel(camera_num, view_num, cfg).to(device)
    model = MobileNetV2(num_classes).to(device)
    
    checkpoint = torch.load(cfg.TEST.WEIGHT)

    model.load_state_dict(checkpoint['model_state_dict'])

    # summary(model=model, 
    #         input_size=(32, 3, 256, 128),
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #         col_width=20,
    #         row_settings=['var_names'])
    
    # img_path1 = 'D:/datasets/market1501/bounding_box_test/0286_c3s1_066967_02.jpg'
    # 286
    # img_path2 = 'D:/datasets/market1501/bounding_box_test/0286_c6s1_067426_02.jpg'
    # img_path2 = 'D:/datasets/market1501/bounding_box_test/0286_c6s1_067526_02.jpg'
    # 
    # 1249
    img_path0 = 'D:/datasets/market1501/bounding_box_test/1249_c5s3_021240_03.jpg'
    img_path1 = 'D:/datasets/market1501/bounding_box_test/1249_c5s3_021365_02.jpg'
    img_path2 = 'D:/datasets/market1501/bounding_box_test/0215_c3s1_044551_01.jpg'
    img_path3 = 'D:/datasets/market1501/bounding_box_test/0001_c5s1_109973_02.jpg'

    img_query1 = 'D:/datasets/market1501/query/0001_c1s1_001051_00.jpg'
    # img_qallery1 = 'D:/datasets/market1501/bounding_box_test/0001_c1s1_001051_03.jpg'
    # img_qallery1 = 'D:/datasets/market1501/bounding_box_test/0001_c1s6_011741_02.jpg'
    img_qallery1 = 'D:/datasets/market1501/bounding_box_test/0001_c1s1_009376_02.jpg'

    img_query1_c3 = 'D:/datasets/market1501/query/0001_c3s1_000551_00.jpg'
    img_qallery1_c3 = 'D:/datasets/market1501/bounding_box_test/0001_c3s3_074344_05.jpg'

    img_query1_c4 = 'D:/datasets/market1501/query/0001_c4s6_000810_00.jpg'
    img_qallery1_c4 = 'D:/datasets/market1501/bounding_box_test/0001_c4s6_000810_06.jpg'

    person_query = Person('0001_c5s1_001426_00.jpg', query=True)
    person_gallery = Person('0001_c5s1_001526_02.jpg')

    img_query = img_query1_c4
    img_gallery = img_qallery1_c4

    feat_query1 = inference_one_pic(
                      model,
                      imgpath=person_query.img_path,
                      device=device)
    
    feat_qallery1 = inference_one_pic(
                      model,
                      imgpath=person_gallery.img_path,
                      device=device)


    # D:\datasets\market1501\query

    # 1077_c5s2_144199_02.jpg

    feat0 = inference_one_pic(
                      model,
                      imgpath=img_path0,
                      device=device)

    feat1 = inference_one_pic(
                      model,
                      imgpath=img_path1,
                      device=device)
    
    feat2 = inference_one_pic(
                      model,
                      imgpath=img_path2,
                      device=device)
    
    feat3 = inference_one_pic(
                      model,
                      imgpath=img_path3,
                      device=device)

    # print(feat1)
    # print(feat2)

    # # Convert to numpy array
    # numpy_array = feat1.cpu().numpy()
    # # Save as .npy file
    # np.savetxt('feat1.npy', numpy_array)
    # # Convert to numpy array
    # numpy_array = feat2.cpu().numpy()
    # # Save as .npy file
    # np.savetxt('feat2.npy', numpy_array)
    # numpy_array = feat3.cpu().numpy()
    # # Save as .npy file
    # np.savetxt('feat3.npy', numpy_array)


    # Calculate distances for each pair of corresponding rows
    differences = feat0 - feat1
    distances = torch.norm(differences, dim=1)
    print(f'Euclidean Distances (0-1): {distances.item()}')
    differences = feat0 - feat2
    distances = torch.norm(differences, dim=1)
    print(f'Euclidean Distances (0-2): {distances.item()}')
    differences = feat0 - feat3
    distances = torch.norm(differences, dim=1)
    print(f'Euclidean Distances (0-3): {distances.item()}')
    differences = feat1 - feat2
    distances = torch.norm(differences, dim=1)
    print(f'Euclidean Distances (1-2): {distances.item()}')

    differences = feat_query1 - feat_qallery1
    distances = torch.norm(differences, dim=1)
    # str_mesage = f'Euclidean Distances ({str(person_query)}-{str(person_gallery)})'
    print(f'Euclidean Distances ({str(person_query)}-{str(person_gallery)}): {distances.item()}')


    exit(0)

    do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 device=device)