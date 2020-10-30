import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
import argparse
from PIL import Image
import utils
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression, scale_coords,
    xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging)

def psuedolabel_generation(output, width, height, pseudo_threshold, boundary_error):
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]

                conf = pred[4]

                if conf.item() > pseudo_threshold:

                    cls = int(pred[5])
                    w = (box[2] - box[0]) / width
                    h = (box[3] - box[1]) / height
                    x = .5 * (box[2] + box[0]) / width
                    y = .5 * (box[3] + box[1]) / height

                    if (x.item() < 1) and (y.item() < 1) and (w.item() < 1) and (h.item() < 1):

                        targets.append([i, cls, x.item(), y.item(), w.item(), h.item(), conf.item()])

                    else:
                        boundary_error += 1

    return np.array(targets), boundary_error

def psuedolabel(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         augment=False,
         pseudo_threshold=.4,
         model=None,
         dataloader=None,
         merge=False):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = utils.torch_utils.select_device(opt.device, batch_size=batch_size)
        merge = opt.merge  # use Merge NMS, save *.txt labels

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['extra'] if opt.task == 'extra' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    count = 1
    uncount = 0
    boundary_error = 0

    for batch_i, (img, targets, paths, shapes) in tqdm(enumerate(dataloader), desc="PseudoLabel", mininterval=0.01):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, _ = model(img, augment=augment)  # inference and training outputs

            # Prediction
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            plabels, boundary_error = psuedolabel_generation(output, width, height, pseudo_threshold, boundary_error)

            if len(plabels) > 0:
                for i in plabels[:,0].astype('int'):

                    idx = np.where(plabels[:,0] == i)[0]
                    save_labels = plabels[idx]
                    labels = save_labels[:,1:-1]

                    file_name = paths[i].replace('images/extra', 'labels/pseudo').replace('jpg','txt').replace('JPG','txt').replace('png','txt').replace('PNG','txt')

                    if (np.sum(np.isnan(labels)) == 0) and (np.sum(np.isinf(labels)) == 0):
                        np.savetxt(file_name, labels, delimiter=' ',fmt=['%d','%4f','%4f','%4f','%4f'])
                        image = Image.open(paths[i])
                        image.save(paths[i].replace('extra', 'pseudo'))
                        count += 1
                        image.close()

                    else:
                        print(file_name)
                        uncount += 1


    print(f'Completed generating {count} pseudo labels.')
    print(f'Eliminated {uncount} images.')
    print(f'Boundary Error: {boundary_error} objects')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='psuedolabel.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)') #./weights/yolov5s.pt #./runs/exp3/weights/best.pt
    parser.add_argument('--data', type=str, default='data/custom.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='extra', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--pseudo_threshold', default=0.4, help='lower bound of class confidence')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')

    opt = parser.parse_args()

    opt.data = check_file(opt.data)  # check file
    print(opt)
    print("Creating pseudo labels...")
    psuedolabel(opt.data,
                opt.weights,
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                opt.augment,
                opt.pseudo_threshold
                )