from fastai.vision.all import *
import cv2
import argparse

def userarguments():
    parser = argparse.ArgumentParser(description="Parameters for splitting a large image into multiple patch")
    # folders names
    parser.add_argument("--path_image", default="/media/ben/Data_linux/code/calcul/Image_patch2/", type=str,
                        help="folder containing the input image")
    parser.add_argument("--path_output", default="/media/ben/Data_linux/code/unsupervised_learning/dataset/Image_output/", type=str,
                        help="folder containing output image")
    # for image resizing
    parser.add_argument("--path_data", default="/media/ben/Data_linux/code/unsupervised_learning/dataset/", type=str,
                        help="folder containind the subfolders with images")

    return parser.parse_args()

args = userarguments()

path_data = Path(args.path_data)
path_valid =path_data/"valid"
folder_resized = "valid_resized"
path_to_resized = path_data/folder_resized

if not os.path.exists(folder_resized):
    os.mkdir(folder_resized)

def make_patch(input_image_size=8192,ndiv=4,
               path_image=Path(args.path_image),
               image_name="image_patch8192x8192.jpg"):
    """ function to split a large image into patches"""
    im_patch = input_image_size // ndiv
    list_patch = []
    im = cv2.imread((path_image/image_name).as_posix())
    for xi in range(0,ndiv):                 # x is the vertical axi, y is the horizontal axis
        for yj in range(0, ndiv):
            patch = im[im_patch*xi:im_patch*(xi+1), im_patch*yj:im_patch*(yj+1), :]
            patch_name = f"patch_x{xi+1}_y{yj+1}.jpg"
            output_path = (path_image/patch_name).as_posix()
            cv2.imwrite(output_path, patch)
    print(f"done writing images patches")


def rebuild_image(im_size=8192,ndiv=4,
                  path_image=Path(args.path_image),
                  image_name="rebuilt.jpg"):
    """ function to rebuild the image, for control"""
    im_patch = im_size // ndiv
    rebuilt = np.zeros((im_size,im_size,3))
    for xi in range(0,ndiv):
        for yj in range(0,ndiv):
            pat = cv2.imread((path_image/f"patch_x{xi+1}_y{yj+1}.jpg").as_posix())
            rebuilt[im_patch*xi:im_patch*(xi+1), im_patch*yj:im_patch*(yj+1),:] = pat

    out_path = (path_image/image_name).as_posix()
    cv2.imwrite(out_path, rebuilt)
    print("rebuilt done")

if __name__ == "__main__":
    #resize_images(path_valid, dest=path_to_resized, max_size=256, recurse=True)
    make_patch()
