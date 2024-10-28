import os, argparse, sys
# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, "clean_fid")
from glob import glob
# from cleanfid import fid
from source_override.clean_fid.cleanfid import fid
# from source_override.fid import compute_fid, compute_kid

parser = argparse.ArgumentParser()
parser.add_argument("--outfile", type=str, default="/home/xinrui/projects/SelfCascade/SDXL/results/sdxl_selfcascade/pexels_txtilm7b_resized_v2_step700_100/inference_image/laion_wofinetune.txt", help="")
parser.add_argument("--real", type=str, default="/home/xinrui/projects/SelfCascade/SDXL/dataset/ScaleCrafter/images_test", help="")
parser.add_argument("--fake", type=str, default="/home/xinrui/projects/SelfCascade/SDXL/results/sdxl_selfcascade/pexels_txtilm7b_resized_v2_step700_100/inference_image/laion_finetune/upscale_image", help="")
parser.add_argument("--test_file", type=str, default="/home/xinrui/projects/SelfCascade/SDXL/dataset/ScaleCrafter/test.txt", help="")
parser.add_argument("--random_crop", action="store_true", help="")
parser.add_argument("--output_block", default=3, choices=[3, 2], type=int, help="set to 2 if sFID")
parser.add_argument("--n_crops_per_img", default=3, type=int, help="")
parser.add_argument("--nimgs_real", default=None, type=int, help="")

opt = parser.parse_args()


outfile=opt.outfile
real=opt.real
fake=opt.fake
with open(opt.test_file, 'r') as f:
    prompt_files = [line.strip() for line in f.readlines()]


if fake.endswith("txt"):
    # multi fake files
    with open(fake, "r") as f:
        fakes = f.readlines()
        fakes = [f.strip() for f in fakes]
else:
    fakes = [fake]

os.makedirs(os.path.dirname(outfile), exist_ok=True)

for fake in fakes:
    f=open(outfile, 'a')
    print("start fid")
    
    # nfake = os.path.join(fake,  os.path.splitext(filename)[0]+"_upscale.png")
    # nreal = os.path.join(real, os.path.splitext(filename)[0]+".jpg")
    
    sfid = fid.compute_fid(real, fake,
                           mode="center_crop",
                           random_crop=False,
                           output_blocks=[opt.output_block],
                           n_crops_per_img=opt.n_crops_per_img,
                           dataset_res=2048)
    print(f'fid={sfid}')
    print(f'fake path: {fake}', file=f)
    print(f'real path: {real}', file=f)
    print(f'fid={sfid}',file=f)

    print("start kid")
    skid = fid.compute_kid(real, fake,
                           mode="center_crop",
                           random_crop=False,
                           output_blocks=[opt.output_block],
                           n_crops_per_img=opt.n_crops_per_img,
                           dataset_res=2048)
    print(f'kid={skid}')

    print(f'kid={skid}',file=f)
    print(f'\n',file=f)

    f.close()
